"""
Build the MULTIPLICATION datasets for affine A2 (Right: w*s, Left: s*w).

Task: given the ShortLex normal form of an element w and a generator s, the
model must autoregressively generate the ShortLex normal form of the product.
One training row per pair (w, s): w ranges over ALL ShortLex normal-form words
of length 1..MAX_LEN (A2~ has 3*l elements of length l, so MAX_LEN = 36 gives
1998 words) and s over the 3 generators — 5994 rows per side.

Token row:   [w_1 .. w_k, MUL_s, p_1 .. p_m, 0-pad ...]
  where MUL_s = s + 3 (tokens 4..6 mean "now multiply by s") and p is the
  normal form of the product (m = k +- 1; m = 0 exactly when w = [s], where
  the product is the identity). FIXED_LENGTH = 74 = 36 + 1 + 37.
Label row:   -1 on the w positions and on padding; at the MUL position the
  first product letter (or 0 = STOP for the identity product); at the product
  position carrying p_i the next letter p_{i+1}; at the last product letter 0
  (STOP). With causal attention the logits at position i see the row up to i,
  so these aligned labels train next-token prediction of the product with no
  shift — and every supervised label is deterministic, unlike the branching
  next-token LM in ../"Affine A2 Generation".

Both sides use the IDENTICAL base words and the IDENTICAL 80/20 train/test
partition (split by base word w, seed 0, all 3 generator pairs of a word on
the same side), so any Left/Right performance gap is attributable to the
multiplication side alone. Output CSVs (header `word,labels`) are written
into Right/ and Left/. Run this script once, from this folder, before
submitting either training job.

The geometric (Tits) machinery is a verbatim copy from
"../Affine A2/build_left_descent_nf_dataset.py"; products are resolved with a
matrix-key -> normal-form dict in the style of enumerate_elements in
"Runs/Finite A2 Classification/build_element_dataset.py" (A2~ matrices are
integer-exact, so the rounded-int tuple is a safe hash key).
"""

import random
from collections import Counter, deque

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COXETER_MATRIX = [[1, 3, 3],
                  [3, 1, 3],
                  [3, 3, 1]]      # A2~ (all off-diagonal entries = 3)
MAX_LEN      = 36                 # base words w: all normal forms of length 1..MAX_LEN
NF_MAX_LEN   = MAX_LEN + 1        # products reach one letter longer
FIXED_LENGTH = 74                 # 36 + 1 (MUL token) + 37; must equal SEQUENCE_LENGTH
SEED         = 0                  # shuffle seed for the shared train/test split
TRAIN_FRAC   = 0.8
OUT_DIRS     = {"Right": "right", "Left": "left"}

# ---------------------------------------------------------------------------
# Geometric (Tits) representation — verbatim from ../Affine A2's builder
# ---------------------------------------------------------------------------

def reflection_matrices(coxeter_matrix):
    """
    Build the n reflection matrices of the geometric (Tits) representation.

    Generator s_k (1-indexed) acts on root-coordinate vectors as
        M_k = I - 2 * outer(e_k, B[:, k]),
    where B[i, j] = -cos(pi / m_ij) is the standard symmetric bilinear form.
    Entries are integer-exact for A2~ (every m_ij in {1, 3}).
    """
    M = np.asarray(coxeter_matrix, dtype=float)
    n = M.shape[0]
    with np.errstate(divide="ignore"):
        B = -np.cos(np.pi / M)
    I = np.eye(n)
    mats = []
    for k in range(n):
        Mk = I.copy()
        Mk[k, :] -= 2.0 * B[:, k]
        mats.append(Mk)
    return mats


def right_descent_path(word, coxeter_matrix, mats=None, tol=1e-9):
    """
    Right descent set of every prefix of `word` (1-indexed generators, no
    padding). Entry t is the right descent set of the prefix s_1 .. s_{t+1}:
    s_j is a right descent iff column j of the accumulated matrix is a
    negative root image (all entries <= 0, not all zero).
    """
    matrix = np.asarray(coxeter_matrix, dtype=float)
    n = matrix.shape[0]
    if mats is None:
        mats = reflection_matrices(matrix)
    P = np.eye(n)
    path = []
    for g in word:
        P = P @ mats[g - 1]
        descents = set()
        for j in range(n):
            col = P[:, j]
            if col.max() <= tol and col.min() < -tol:
                descents.add(j + 1)
        path.append(descents)
    return path


def matrix_key(P):
    """Rounded-int-tuple hash of an accumulated matrix (integer-exact for A2~)."""
    return tuple(int(round(x)) for x in P.flatten())


# ---------------------------------------------------------------------------
# ShortLex enumeration with an element -> normal-form dict
# ---------------------------------------------------------------------------

def enumerate_normal_forms(n, max_len, mats):
    """
    BFS over group elements (children in increasing generator order, so the
    first word reaching an element is its ShortLex normal form). Returns
      words:     [(word tuple, accumulated matrix)] for lengths 1..max_len,
                 in shortlex order
      nf_by_key: {matrix_key: normal-form word tuple} for every element of
                 length 0..max_len (identity -> ())
    """
    I = np.eye(n)
    nf_by_key = {matrix_key(I): ()}
    queue = deque([((), I)])
    words = []
    while queue:
        word, P = queue.popleft()
        if len(word) == max_len:
            continue
        for g in range(1, n + 1):
            Q = P @ mats[g - 1]
            k = matrix_key(Q)
            if k not in nf_by_key:
                w2 = word + (g,)
                nf_by_key[k] = w2
                words.append((w2, Q))
                queue.append((w2, Q))
    return words, nf_by_key


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------

def make_row(w, s, p):
    """Padded token and label rows for the pair (w, s) with product NF p."""
    k, m = len(w), len(p)
    tokens = list(w) + [s + 3] + list(p)
    labels = [-1] * k
    labels.append(p[0] if m else 0)                    # at the MUL position
    labels += [p[i + 1] for i in range(m - 1)]         # within the product
    if m:
        labels.append(0)                               # STOP at the last letter
    assert len(labels) == len(tokens) <= FIXED_LENGTH
    tokens += [0] * (FIXED_LENGTH - len(tokens))
    labels += [-1] * (FIXED_LENGTH - len(labels))
    return tokens, labels


def build_pairs(words, nf_by_key, mats, side):
    """[(w, s, product NF)] for every base word and generator, one side."""
    pairs = []
    for w, P in words:
        if len(w) > MAX_LEN:
            continue
        for s in range(1, len(mats) + 1):
            Q = (P @ mats[s - 1]) if side == "Right" else (mats[s - 1] @ P)
            pairs.append((w, s, nf_by_key[matrix_key(Q)]))
    return pairs


# ---------------------------------------------------------------------------
# Smoke tests (run on every build)
# ---------------------------------------------------------------------------

def run_smoke_tests(words, nf_by_key, pairs_by_side, mats, matrix):
    n = len(mats)

    # 1. Shell counts: A2~ has exactly 3*l elements of length l >= 1, and the
    #    enumeration reaches NF_MAX_LEN so every product resolves in the dict.
    counts = Counter(len(w) for w, _ in words)
    for l in range(1, NF_MAX_LEN + 1):
        assert counts[l] == 3 * l, f"shell {l}: got {counts[l]}, expected {3 * l}"

    for side, pairs in pairs_by_side.items():
        assert len(pairs) == 3 * sum(3 * l for l in range(1, MAX_LEN + 1))
        n_identity = 0
        for w, s, p in pairs:
            # 2. Length rule: multiplying by a generator changes length by 1.
            assert abs(len(p) - len(w)) == 1 or (len(p) == 0 and w == (s,)), \
                f"{side}: |{p}| vs |{w}|"
            n_identity += len(p) == 0
            # 3. Descent cross-check (independent of the dict): the length
            #    drops iff s is a right/left descent of w. D_L(w) = D_R(w^-1),
            #    and a word for w^-1 is the reverse of w.
            check_word = w if side == "Right" else w[::-1]
            is_descent = s in right_descent_path(check_word, matrix, mats=mats)[-1]
            assert (len(p) < len(w)) == is_descent, f"{side}: {w} * {s} -> {p}"
            # 4. Round-trip: the product's normal form multiplies back to the
            #    product matrix.
            R = np.eye(n)
            for g in w:
                R = R @ mats[g - 1]
            R = (R @ mats[s - 1]) if side == "Right" else (mats[s - 1] @ R)
            Pp = np.eye(n)
            for g in p:
                Pp = Pp @ mats[g - 1]
            assert matrix_key(Pp) == matrix_key(R), f"{side}: {w} * {s} -> {p}"
            # 5. Label decode: walking the labels from the MUL position
            #    reconstructs the product and ends with STOP.
            tokens, labels = make_row(w, s, p)
            assert tokens[len(w)] == s + 3
            decoded, i = [], len(w)
            while labels[i] != 0:
                decoded.append(labels[i])
                i += 1
            assert tuple(decoded) == p, f"{side}: labels decode {decoded} != {p}"
            assert all(x == -1 for x in labels[:len(w)])
            assert all(x == -1 for x in labels[i + 1:])
        assert n_identity == 3, f"{side}: {n_identity} identity products, expected 3"

    print(f"Smoke tests passed: shells 1..{NF_MAX_LEN} exhaustive "
          f"({len(words)} words), and for all {len(pairs_by_side['Right'])} pairs "
          f"per side: length rule, descent cross-check, matrix round-trip, "
          f"label decode.")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def main():
    matrix = np.array(COXETER_MATRIX, dtype=float)
    n = matrix.shape[0]
    mats = reflection_matrices(matrix)

    words, nf_by_key = enumerate_normal_forms(n, NF_MAX_LEN, mats)
    pairs_by_side = {side: build_pairs(words, nf_by_key, mats, side)
                     for side in OUT_DIRS}
    run_smoke_tests(words, nf_by_key, pairs_by_side, mats, matrix)

    # Shared 80/20 split by base word (all 3 generator pairs stay together).
    base_words = [w for w, _ in words if len(w) <= MAX_LEN]
    order = list(range(len(base_words)))
    random.seed(SEED)
    random.shuffle(order)
    n_train = int(TRAIN_FRAC * len(base_words))
    split = {"train": order[:n_train], "test": order[n_train:]}

    for side, out_dir in OUT_DIRS.items():
        pairs = pairs_by_side[side]                    # 3 consecutive rows per word
        for name, idxs in split.items():
            rows = [make_row(*pairs[3 * i + j]) for i in idxs for j in range(3)]
            df = pd.DataFrame({
                "word":   [str([str(t) for t in tokens]) for tokens, _ in rows],
                "labels": [str([str(l) for l in labels]) for _, labels in rows],
            })
            path = f"{out_dir}/{name}.csv"
            df.to_csv(path, index=False)
            print(f"{side}: wrote {len(df)} rows to {path}")


if __name__ == "__main__":
    main()
