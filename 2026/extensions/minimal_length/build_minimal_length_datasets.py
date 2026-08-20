"""
Build the two minimal-length-word descent datasets (LEFT and RIGHT labels).

Dataset: ALL minimal-length (globally reduced) words of length 1..MAX_LEN in the
configured Coxeter group — every reduced expression of every element, not just a
normal form. A word w·s is reduced iff s is not a right descent of w, so the
language is enumerated by BFS and is prefix-closed by construction: every prefix
of a dataset word is itself a minimal-length word. For A2~ with MAX_LEN = 16
this gives 5259 words (3*l elements of length l, with multiple reduced words per
element from length 3 on).

Both tests use the SAME words and the SAME train/test partition (one shuffle,
seed 0, 80/20); only the labels differ:
  - left_descent/train.csv,  left_descent/test.csv  — per-prefix LEFT  descent sets
  - right_descent/train.csv, right_descent/test.csv — per-prefix RIGHT descent sets

Self-contained: embeds the geometric (Tits) representation machinery from
"../Data Generation/descents.py" (which stays untouched) plus the left-descent
variant, matching "../UnreversedTransformer/Affine A2/build_left_descent_nf_dataset.py".

Output CSV format (header `word,descents`), each row aligned position-by-position:
  - word:     padded word as a list-string of generator IDs, e.g. "['1', '3', '0', ...]"
              (generators 1..n, padded with '0').
  - descents: per-prefix descent set as a bitmask int (bit j set <=> generator
              j+1 is a descent of the prefix s_1..s_i). Padding positions are
              '-1'. An empty descent set is 0 (distinct from the -1 padding sentinel).
"""

import random
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COXETER_MATRIX = [[1, 3, 3],
                  [3, 1, 3],
                  [3, 3, 1]]      # A2~ (all off-diagonal entries = 3)
MAX_LEN      = 16                 # enumerate ALL minimal-length words of length 1..MAX_LEN
FIXED_LENGTH = 16                 # pad to this; must equal SEQUENCE_LENGTH in each config.py
SEED         = 0                  # shuffle seed for the train/test split
TRAIN_FRAC   = 0.8
_DIR         = Path(__file__).resolve().parent
OUT_DIRS     = {"left": _DIR / "left_descent", "right": _DIR / "right_descent"}

# ---------------------------------------------------------------------------
# Geometric (Tits) representation — reflection_matrices and right_descent_path
# are verbatim copies from "../Data Generation/descents.py".
# ---------------------------------------------------------------------------

def reflection_matrices(coxeter_matrix):
    """
    Build the n reflection matrices of the geometric (Tits) representation.

    Generator s_k (1-indexed) acts on root-coordinate vectors as
        M_k = I - 2 * outer(e_k, B[:, k]),
    where B[i, j] = -cos(pi / m_ij) is the standard symmetric bilinear form.
    m_ij = inf falls out correctly (pi/inf = 0 -> -cos(0) = -1) and the diagonal
    m_ii = 1 gives -cos(pi) = 1.

    Entries are integer-exact when every m_ij is in {2, 3, inf}; otherwise they
    are floats and descent signs are read off with a tolerance.
    """
    M = np.asarray(coxeter_matrix, dtype=float)
    n = M.shape[0]
    with np.errstate(divide="ignore"):      # pi / inf -> 0 cleanly
        B = -np.cos(np.pi / M)
    I = np.eye(n)
    mats = []
    for k in range(n):
        Mk = I.copy()
        Mk[k, :] -= 2.0 * B[:, k]           # s_k(alpha_i) = alpha_i - 2 B[i,k] alpha_k
        mats.append(Mk)
    return mats


def right_descents(P, n, tol=1e-9):
    """Right descent set read off the accumulated matrix P of a word."""
    descents = set()
    for j in range(n):
        col = P[:, j]
        if col.max() <= tol and col.min() < -tol:
            descents.add(j + 1)
    return descents


def right_descent_path(word, coxeter_matrix, mats=None, tol=1e-9):
    """
    Right descent set of every prefix of `word` (a list of 1-indexed generators,
    no padding). Returns a list of sets; entry t is the right descent set of the
    prefix s_1 ... s_{t+1}.

    s_j is a right descent of a prefix p iff p(alpha_j) is a negative root, i.e.
    the j-th column of the accumulated reflection-matrix product has every coord
    <= 0 and is nonzero.
    """
    if mats is None:
        mats = reflection_matrices(coxeter_matrix)
    n = len(mats)
    P = np.eye(n)
    path = []
    for g in word:
        P = P @ mats[g - 1]
        path.append(right_descents(P, n, tol))
    return path


def left_descent_path(word, coxeter_matrix, mats=None, tol=1e-9):
    """
    LEFT descent set of every prefix of `word` (a list of 1-indexed generators,
    no padding). Returns a list of sets; entry t is the left descent set of the
    prefix s_1 ... s_{t+1}.

    s_j is a left descent of a prefix p iff p^{-1}(alpha_j) is a negative root.
    The matrix of p^{-1} = s_i ... s_1 is accumulated by LEFT-multiplication
    (P <- M_g @ P); the sign read-off on columns is then identical to the
    right-descent case.
    """
    if mats is None:
        mats = reflection_matrices(coxeter_matrix)
    n = len(mats)
    P = np.eye(n)
    path = []
    for g in word:
        P = mats[g - 1] @ P
        path.append(right_descents(P, n, tol))
    return path


# ---------------------------------------------------------------------------
# Minimal-length word enumeration
# ---------------------------------------------------------------------------

def minimal_length_words(n, max_len, mats):
    """
    All minimal-length (reduced) words of length 1..max_len, in shortlex order.

    BFS over words: w·s_g is reduced iff g is not a right descent of w, so
    extending only by non-descent generators enumerates exactly the reduced
    words. The kept set is prefix-closed by construction.
    """
    I = np.eye(n)
    queue = deque([([], I)])
    words = []
    while queue:
        word, P = queue.popleft()
        if len(word) == max_len:
            continue
        blocked = right_descents(P, n)
        for g in range(1, n + 1):
            if g in blocked:
                continue
            w2 = word + [g]
            Q = P @ mats[g - 1]
            words.append(w2)
            queue.append((w2, Q))
    return words


# ---------------------------------------------------------------------------
# Smoke tests (run on every build)
# ---------------------------------------------------------------------------

def run_smoke_tests(words, matrix, mats, brute_max_len=8):
    n = len(mats)

    def key(P):
        return tuple(int(round(x)) for x in P.flatten())

    def elem_key(word):
        P = np.eye(n)
        for g in word:
            P = P @ mats[g - 1]
        return key(P)

    # 1. Element counts: A2~ has exactly 3*l elements of length l >= 1, and every
    #    element of length l must be reached by at least one kept word of length l.
    by_len = {}
    for w in words:
        by_len.setdefault(len(w), set()).add(elem_key(w))
    for l in range(1, MAX_LEN + 1):
        assert len(by_len[l]) == 3 * l, f"length {l}: {len(by_len[l])} elements, expected {3 * l}"

    # 2. Independent brute-force check up to brute_max_len: BFS the group to get
    #    true element lengths, then verify {kept words} == {words over 1..n whose
    #    element length equals the word length} — no descent logic involved.
    dist = {key(np.eye(n)): 0}
    frontier = [np.eye(n)]
    for l in range(1, brute_max_len + 1):
        nxt = []
        for P in frontier:
            for g in range(n):
                Q = P @ mats[g]
                k = key(Q)
                if k not in dist:
                    dist[k] = l
                    nxt.append(Q)
        frontier = nxt

    kept = {tuple(w) for w in words if len(w) <= brute_max_len}
    brute = set()
    stack = [([], np.eye(n))]
    while stack:
        word, P = stack.pop()
        if word and dist[key(P)] == len(word):
            brute.add(tuple(word))
        if len(word) < brute_max_len:
            for g in range(1, n + 1):
                stack.append((word + [g], P @ mats[g - 1]))
    assert kept == brute, "kept words != brute-force reduced words up to brute_max_len"

    # 3. Prefix-closure: every proper prefix of a kept word is a kept word.
    wordset = {tuple(w) for w in words}
    for w in words:
        for i in range(1, len(w)):
            assert tuple(w[:i]) in wordset, f"prefix {w[:i]} of {w} not in language"

    # 4. Left-descent cross-check against the trusted right-descent logic:
    #    D_L(w) = D_R(w^{-1}), and a word for w^{-1} is the reverse of w.
    for w in words:
        if len(w) > brute_max_len:
            continue
        lpath = left_descent_path(w, matrix, mats=mats)
        for i in range(len(w)):
            expected = right_descent_path(w[i::-1], matrix, mats=mats)[-1]
            assert lpath[i] == expected, (
                f"word {w} prefix {i + 1}: left {lpath[i]} != reversed-right {expected}"
            )

    print(f"Smoke tests passed: {len(words)} minimal-length words (lengths 1..{MAX_LEN}), "
          f"3l elements per shell, brute-force-verified and prefix-closed up to length "
          f"{brute_max_len}, left descents match reversed-right.")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def descent_bitmask(descent_set):
    """Encode a set of 1-indexed generators as a bitmask int (bit g-1 <=> generator g)."""
    b = 0
    for g in descent_set:
        b |= 1 << (g - 1)
    return b


def encode(word, path):
    pad = FIXED_LENGTH - len(word)
    padded_word = word + [0] * pad
    padded_desc = [descent_bitmask(s) for s in path] + [-1] * pad
    return [str(x) for x in padded_word], [str(x) for x in padded_desc]


def main():
    matrix = np.array(COXETER_MATRIX, dtype=float)
    n = matrix.shape[0]
    mats = reflection_matrices(matrix)        # built once, reused for every word

    words = minimal_length_words(n, MAX_LEN, mats)
    run_smoke_tests(words, matrix, mats)

    # One shuffle and one split, shared by both labelings.
    words = list(words)
    random.Random(SEED).shuffle(words)
    n_train = int(TRAIN_FRAC * len(words))
    splits = {"train.csv": words[:n_train], "test.csv": words[n_train:]}

    path_fns = {"left": left_descent_path, "right": right_descent_path}
    for label, path_fn in path_fns.items():
        out_dir = OUT_DIRS[label]
        out_dir.mkdir(parents=True, exist_ok=True)
        for csv_name, split_words in splits.items():
            rows = [encode(w, path_fn(w, matrix, mats=mats)) for w in split_words]
            df = pd.DataFrame({"word": [r[0] for r in rows],
                               "descents": [r[1] for r in rows]})
            out_csv = out_dir / csv_name
            df.to_csv(out_csv, index=False)
            print(f"Wrote {len(df)} rows to {out_csv} "
                  f"({label} descents, {n} generators, fixed length {FIXED_LENGTH})")


if __name__ == "__main__":
    main()
