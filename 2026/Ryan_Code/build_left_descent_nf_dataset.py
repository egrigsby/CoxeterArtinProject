"""
Build the LEFT-descent / ShortLex-normal-form dataset for a single Coxeter group.

Self-contained: embeds the geometric (Tits) representation machinery from
"../Data Generation/descents.py" (which stays untouched) plus the new
left-descent variant, so everything this experiment needs lives in this folder.

Dataset: ALL ShortLex normal-form words (the lexicographically-least reduced
word for each element, generator order 1 < 2 < ... < n) of length 1..MAX_LEN.
This is exhaustive: A2~ has 3*l elements of length l, so MAX_LEN = 36 gives
3*36*37/2 = 1998 words. The ShortLex language is prefix-closed — every prefix
of a dataset word is itself a ShortLex normal form.

Output CSVs (pre-split train.csv / test.csv, header `word,descents`), each row
aligned position-by-position with the right-descent pipeline's format:
  - word:     padded word as a list-string of generator IDs, e.g. "['1', '3', '0', ...]"
              (generators 1..n, padded with '0').
  - descents: per-prefix LEFT descent set as a bitmask int (bit j set <=> generator
              j+1 is a left descent of the prefix s_1..s_i). Padding positions are
              '-1'. An empty descent set is 0 (distinct from the -1 padding sentinel).
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
MAX_LEN      = 36                 # enumerate ALL normal-form words of length 1..MAX_LEN
FIXED_LENGTH = 36                 # pad to this; must equal SEQUENCE_LENGTH in config.py
SEED         = 0                  # shuffle seed for the train/test split
TRAIN_FRAC   = 0.8
TRAIN_CSV    = "train.csv"
TEST_CSV     = "test.csv"

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


def right_descent_path(word, coxeter_matrix, mats=None, tol=1e-9):
    """
    Right descent set of every prefix of `word` (a list of 1-indexed generators,
    no padding). Returns a list of sets; entry t is the right descent set of the
    prefix s_1 ... s_{t+1}.

    s_j is a right descent of a prefix p iff p(alpha_j) is a negative root, i.e.
    the j-th column of the accumulated reflection-matrix product has every coord
    <= 0 and is nonzero.

    Pass a prebuilt `mats` (from reflection_matrices) to avoid rebuilding it per
    word when processing a whole dataset.
    """
    if mats is None:
        mats = reflection_matrices(coxeter_matrix)
    n = len(mats)
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
        descents = set()
        for j in range(n):
            col = P[:, j]
            if col.max() <= tol and col.min() < -tol:
                descents.add(j + 1)
        path.append(descents)
    return path


# ---------------------------------------------------------------------------
# ShortLex normal-form enumeration
# ---------------------------------------------------------------------------

def shortlex_words(n, max_len, mats):
    """
    All ShortLex normal-form words of length 1..max_len, in shortlex order.

    BFS over group elements represented by their accumulated reflection matrix
    (entries are integer-exact for A2~, so a rounded-int tuple is a safe hash
    key). The FIFO queue is processed with children generated in increasing
    generator order, so elements are discovered in shortlex order and the first
    word reaching an element is exactly its shortlex-least reduced word. The
    kept set is prefix-closed by construction (every kept word extends a kept
    word).
    """
    def key(P):
        return tuple(int(round(x)) for x in P.flatten())

    I = np.eye(n)
    seen = {key(I)}
    queue = deque([([], I)])
    words = []
    while queue:
        word, P = queue.popleft()
        if len(word) == max_len:
            continue
        for g in range(1, n + 1):
            Q = P @ mats[g - 1]
            k = key(Q)
            if k not in seen:
                seen.add(k)
                w2 = word + [g]
                words.append(w2)
                queue.append((w2, Q))
    return words


# ---------------------------------------------------------------------------
# Smoke tests (run on every build)
# ---------------------------------------------------------------------------

def run_smoke_tests(words, matrix, mats, crosscheck_max_len=8):
    # 1. Shell counts: A2~ has exactly 3*l elements of length l >= 1.
    counts = Counter(len(w) for w in words)
    for l in range(1, MAX_LEN + 1):
        assert counts[l] == 3 * l, f"shell {l}: got {counts[l]}, expected {3 * l}"

    # 2. Prefix-closure: every proper prefix of a kept word is a kept word.
    wordset = {tuple(w) for w in words}
    for w in words:
        for i in range(1, len(w)):
            assert tuple(w[:i]) in wordset, f"prefix {w[:i]} of {w} not in language"

    # 3. Left-descent cross-check against the trusted right-descent logic:
    #    D_L(w) = D_R(w^{-1}), and a word for w^{-1} is the reverse of w.
    for w in words:
        if len(w) > crosscheck_max_len:
            continue
        lpath = left_descent_path(w, matrix, mats=mats)
        for i in range(len(w)):
            expected = right_descent_path(w[i::-1], matrix, mats=mats)[-1]
            assert lpath[i] == expected, (
                f"word {w} prefix {i + 1}: left {lpath[i]} != reversed-right {expected}"
            )

    print(f"Smoke tests passed: shells 1..{MAX_LEN} exhaustive ({len(words)} words), "
          f"prefix-closed, left descents match reversed-right up to length {crosscheck_max_len}.")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def descent_bitmask(descent_set):
    """Encode a set of 1-indexed generators as a bitmask int (bit g-1 <=> generator g)."""
    b = 0
    for g in descent_set:
        b |= 1 << (g - 1)
    return b


def main():
    matrix = np.array(COXETER_MATRIX, dtype=float)
    n = matrix.shape[0]
    mats = reflection_matrices(matrix)        # built once, reused for every word

    words = shortlex_words(n, MAX_LEN, mats)
    run_smoke_tests(words, matrix, mats)

    word_cols, desc_cols = [], []
    for word in words:
        path = left_descent_path(word, matrix, mats=mats)   # list of sets, one per prefix
        bitmasks = [descent_bitmask(s) for s in path]

        pad = FIXED_LENGTH - len(word)
        padded_word = word + [0] * pad
        padded_desc = bitmasks + [-1] * pad

        word_cols.append([str(x) for x in padded_word])
        desc_cols.append([str(x) for x in padded_desc])

    rows = list(zip(word_cols, desc_cols))
    random.Random(SEED).shuffle(rows)
    n_train = int(TRAIN_FRAC * len(rows))

    for rows_split, out_csv in ((rows[:n_train], TRAIN_CSV), (rows[n_train:], TEST_CSV)):
        df = pd.DataFrame({"word": [r[0] for r in rows_split],
                           "descents": [r[1] for r in rows_split]})
        df.to_csv(out_csv, index=False)
        print(f"Wrote {len(df)} rows to {out_csv} "
              f"(group {n} generators, fixed length {FIXED_LENGTH})")


if __name__ == "__main__":
    main()
