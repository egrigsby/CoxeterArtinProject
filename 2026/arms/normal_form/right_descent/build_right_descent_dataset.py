"""
Build the RIGHT-descent counterpart of the left-descent normal-form experiment.

Reads ../left_descent/train.csv and test.csv (exhaustive ShortLex normal-form words of
length 1..36 in A2~, split 30/70) and rewrites ONLY the labels: the `descents`
column becomes the per-prefix RIGHT descent set (bitmask int, bit j <=> generator
j+1, '-1' at padding). The `word` column and the train/test partition are copied
unchanged, so the two experiments differ in nothing but the descent direction.

Self-contained: embeds the geometric (Tits) representation machinery verbatim
from the left-descent generator (right_descent_path is the original from
"../../shared/descents.py"; left_descent_path is kept for the
smoke-test cross-check).
"""

import ast

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COXETER_MATRIX = [[1, 3, 3],
                  [3, 1, 3],
                  [3, 3, 1]]      # A2~ (all off-diagonal entries = 3)
SOURCE_DIR = "../left_descent"    # experiment whose words/partition are inherited
TRAIN_CSV  = "train.csv"
TEST_CSV   = "test.csv"

# ---------------------------------------------------------------------------
# Geometric (Tits) representation
# ---------------------------------------------------------------------------

def reflection_matrices(coxeter_matrix):
    """
    Build the n reflection matrices of the geometric (Tits) representation.

    Generator s_k (1-indexed) acts on root-coordinate vectors as
        M_k = I - 2 * outer(e_k, B[:, k]),
    where B[i, j] = -cos(pi / m_ij) is the standard symmetric bilinear form.
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
    LEFT descent set of every prefix (matrix of the prefix inverse, accumulated
    by left-multiplication). Kept only for the smoke-test cross-check.
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


def descent_bitmask(descent_set):
    """Encode a set of 1-indexed generators as a bitmask int (bit g-1 <=> generator g)."""
    b = 0
    for g in descent_set:
        b |= 1 << (g - 1)
    return b


# ---------------------------------------------------------------------------
# Relabel
# ---------------------------------------------------------------------------

def relabel(df, matrix, mats, crosscheck_max_len=8):
    desc_cols = []
    for w in df["word"]:
        padded = [int(x) for x in ast.literal_eval(w)]
        word = [x for x in padded if x != 0]
        assert padded[:len(word)] == word, f"padding not at the end: {padded}"

        path = right_descent_path(word, matrix, mats=mats)   # one set per prefix
        # Cross-check against the left-descent logic: D_R(w) = D_L(reverse(w)).
        if len(word) <= crosscheck_max_len:
            for i in range(len(word)):
                assert path[i] == left_descent_path(word[i::-1], matrix, mats=mats)[-1]

        bitmasks = [descent_bitmask(s) for s in path]
        desc_cols.append([str(x) for x in bitmasks + [-1] * (len(padded) - len(word))])
    return desc_cols


def main():
    matrix = np.array(COXETER_MATRIX, dtype=float)
    mats = reflection_matrices(matrix)

    for name in (TRAIN_CSV, TEST_CSV):
        df = pd.read_csv(f"{SOURCE_DIR}/{name}")
        out = pd.DataFrame({"word": df["word"], "descents": relabel(df, matrix, mats)})
        out.to_csv(name, index=False)
        print(f"Wrote {len(out)} rows to {name} (words and split inherited from "
              f"{SOURCE_DIR}/{name}, labels = per-prefix RIGHT descents)")


if __name__ == "__main__":
    main()
