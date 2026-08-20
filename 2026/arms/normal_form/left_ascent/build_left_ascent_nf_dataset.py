"""
Build the LEFT-ASCENT counterpart of the "Affine A2" experiment.

Reads ../left_descent/train.csv and test.csv (exhaustive ShortLex normal-form
words of length 1..36 in A2~, split 30/70) and rewrites ONLY the labels: the
`descents` column becomes the per-prefix LEFT ASCENT set (bitmask int, bit j
<=> generator j+1, '-1' at padding). The `word` column and the train/test
partition are copied unchanged, so the two experiments differ in nothing but
descent-vs-ascent labels.

In any Coxeter group l(sw) = l(w) +- 1, so every generator is exactly one of
{left ascent, left descent} of an element: the ascent bitmask is
(2^n - 1) - descent_bitmask. The ascent path is nevertheless recomputed here
from the geometric (Tits) representation (machinery embedded verbatim from
../left_descent/build_left_descent_nf_dataset.py) and cross-checked against the
complement of the inherited CSV labels on every position of every row.

The output column NAME stays `descents` because load_descent_dataset in
Transformer.py hardcodes it; the VALUES are ascents.
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
# Geometric (Tits) representation — verbatim from the source generator
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


def left_descent_path(word, coxeter_matrix, mats=None, tol=1e-9):
    """
    LEFT descent set of every prefix of `word` (a list of 1-indexed generators,
    no padding). s_j is a left descent of a prefix p iff p^{-1}(alpha_j) is a
    negative root; the matrix of p^{-1} = s_i ... s_1 is accumulated by
    LEFT-multiplication (P <- M_g @ P).
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
# Relabel: left descents -> left ascents (complement), cross-checked per position
# ---------------------------------------------------------------------------

def relabel(df, matrix, mats):
    n = len(mats)
    full = (1 << n) - 1
    asc_cols = []
    for w_str, d_str in zip(df["word"], df["descents"]):
        padded = [int(x) for x in ast.literal_eval(w_str)]
        src = [int(x) for x in ast.literal_eval(d_str)]
        word = [x for x in padded if x != 0]
        assert padded[:len(word)] == word, f"padding not at the end: {padded}"

        path = left_descent_path(word, matrix, mats=mats)   # one descent set per prefix
        asc = [full - descent_bitmask(s) for s in path] + [-1] * (len(padded) - len(word))

        # Cross-check against the inherited labels: off padding the ascent
        # bitmask must be the complement of the source descent bitmask.
        for i in range(len(padded)):
            if i < len(word):
                assert asc[i] == full - src[i], (
                    f"word {word} prefix {i + 1}: ascent {asc[i]} != complement of {src[i]}"
                )
            else:
                assert src[i] == -1, f"source padding label {src[i]} != -1"
        asc_cols.append([str(x) for x in asc])
    return asc_cols


def main():
    matrix = np.array(COXETER_MATRIX, dtype=float)
    mats = reflection_matrices(matrix)

    checked = 0
    for name in (TRAIN_CSV, TEST_CSV):
        df = pd.read_csv(f"{SOURCE_DIR}/{name}")
        out = pd.DataFrame({"word": df["word"], "descents": relabel(df, matrix, mats)})
        out.to_csv(name, index=False)
        checked += len(out)
        print(f"Wrote {len(out)} rows to {name} (words and split inherited from "
              f"{SOURCE_DIR}/{name}, labels = per-prefix LEFT ascents)")
    print(f"Smoke tests passed: all {checked} rows position-by-position complement-checked "
          f"against the inherited descent labels.")


if __name__ == "__main__":
    main()
