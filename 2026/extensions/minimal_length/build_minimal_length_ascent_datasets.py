"""
Build the two minimal-length-word ASCENT datasets (LEFT and RIGHT labels).

Reads the left_descent/ and right_descent/ datasets built alongside this file (ALL minimal-length words of
length 1..16 in A2~, 5259 words, 80/20 split, seed 0) and rewrites ONLY the
labels: the `descents` column becomes the per-prefix LEFT/RIGHT ASCENT set
(bitmask int, bit j <=> generator j+1, '-1' at padding). The `word` column and
the train/test partition are copied unchanged, so each ascent test differs from
its descent counterpart in nothing but the label direction.

In any Coxeter group l(ws) = l(w) +- 1, so every generator is exactly one of
{ascent, descent} of an element: the ascent set is the complement of the
descent set within the generating set, and the ascent bitmask is
(2^n - 1) - descent_bitmask. The ascent path is nevertheless recomputed here
from the geometric (Tits) representation (machinery embedded verbatim from
"build_minimal_length_datasets.py") and cross-checked against
the complement of the inherited CSV labels on every position of every row.

Output (header `word,descents` — the column NAME is kept because
load_descent_dataset in Transformer.py hardcodes it; the VALUES are ascents):
  - left_ascent/train.csv,  left_ascent/test.csv  — per-prefix LEFT  ascent sets
  - right_ascent/train.csv, right_ascent/test.csv — per-prefix RIGHT ascent sets
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COXETER_MATRIX = [[1, 3, 3],
                  [3, 1, 3],
                  [3, 3, 1]]      # A2~ (all off-diagonal entries = 3)
_DIR       = Path(__file__).resolve().parent
SOURCE_DIR = _DIR                             # experiment whose words/partition are inherited
CSV_NAMES  = ("train.csv", "test.csv")
SIDES      = ("left", "right")   # <side>_descent -> <side>_ascent

# ---------------------------------------------------------------------------
# Geometric (Tits) representation — verbatim from
# "../Minimal Length/build_minimal_length_datasets.py".
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
    no padding). s_j is a right descent of a prefix p iff p(alpha_j) is a
    negative root.
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
    LEFT descent set of every prefix of `word`. s_j is a left descent of a
    prefix p iff p^{-1}(alpha_j) is a negative root; the matrix of
    p^{-1} = s_i ... s_1 is accumulated by LEFT-multiplication.
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


def descent_bitmask(descent_set):
    """Encode a set of 1-indexed generators as a bitmask int (bit g-1 <=> generator g)."""
    b = 0
    for g in descent_set:
        b |= 1 << (g - 1)
    return b


# ---------------------------------------------------------------------------
# Relabel: descents -> ascents (complement), cross-checked per position
# ---------------------------------------------------------------------------

def relabel_to_ascents(df, matrix, mats, path_fn):
    n = len(mats)
    full = (1 << n) - 1
    asc_cols = []
    for w_str, d_str in zip(df["word"], df["descents"]):
        padded = [int(x) for x in ast.literal_eval(w_str)]
        src = [int(x) for x in ast.literal_eval(d_str)]
        word = [x for x in padded if x != 0]
        assert padded[:len(word)] == word, f"padding not at the end: {padded}"

        path = path_fn(word, matrix, mats=mats)          # descent set per prefix
        asc = [full - descent_bitmask(s) for s in path] + [-1] * (len(padded) - len(word))

        # Cross-check against the inherited labels: off padding the ascent
        # bitmask must be the complement of the source descent bitmask, and
        # padding must stay -1 on both sides.
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
    path_fns = {"left": left_descent_path, "right": right_descent_path}

    checked = 0
    for side in SIDES:
        out_dir = _DIR / f"{side}_ascent"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in CSV_NAMES:
            src = SOURCE_DIR / f"{side}_descent" / name
            df = pd.read_csv(src)
            out = pd.DataFrame({"word": df["word"],
                                "descents": relabel_to_ascents(df, matrix, mats, path_fns[side])})
            out.to_csv(out_dir / name, index=False)
            checked += len(out)
            print(f"Wrote {len(out)} rows to {out_dir / name} (words and split inherited "
                  f"from {src}, labels = per-prefix {side.upper()} ascents)")
    print(f"Smoke tests passed: all {checked} rows position-by-position complement-checked "
          f"against the inherited descent labels.")


if __name__ == "__main__":
    main()
