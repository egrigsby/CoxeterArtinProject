"""
Right descent sets for Coxeter groups via the geometric (Tits) representation.

A generator s_j is a right descent of a word w iff w(alpha_j) is a negative root.
We build one reflection matrix per generator from the Coxeter matrix, accumulate
the product while walking the word, and after each letter read off the sign of
each simple root's image.

Self-contained: takes a Coxeter matrix (numpy array, 1 on the diagonal, m_ij off
the diagonal, np.inf for an infinite bond). Generators are 1-indexed (1..n).
"""

import numpy as np


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
