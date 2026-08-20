"""
Build the ELEMENT-CLASSIFICATION dataset for the FINITE A2 Coxeter group (S3).

Task: random words over the two generators (adjacent repeats ALLOWED — the
group is finite, so every word of every length is meaningful); the label at
position i is WHICH of the 6 group elements the prefix s_1..s_i equals. This is
the word problem asked directly: the model must track the group state like a
finite automaton, since descent sets do not determine the element.

Element IDs 0..5 are assigned in ShortLex-BFS discovery order:
  0 = e,  1 = s1,  2 = s2,  3 = s1s2,  4 = s2s1,  5 = s1s2s1 (longest element).

Output CSVs (pre-split train.csv / test.csv, header `word,labels`):
  - word:   fixed-length word as a list-string of generator IDs, e.g.
            "['1', '2', '2', ...]" (generators 1..2; length FIXED_LENGTH, so
            there is no padding, but the format matches the descent pipeline).
  - labels: per-prefix element ID (int 0..5), one per position.

Self-contained: embeds the geometric (Tits) representation machinery from the
descent builders. For finite A2 every matrix entry is a multiple of 1/2, so
tuple(round(2x)) is an exact hash key for group elements.
"""

import random
from collections import deque

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COXETER_MATRIX = [[1, 3],
                  [3, 1]]         # finite A2 (= S3, 6 elements)
NUM_WORDS    = 4000               # distinct random words (2^18 = 262144 possible)
FIXED_LENGTH = 18                 # must equal SEQUENCE_LENGTH in config.py
WORD_SEED    = 42                 # RNG for word sampling
SEED         = 0                  # shuffle seed for the train/test split
TRAIN_FRAC   = 0.3
TRAIN_CSV    = "train.csv"
TEST_CSV     = "test.csv"

# ---------------------------------------------------------------------------
# Geometric (Tits) representation — verbatim from the descent builders.
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


def key(P):
    """Exact hash key for a group element's matrix (entries are multiples of 1/2)."""
    return tuple(int(round(2 * x)) for x in P.flatten())


def enumerate_elements(mats):
    """
    All group elements by ShortLex BFS. Returns (id_by_key, words_by_id): the
    element-ID map keyed by matrix hash, and each element's ShortLex normal form
    (id 0 = identity, discovery order = ShortLex order).
    """
    n = len(mats)
    I = np.eye(n)
    id_by_key = {key(I): 0}
    words_by_id = [[]]
    queue = deque([([], I)])
    while queue:
        word, P = queue.popleft()
        for g in range(1, n + 1):
            Q = P @ mats[g - 1]
            k = key(Q)
            if k not in id_by_key:
                id_by_key[k] = len(words_by_id)
                words_by_id.append(word + [g])
                queue.append((word + [g], Q))
    return id_by_key, words_by_id


def element_path(word, mats, id_by_key):
    """Element ID of every prefix of `word` (list of 1-indexed generators)."""
    P = np.eye(len(mats))
    path = []
    for g in word:
        P = P @ mats[g - 1]
        path.append(id_by_key[key(P)])
    return path


# ---------------------------------------------------------------------------
# Smoke tests (run on every build)
# ---------------------------------------------------------------------------

# S3 cross-check: represent s1, s2 as the adjacent transpositions (1 2), (2 3)
# acting on the tuple (1, 2, 3) by composition on the right.

def _perm_mul(p, g):
    """Right-multiply permutation tuple p by generator g (swap positions g-1, g)."""
    q = list(p)
    q[g - 1], q[g] = q[g], q[g - 1]
    return tuple(q)


def run_smoke_tests(words, mats, id_by_key, words_by_id, crosscheck_words=50):
    I = np.eye(len(mats))

    # 1. Group relations: s1^2 = s2^2 = e, (s1 s2)^3 = e.
    assert np.allclose(mats[0] @ mats[0], I) and np.allclose(mats[1] @ mats[1], I)
    P = np.eye(2)
    for _ in range(3):
        P = P @ mats[0] @ mats[1]
    assert np.allclose(P, I), "(s1 s2)^3 != e"

    # 2. Exactly 6 elements, with the expected ShortLex normal forms.
    assert len(words_by_id) == 6, f"expected 6 elements, found {len(words_by_id)}"
    assert words_by_id == [[], [1], [2], [1, 2], [2, 1], [1, 2, 1]], words_by_id

    # 3. Matrix labels match S3 permutation composition on sample words.
    #    Build the same-ID map for permutations by replaying the BFS order.
    perm_id = {}
    for i, w in enumerate(words_by_id):
        p = (1, 2, 3)
        for g in w:
            p = _perm_mul(p, g)
        perm_id[p] = i
    for w in words[:crosscheck_words]:
        p = (1, 2, 3)
        for i, g in enumerate(w):
            p = _perm_mul(p, g)
            expected = perm_id[p]
            got = element_path(w[:i + 1], mats, id_by_key)[-1]
            assert got == expected, f"word {w[:i + 1]}: matrix id {got} != perm id {expected}"

    # 4. All 6 classes appear somewhere in the dataset labels.
    seen = set()
    for w in words:
        seen.update(element_path(w, mats, id_by_key))
    assert seen == set(range(6)), f"classes missing from data: {set(range(6)) - seen}"

    print(f"Smoke tests passed: S3 relations, 6 elements, permutation cross-check on "
          f"{crosscheck_words} words, all 6 classes present in {len(words)} words.")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def main():
    mats = reflection_matrices(COXETER_MATRIX)
    id_by_key, words_by_id = enumerate_elements(mats)

    rng = random.Random(WORD_SEED)
    words_set = set()
    while len(words_set) < NUM_WORDS:
        words_set.add(tuple(rng.choice((1, 2)) for _ in range(FIXED_LENGTH)))
    words = [list(w) for w in sorted(words_set)]   # sort -> deterministic build

    run_smoke_tests(words, mats, id_by_key, words_by_id)

    rows = []
    for word in words:
        path = element_path(word, mats, id_by_key)
        rows.append(([str(x) for x in word], [str(x) for x in path]))

    random.Random(SEED).shuffle(rows)
    n_train = int(TRAIN_FRAC * len(rows))

    for rows_split, out_csv in ((rows[:n_train], TRAIN_CSV), (rows[n_train:], TEST_CSV)):
        df = pd.DataFrame({"word": [r[0] for r in rows_split],
                           "labels": [r[1] for r in rows_split]})
        df.to_csv(out_csv, index=False)
        print(f"Wrote {len(df)} rows to {out_csv} "
              f"(finite A2 element classification, fixed length {FIXED_LENGTH})")


if __name__ == "__main__":
    main()
