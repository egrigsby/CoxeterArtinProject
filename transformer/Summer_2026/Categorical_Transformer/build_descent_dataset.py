"""
Build the two-column descent-prediction dataset for a single Coxeter group.

Output CSV (header `word,descents`), each row aligned position-by-position:
  - word:     padded word as a list-string of generator IDs, e.g. "['1', '3', '0', ...]"
              (generators 1..n, padded with '0'), matching the 2025_Replication format.
  - descents: per-prefix RIGHT descent set as a bitmask int (bit j set <=> generator
              j+1 is a descent), e.g. "['1', '6', '-1', ...]". Padding positions are '-1'
              and are masked out at train time. An empty descent set is 0 (distinct from
              the -1 padding sentinel).

Words are generated over the generators avoiding immediate repeats (locally reduced);
the descent path still correctly handles globally non-reduced elements.
"""

import random
import numpy as np
import pandas as pd

from descents import reflection_matrices, right_descent_path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COXETER_MATRIX = [[1, 3, 3],
                  [3, 1, 3],
                  [3, 3, 1]]      # A2~ (all off-diagonal entries = 3)
NUM_WORDS     = 4000              # number of unique words to generate
MIN_LEN       = 1                 # shortest word
MAX_LEN       = 22                # longest word
FIXED_LENGTH  = 22                # pad/truncate to this; must equal SEQUENCE_LENGTH in config.py
SEED          = 0
OUTPUT_CSV    = "data.csv"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def generate_word(n, length, rng):
    """Random word over generators 1..n avoiding immediate repeats."""
    word, last = [], 0
    for _ in range(length):
        g = rng.randint(1, n)
        while g == last:
            g = rng.randint(1, n)
        word.append(g)
        last = g
    return word


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
    rng = random.Random(SEED)

    seen = set()
    word_cols, desc_cols = [], []
    attempts = 0
    while len(word_cols) < NUM_WORDS and attempts < NUM_WORDS * 50:
        attempts += 1
        length = rng.randint(MIN_LEN, MAX_LEN)
        word = generate_word(n, length, rng)
        key = tuple(word)
        if key in seen:
            continue
        seen.add(key)

        path = right_descent_path(word, matrix, mats=mats)   # list of sets, one per prefix
        bitmasks = [descent_bitmask(s) for s in path]

        pad = FIXED_LENGTH - length
        padded_word = word + [0] * pad
        padded_desc = bitmasks + [-1] * pad

        word_cols.append([str(x) for x in padded_word])
        desc_cols.append([str(x) for x in padded_desc])

    df = pd.DataFrame({"word": word_cols, "descents": desc_cols})
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV} "
          f"(group {matrix.shape[0]} generators, fixed length {FIXED_LENGTH})")


if __name__ == "__main__":
    main()
