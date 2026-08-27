"""
Split a single generated dataset CSV (word,descents) into train.csv / test.csv
with a 70/30 train/test share.

TRAIN_FRAC = 0.7: the model trains on 70% of the words, tested on the
remaining 30%. Shuffles the rows with SEED before splitting for reproducibility.
"""

import random
import pandas as pd

SEED       = 0
TRAIN_FRAC = 0.7
INPUT_CSV  = "shortlex_left_descents.csv"   # change to your file (or run twice, once per file)
TRAIN_CSV  = "train.csv"
TEST_CSV   = "test.csv"


def main():
    df = pd.read_csv(INPUT_CSV)
    assert not df["word"].duplicated().any(), "duplicate words in input file"

    rows = df.to_dict("records")
    random.Random(SEED).shuffle(rows)
    n_train = int(TRAIN_FRAC * len(rows))

    pd.DataFrame(rows[:n_train]).to_csv(TRAIN_CSV, index=False)
    pd.DataFrame(rows[n_train:]).to_csv(TEST_CSV, index=False)
    print(f"Split {len(rows)} rows from {INPUT_CSV} -> {n_train} train / "
          f"{len(rows) - n_train} test (TRAIN_FRAC={TRAIN_FRAC}, seed={SEED})")


if __name__ == "__main__":
    main()
