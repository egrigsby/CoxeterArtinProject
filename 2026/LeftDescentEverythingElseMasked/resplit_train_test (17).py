"""
Recombine train.csv + test.csv (the exhaustive 1998-word left-descent / ShortLex
normal-form dataset copied from ../) and re-split with a smaller training share.

TRAIN_FRAC = 0.3: the model trains on 30% of the words (599) and is tested on the
remaining 70% (1399). Shuffles the combined rows with SEED before splitting, so
the new partition is independent of the original 80/20 one. Overwrites train.csv
and test.csv in place.
"""

import random
import pandas as pd

SEED       = 0
TRAIN_FRAC = 0.3
TRAIN_CSV  = "train.csv"
TEST_CSV   = "test.csv"


def main():
    df = pd.concat([pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)], ignore_index=True)
    assert not df["word"].duplicated().any(), "duplicate words after combining"

    rows = df.to_dict("records")
    random.Random(SEED).shuffle(rows)
    n_train = int(TRAIN_FRAC * len(rows))

    pd.DataFrame(rows[:n_train]).to_csv(TRAIN_CSV, index=False)
    pd.DataFrame(rows[n_train:]).to_csv(TEST_CSV, index=False)
    print(f"Combined {len(rows)} rows -> {n_train} train / {len(rows) - n_train} test "
          f"(TRAIN_FRAC={TRAIN_FRAC}, seed={SEED})")


if __name__ == "__main__":
    main()
