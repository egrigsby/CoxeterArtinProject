"""
Build the NEXT-TOKEN (autoregressive LM) dataset over the ShortLex normal-form
language of affine A2.

Reads ../"Affine A2"/train.csv and test.csv (exhaustive ShortLex normal-form
words of length 1..36 in A2~, 1998 words, split 30/70) and keeps the `word`
column and the train/test partition unchanged (same inherit-and-relabel pattern
as the Right-descent folders). Only the labels change: the label at letter
position i (0-based) is the NEXT letter word[i+1], and at the LAST letter it is
0 (STOP — the padding token id doubles as the end-of-word class). Padding
positions are -1.

Because the causal model's logits at position i depend only on s_1..s_i, these
aligned labels train exactly next-token prediction with no shift in the
training code. Since the ShortLex language is prefix-closed and branching,
several successors can be legal at a prefix; the trained model is evaluated by
generate.py (greedy rollout legality), not only by top-1 accuracy.

Output CSVs: train.csv / test.csv, header `word,labels`.
"""

import ast

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SOURCE_DIR = "../../arms/normal_form/left_descent"  # experiment whose words/partition are inherited
TRAIN_CSV  = "train.csv"
TEST_CSV   = "test.csv"

# ---------------------------------------------------------------------------
# Relabel
# ---------------------------------------------------------------------------

def next_token_labels(padded):
    """Per-position next-token label for one padded word (list of ints)."""
    word = [x for x in padded if x != 0]
    assert padded[:len(word)] == word, f"padding not at the end: {padded}"
    labels = [word[i + 1] for i in range(len(word) - 1)]   # next letter
    labels.append(0)                                       # STOP at the last letter
    labels += [-1] * (len(padded) - len(word))             # padding sentinel
    return labels


def run_smoke_tests(df, sample=20):
    for w in df["word"][:sample]:
        padded = [int(x) for x in ast.literal_eval(w)]
        word = [x for x in padded if x != 0]
        labels = next_token_labels(padded)
        assert len(labels) == len(padded)
        # STOP appears exactly once, at the last letter position.
        assert labels[len(word) - 1] == 0 and labels[:len(word)].count(0) == 1
        # The word is reconstructible from its first letter plus the labels.
        rebuilt = [word[0]]
        while labels[len(rebuilt) - 1] != 0:
            rebuilt.append(labels[len(rebuilt) - 1])
        assert rebuilt == word, f"{rebuilt} != {word}"
    print(f"Smoke tests passed on {sample} words.")


def main():
    for name in (TRAIN_CSV, TEST_CSV):
        df = pd.read_csv(f"{SOURCE_DIR}/{name}")
        run_smoke_tests(df)
        labels = []
        for w in df["word"]:
            padded = [int(x) for x in ast.literal_eval(w)]
            labels.append([str(x) for x in next_token_labels(padded)])
        out = pd.DataFrame({"word": df["word"], "labels": labels})
        out.to_csv(name, index=False)
        print(f"Wrote {len(out)} rows to {name} (words and split inherited from "
              f"{SOURCE_DIR}/{name}, labels = next letter, 0 = STOP)")


if __name__ == "__main__":
    main()
