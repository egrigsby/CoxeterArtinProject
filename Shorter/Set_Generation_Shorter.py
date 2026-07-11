import random
import time
from pathlib import Path

import pandas as pd

from config_shorter import (
    SEQUENCE_LENGTH,
    DATA_SEED,
    CURRICULUM_DIR,
    CURRICULUM_STAGES,
    CURRICULUM_LENGTHS_TO_GENERATE,
    EXAMPLES_PER_LENGTH,
)

# ---------------------------------------------------------------------------
# Basic right-descent code from the original Set_Generation(Right_Descent).py
# ---------------------------------------------------------------------------

def generateRandomString(length: int) -> str:
    CHARACTERS = "abc"
    return "".join(random.choice(CHARACTERS) for _ in range(length))

class Timer:
    def __init__(self):
        self.m_beg = time.perf_counter()
    def elapsed(self) -> float:
        return time.perf_counter() - self.m_beg

def order(s: str) -> int:
    if s == 'a': return 0
    if s == 'b': return 1
    if s == 'c': return 2
    if s == 'd': return 3
    if s == 'e': return 4
    if s == 'f': return 5
    return 6

def RootRefTable(s: str, a: str) -> str:
    Table = [
        ['-', 'd', 'f'],
        ['d', '-', 'e'],
        ['f', 'e', '-'],
        ['b', 'a', '+'],
        ['+', 'c', 'b'],
        ['c', '+', 'a']
    ]
    return Table[order(a)][order(s)]

def InsertChar(t: str, w: str, k: int) -> str:
    if k == 0:
        return t + w
    return w[:k] + t + w[k:]

def MultRight(s: str, w: str) -> str:
    t = s
    lambda_val = s
    k = len(w)
    for i in range(len(w) - 1, -1, -1):
        lambda_val = RootRefTable(w[i], lambda_val)
        if lambda_val == '-':
            return w[:k-1] + w[k:]
        elif lambda_val == '+':
            return InsertChar(t, w, k)
        elif order(lambda_val) < order(w[i]):
            k = i
            t = lambda_val
    return InsertChar(t, w, k)

def isRightDescent(s: str, w: str) -> bool:
    lambda_val = s
    for i in range(len(w) - 1, -1, -1):
        lambda_val = RootRefTable(w[i], lambda_val)
        if lambda_val == '-':
            return True
        elif lambda_val == '+':
            return False
    return False

def GetStepDescents(w: str) -> list:
    """
    Computes the right descent set for every prefix of w.
    For a word of true length L, returns L descent-set strings.
    """
    descents_list = []
    gens = ["a", "b", "c"]

    x = w[0]

    for i in range(1, len(w)):
        descent = ""
        newx = ""
        for j in range(3):
            if gens[j] == w[i]:
                newx = MultRight(gens[j], x)
                if len(newx) < len(x):
                    descent += gens[j]
            else:
                if isRightDescent(gens[j], x):
                    descent += gens[j]
        x = newx
        descents_list.append(descent)

    final_descent = ""
    for j in range(3):
        if isRightDescent(gens[j], x):
            final_descent += gens[j]
    descents_list.append(final_descent)

    return descents_list

def descent_string_to_bitmask(descent_str: str) -> int:
    bit = {'a': 0, 'b': 1, 'c': 2}
    mask = 0
    for ch in descent_str:
        mask |= 1 << bit[ch]
    return mask

def word_to_token_ids(word_str: str) -> list:
    mapping = {'a': 1, 'b': 2, 'c': 3}
    return [mapping[char] for char in word_str]

# ---------------------------------------------------------------------------
# Exact-length curriculum helpers
# ---------------------------------------------------------------------------

def pad_word_and_descents(token_ids: list[int], bitmasks: list[int], fixed_len: int):
    """Pad token IDs with 0 and descent bitmasks with -1 to fixed_len."""
    if len(token_ids) != len(bitmasks):
        raise ValueError("token_ids and bitmasks must have the same true length")
    if len(token_ids) > fixed_len:
        raise ValueError(f"word length {len(token_ids)} exceeds fixed_len={fixed_len}")

    pad_amount = fixed_len - len(token_ids)
    padded_tokens = token_ids + [0] * pad_amount
    padded_descents = bitmasks + [-1] * pad_amount
    return padded_tokens, padded_descents

def make_rows_for_exact_length(length: int, instances: int, fixed_len: int):
    """Generate rows whose true word length is exactly `length`."""
    rows = []
    for i in range(instances):
        if i > 0 and i % 10000 == 0:
            print(f"    processed {i:,} examples for length {length}")

        word = generateRandomString(length)
        token_ids = word_to_token_ids(word)
        bitmasks = [descent_string_to_bitmask(d) for d in GetStepDescents(word)]
        padded_tokens, padded_descents = pad_word_and_descents(token_ids, bitmasks, fixed_len)

        rows.append({
            "word": [str(x) for x in padded_tokens],
            "descents": [str(x) for x in padded_descents],
            "true_length": length,
        })
    return rows

def save_dataframe(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):,} rows to {path}")

def main():
    random.seed(DATA_SEED)
    t = Timer()
    CURRICULUM_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating EXACT-length right-descent curriculum datasets...")
    exact_dfs = []

    # Main curriculum files: one file per exact true length.
    for length in CURRICULUM_LENGTHS_TO_GENERATE:
        print(f"  exact length {length}: generating {EXAMPLES_PER_LENGTH:,} examples")
        rows = make_rows_for_exact_length(length, EXAMPLES_PER_LENGTH, SEQUENCE_LENGTH)
        exact_df = pd.DataFrame(rows)

        output_path = CURRICULUM_DIR / f"exact_len_{length}.csv"
        save_dataframe(exact_df, output_path)
        exact_dfs.append(exact_df)

    # Optional combined file, useful only for inspection/analysis, not for staged training.
    all_df = pd.concat(exact_dfs, ignore_index=True)
    save_dataframe(all_df, CURRICULUM_DIR / "all_exact_curriculum_data.csv")

    print("\nDone. Train/test split is NOT done here; the transformer file splits each exact-length stage.")
    print(f"Total elapsed time: {t.elapsed():.2f} seconds")
    print(f"Curriculum files are in: {CURRICULUM_DIR}")
    print("Expected stage files:")
    for stage in CURRICULUM_STAGES:
        print(f"  {CURRICULUM_DIR / f'exact_len_{stage}.csv'}")

if __name__ == "__main__":
    main()
