import random
import time
import pandas as pd

from config import SEQUENCE_LENGTH

INSTANCES = 10000

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
    Computes the right descent strings step-by-step for every incremental substring.
    For a word of length 20, it returns a list of 20 strings.
    """
    descents_list = []
    gens = ["a", "b", "c"]
    
    # 1. Base case: first character
    x = w[0]
    
    # 2. Iteratively process every substring expansion
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
        
    # 3. Final step descent elements
    final_descent = ""
    for j in range(3):
        if isRightDescent(gens[j], x):
            final_descent += gens[j]
    descents_list.append(final_descent)
    
    return descents_list

def descent_string_to_bitmask(descent_str: str) -> int:
    """
    Encodes a right-descent set string (e.g. 'bc') as a bitmask int matching the
    project convention used by build_descent_dataset.py: bit j set <=> generator
    j+1 is a descent ('a' -> gen 1 -> bit 0, 'b' -> gen 2 -> bit 1,
    'c' -> gen 3 -> bit 2). An empty descent set encodes to 0.
    """
    bit = {'a': 0, 'b': 1, 'c': 2}
    mask = 0
    for ch in descent_str:
        mask |= 1 << bit[ch]
    return mask

def word_to_token_ids(word_str: str) -> list:
    """
    Maps generator characters to categorical integer token IDs.
    'a' -> 1, 'b' -> 2, 'c' -> 3
    """
    mapping = {'a': 1, 'b': 2, 'c': 3}
    return [mapping[char] for char in word_str]

def main():
    # Adjusted configuration per your parameters
    instances = INSTANCES
    length = SEQUENCE_LENGTH
    output_csv = "data.csv"   # drop-in for the Categorical Transformer (set SEQUENCE_LENGTH = 20)

    print(f"Generating {instances} Coxeter words of length {length}...")
    t = Timer()

    # Two aligned columns matching build_descent_dataset.py:
    #   word     -> list-string of token IDs (a->1, b->2, c->3)
    #   descents -> list-string of per-prefix right-descent bitmask ints
    # Every word is exactly `length` long, so there is no padding (no '0' / '-1').
    word_cols, desc_cols = [], []

    for i in range(instances):
        if i > 0 and i % 10000 == 0:
            print(f"Processed {i} words...")

        # 1. Generate random word string over {a, b, c}
        word = generateRandomString(length)

        # 2. Map input string characters into token IDs
        token_ids = word_to_token_ids(word)

        # 3. Fetch step-by-step per-prefix descent sets, then bitmask-encode each
        step_descents = GetStepDescents(word)
        bitmasks = [descent_string_to_bitmask(d) for d in step_descents]

        word_cols.append([str(x) for x in token_ids])
        desc_cols.append([str(x) for x in bitmasks])

    print("Generation Complete! Writing two-column CSV...")
    df = pd.DataFrame({"word": word_cols, "descents": desc_cols})
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} rows to {output_csv} (length {length}, no padding).")
    print(f"Total processing elapsed time: {t.elapsed():.2f} seconds.")

if __name__ == "__main__":
    main()
