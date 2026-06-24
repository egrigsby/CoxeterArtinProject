import random
import time
import torch
import numpy as np

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

def string_to_soft_vector(descent_str: str) -> list:
    """
    Converts a descent string (e.g., 'bc') into a soft probability 
    vector over individual generators that sums to 1.0.
    """
    vector = [0.0, 0.0, 0.0]
    active_count = len(descent_str)
    
    if active_count == 0:
        return vector # [0.0, 0.0, 0.0] if empty
        
    fraction = 1.0 / active_count
    if 'a' in descent_str: vector[0] = fraction
    if 'b' in descent_str: vector[1] = fraction
    if 'c' in descent_str: vector[2] = fraction
    
    return vector

def word_to_token_ids(word_str: str) -> list:
    """
    Maps generator characters to categorical integer token IDs.
    'a' -> 1, 'b' -> 2, 'c' -> 3
    """
    mapping = {'a': 1, 'b': 2, 'c': 3}
    return [mapping[char] for char in word_str]

def main():
    # Adjusted configuration per your parameters
    instances = 50000
    length = 20
    
    print(f"Generating {instances} Coxeter words of length {length}...")
    t = Timer()
    
    # Initialize empty numpy structures for efficiency before moving to torch
    all_inputs = np.zeros((instances, length), dtype=np.int64)
    all_targets = np.zeros((instances, length, 3), dtype=np.float32)
    
    for i in range(instances):
        if i > 0 and i % 10000 == 0:
            print(f"Processed {i} words...")
            
        # 1. Generate random word string
        word = generateRandomString(length)
        
        # 2. Map input string characters directly into token IDs
        all_inputs[i] = word_to_token_ids(word)
        
        # 3. Fetch step-by-step substring descents
        step_descents = GetStepDescents(word)
        
        # 4. Map each substring descent string into a soft-probability vector
        for step_idx, descent_str in enumerate(step_descents):
            all_targets[i, step_idx] = string_to_soft_vector(descent_str)
            
    print(f"Generation Complete! Structuring into paired columns...")
    
    # Create a structured list of records (Rows)
    # Column 1: "letters" (the original string or token list)
    # Column 2: "descent_vectors" (the list of 20 probability vectors)
    dataset_records = []
    
    for i in range(instances):
        record = {
            "letters": all_inputs[i].tolist(),          # List of 20 token IDs
            "descent_vectors": all_targets[i].tolist()  # List of 20 probability vectors
        }
        dataset_records.append(record)
        
    # Save this structured list directly
    torch.save(dataset_records, "soft_descent_dataset_paired.pt")
    print(f"Dataset successfully compiled and stored as a paired list of records!")
    print(f"Total processing elapsed time: {t.elapsed():.2f} seconds.")

if __name__ == "__main__":
    main()
