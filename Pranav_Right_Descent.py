import random
import time

def generateRandomString(length: int) -> str:
    CHARACTERS = "abc"
    # Using random.choice to simulate uniform distribution picking from "abc"
    random_string = "".join(random.choice(CHARACTERS) for _ in range(length))
    return random_string

class Timer:
    def __init__(self):
        self.m_beg = time.perf_counter()
        
    def reset(self):
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
    else: return 6

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
    else:
        return w[:k] + t + w[k:]

def MultRight(s: str, w: str) -> str:
    t = s
    lambda_val = s
    k = len(w)
    for i in range(len(w) - 1, -1, -1):
        lambda_val = RootRefTable(w[i], lambda_val)
        if lambda_val == '-':
            # Simulating w.erase(k-1, 1) -> remove 1 character at index k-1
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

class PathData:
    def __init__(self):
        self.elementList = []
        self.descentsList = []

def DescentPathRight(w: str) -> PathData:
    returningPathData = PathData()
    if len(w) == 1:
        returningPathData.elementList.append(w)
        returningPathData.descentsList.append(w)
        return returningPathData
        
    wordSize = len(w)
    x = w[0:1]
    returningPathData.elementList.append(w[0:1])
    gens = ["a", "b", "c"]
    
    for i in range(1, wordSize):
        descent = ""
        newx = ""
        for j in range(3):
            # C++ .compare() == 0 checks for equality
            if gens[j] == w[i:i+1]:
                newx = MultRight(gens[j][0], x)
                if len(newx) < len(x):
                    descent += gens[j]
                returningPathData.elementList.append(newx)
            else:
                ifRD = isRightDescent(gens[j][0], x)
                if ifRD:
                    descent += gens[j]
        x = newx
        returningPathData.descentsList.append(descent)
        
    finaldescent = ""
    for j in range(3):
        if isRightDescent(gens[j][0], x):
            finaldescent += gens[j]
            
    returningPathData.descentsList.append(finaldescent)
    return returningPathData

def VectorValuesAsString(x: list) -> str:
    if not x:
        return "[]"
    output = "["
    for i in range(len(x) - 1):
        output += x[i] + ", "
    output += x[-1] + "]"
    return output

def main():
    instances = 100000
    length = 200
    t = Timer()
    
    for i in range(instances):
        if i % 1000 == 0:
            print(i)
        s = generateRandomString(length)
        x = DescentPathRight(s)
        
    print(f"It took {t.elapsed()} seconds at {instances} instances of length {length}")

if __name__ == "__main__":
    main()