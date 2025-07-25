# NOTE: assumed to be running from the root directory of the Github Repository

import numpy as np
from operator import neg  #for negating integers
import random
from typing import List, Tuple
import os
from pathlib import Path

# misc:
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# Modes:
COXETER = 0
ARTIN = 1

# Paths:
DATASET_DIR = Path("datasets")

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

class Coxeter:
  def __init__(self, matrix):
    self.matrix = matrix
    self.GENERATORS = self.get_generators()
    self.RELATORS:set = self.get_relators()
    
    # TODO: make sure these two would work similarly as caching for an Artin group object 
    self.ALL_RELATORS = self.get_extensive_relatorSet()
    self.ALL_BIG_RELATORS = self.get_validlySized_relators()

  def get_generators(myGroup):
    n = np.sum(myGroup.matrix == 1)            #masks for 1s and sums all true values
    generators = list(range(1,n+1))    #generates integers from 1 through n
    return generators 

  def get_relators(myGroup) -> set:
    #generate pairs to check cases where s != s'
    generators = myGroup.get_generators()
    pairs = [(generators[i], generators[j]) for i in range(len(generators)) for j in range(i + 1, len(generators))]
    relators = []

    #getting relators of form s^2
    for g in generators:
      relators.append((g,g))

    #getting braid relators
    for p in pairs:
      m = myGroup.matrix[p[0]-1,p[1]-1]       #subtracting 1 retrieves the correct row and column, eg what we call row 1 is actually indexed as row 0
      if np.any(np.isinf(m)):         #skipping to the next pair if m(s,s') = infinity
        continue
      relators.append(p*(int(m)))     #otherwise, appends the pair p m times, representing the relation (ss')^m(s,s') = e
    return relators

  def get_validlySized_relators(myGroup):
    return [rel for rel in myGroup.ALL_RELATORS if len(rel) != 2]

  def get_extensive_relatorSet(myGroup):
    all_relators = set()
    for rel in myGroup.RELATORS:
      rel_inv = Coxeter.word_inverse(rel)
      all_relators.update([rel, rel_inv])
    return all_relators

  # static functions (excepte for subroutine B)
  @staticmethod
  def as_string(w):
    return "".join(f's{i}'.translate(SUB) for i in w)

  # Checks if the coxeter matrix is a valid one 
  @staticmethod
  def is_matrix(n, m):

      # Converts values to numeric, fails if any non-numeric values
      try:
          m = m.astype(float)
      except ValueError:
          print("Non-numeric error: Matrix contains non-numeric entries.")
          return

      # Checks if input matrix is of size rank x rank
      if m.shape != (n, n):
          print(f"Invalid shape: Expected a square matrix of size {n}×{n}, but got {m.shape}.")
          return

      # Checks if diagonal entries are 1
      if not np.all(np.diag(m) == 1):
          print("Invalid diagonal: All diagonal entries must be 1 in a Coxeter matrix.")

      # Check if matrix is symmetric (including handling NaN/inf)
      if not np.allclose(m, m.T, equal_nan=True):
          print("Symmetry error: Coxeter matrices must be symmetric across the diagonal.")
          return

      # Mask off diagonal ones if they exist
      off_diagonal = ~np.eye(n, dtype=bool)
      invalid_ones = (m == 1) & off_diagonal

      # Checks for off diagonal ones
      if np.any(invalid_ones):
          print("Off-diagonal 1s detected: Only diagonal entries can be 1 in a Coxeter matrix.")
          return

      # Mask infinities before checking cast
      is_inf = np.isinf(m)
      is_pos_int = (m > 0) & ~is_inf & (m == np.floor(m))
      valid_entries = is_inf | is_pos_int

      # Checks if all values are positive, and if they are an integer or an infinity
      if not np.all(valid_entries):
          print("Entry error: All off-diagonal entries must be integers ≥ 2 or ∞.")
          return

      print("This is a valid Coxeter matrix.")
  
  @staticmethod
  def word_inverse(w):
    return w[::-1]

  ### Subroutine A: reduces a coxeter word by removing adjacent equal generators iteratively.
  @staticmethod
  def reduceVisible(w):
    reduced = True
    while reduced:
      reduced = False
      i = 0
      while i < len(w) - 1:
        if w[i] == w[i+1]:
          w = w[:i] + w[i+2:]
          i = max(i-1,0)
        else:
          i += 1
    return w

  # NOTE issue about there being 2 or less generators present.. 
  # "subroutine b" (instance variable)
  def expand_trivial_word(coxGroup, tWord, maxWordLen):
    # subroutine a preemptively
    tWord = Coxeter.reduceVisible(tWord)
    tWordLen = len(tWord)
    insertion_point = random.randint(0, len(tWord))

    #define gens between insertion point (T = ttttt a (conjugate) b tttt)
    a, b = None, None
    if tWordLen > 0:
      if insertion_point > 0:
          a = tWord[insertion_point - 1]
      if insertion_point < tWordLen:
          b = tWord[insertion_point]

    ## Get list of ALL validly sized relators (+ inverses hence all)
    all_relators = list(coxGroup.ALL_BIG_RELATORS)

    ## Generate W (generated semi randomly) (returning w * conjugate * w_inv)
    w = []
    # First remove the relators that are greater than maxWordLen    #TODO: cache all possibilities to remove inefficency? 
    smallishRelators = [rel for rel in coxGroup.RELATORS if len(rel) <= maxWordLen]
    w_max_length = (maxWordLen - max([len(x) for x in smallishRelators]))  // 2        # max word len - max relator length // 2
    w_length = random.randint(0,w_max_length)    
    for _ in range(w_length):
      w.append(random.choice(coxGroup.GENERATORS))

    # reduce w using subroutine A
    w = Coxeter.reduceVisible(w)

    # NOTE: Goal: get a conjugate: w R W to put into T

    # IF W has length of 0, insert a random relator into t (which could be empty) and RETURN
    if len(w) == 0:
      tWord[insertion_point:insertion_point] = random.choice(all_relators)
      return tWord

    # calculate w inverse(note its for coxeter group)
    w_inv = Coxeter.word_inverse(w)

    # creating list of candidates
    candidates = all_relators # NOTE: old logic gone, all_relators include any relator greater than length 2


    #candidates = []
    # w (R) W
    #for rel in all_relators:
    #    # w_end == relStart or relEnd = W_start
    #    if w[-1] == rel[0] or rel[-1] == w_inv[0]:
    #      pass
    #    else:
    #      candidates.append(rel)
    # NOTE, a rerun check was removed because it's not possible to have no relators (no relators without causing a collision was possible, but that's no longer used)
    #if len(candidates) == 0:
    #  return coxGroup.expand_trivial_word(tWord, maxWordLen)

    # select a relator to use for trivial conjugate 
    r = random.choice(candidates)

    ## Insert conjugate: w + r + w_inv
    conjugate = w + list(r) + w_inv
    tWord[insertion_point:insertion_point] = conjugate
    return tWord

  
class Artin:
  def __init__(self, matrix):
    self.matrix = matrix
    self.GENERATORS = self.get_generators()
    self.RELATORS = self.get_relators()
    

  def get_generators(myGroup):
    n = np.sum(myGroup.matrix == 1)            #masks for 1s and sums all true values
    generators = list(range(-n,n+1))   #generates integers from -n through n
    generators.remove(0)
    return generators

  def get_relators(myGroup):
    #generate pairs to check cases where s != s'
    generators = myGroup.matrix
    pairs = [(generators[i], generators[j]) for i in range(len(generators)) for j in range(i + 1, len(generators))]

    relators = []

    #retrieving length m from m(s,s')
    for p in pairs:
      m = myGroup.matrix[p[0]-1,p[1]-1]
      if np.any(np.isinf(m)):         #skipping to the next pair if m(s,s') = infinity
        continue

      #building pi(s,s',m)
      pi = []

      #alternating between s and s' for an m-length list
      for i in range(int(m)):
        if i % 2 == 0:
          pi.append(p[0])     #even indices give s
        else:
          pi.append(p[1])     #odd indices give s'

      #building pi(s',s, m) inverse
      pi_inv = []
      for i in range(int(m)):               #same process as above except
        if i % 2 != 0:
          pi_inv.append(p[0])               #even indices now give s'
        else:
          pi_inv.append(p[1])               #and odd indices give s
      pi_inv = list(map(neg, pi_inv))       #flip signs to denote inverses

      #combining pi and pi inverse
      relators.append(pi + pi_inv)

    return relators

  # Class functions
  @staticmethod
  def as_string(w):
    result = []
    for i in w:
      index = abs(i)
      gen = f's{str(index).translate(SUB)}'
      if i < 0:
        gen += '⁻¹'
      result.append(gen)
    return "".join(result)

  @staticmethod
  def reduceVisible(w):
    stack = []
    for x in w:
      if stack and stack[-1] == -x:
        stack.pop()
      else:
        stack.append(x)
    return stack

  @staticmethod
  def word_inverse(w):
    return [-g for g in reversed(w)]

  # TODO reimplement this.. make sure ALL_BIG_RELATORS and the other variable are initialized on the artin group object if possible for caching
  def expand_trivial_word(artGroup, tWord, maxWordLen):
    pass

# Note: good to run unless only 1 generator. Then, abandon.
  def subroutine_b_artin(artGroup, tWord, set_of_generators, set_of_relators, maxWordLen):
      tWord = Artin.reduceVisible(tWord)
      tWordLen = len(tWord)
      insertion_point = random.randint(0, tWordLen)

      # Get neighbors to avoid cancellation at insertion boundaries
      a = tWord[insertion_point - 1] if insertion_point > 0 else None
      b = None
      if tWordLen > 0:
        b = tWord[insertion_point] if insertion_point < tWordLen else None

      ## Get list of ALL relators 
      all_relators = []
      validlySized_relators = [rel for rel in set_of_relators if len(rel) != 2] # TODO replace with len(rel) != 2 (NOTE remove this todo, change done)
      for rel in validlySized_relators:
        inv_rel = Artin.word_inverse(rel)         # specific to artin group 
        all_relators.extend([rel, inv_rel])

      ## Generate W (generated semi randomely)
      
      # max w len = max word len - max relator length // 2  
      w_max_length = (maxWordLen - max([len(x) for x in set_of_relators]))  // 2
      w_length = random.randint(0,w_max_length)     # TODO debug, make sure this works (seems to work)
      w = []
      for i in range(w_length):
        w.append(random.choice(set_of_generators))    # TODO IMPORTANT make function that definitely generates a non-obviously trivial word w? (NOTE would prevent accidental w length of 0)

      # reduce w using subroutine A
      w = Artin.reduceVisible(w)    
      # TODO LOGIC DIFFERS (1): add boundary check on realtor (like code for standard non zero w length case below)?
      if len(w) == 0:
        tWord[insertion_point:insertion_point] = random.choice(all_relators)
        return tWord

      # calculate w inverse
      w_inv = Artin.word_inverse(w)

      # TODO LOGIC DIFFERS (1): Early check: avoid reduction at boundaries with t
      #if (a is not None and w and a == -w[0]) or (b is not None and w_inv and w_inv[-1] == -b):
      #    return subroutine_b_artin(t, set_of_generators, set_of_relators, maxWordLen)


      ####### Choose a non-reducing relator     TODO LOGIC DIFFERS (1): from len(w) == 0 case, makes sure relator picked 
      valid_relators = []
      for rel in set_of_relators:
          # Skip trivial relators like (g, -g)
          if len(rel) == 2 and rel[0] == -rel[1]:
              continue
          for rel_tuple in [list(rel), [-g for g in reversed(rel)]]:
              # TODO debug make sure commenting these out works
              #if w and rel_tuple and w[-1] == -rel_tuple[0]:
              #    continue  # would cancel with end of w
              #if rel_tuple and w_inv and rel_tuple[-1] == -w_inv[0]:
              #    continue  # would cancel with start of w_inv
              #rel_reduced = reduce_artin_word(rel_tuple)
              #if not rel_reduced:
              #  continue
              valid_relators.append(rel_tuple)

      if not valid_relators:
          return artGroup.subroutine_b_artin(tWord, set_of_generators, set_of_relators, maxWordLen)

      r = random.choice(valid_relators)

      ####### Form conjugate and insert
      conjugate = w + r + w_inv
      tWord[insertion_point:insertion_point] = conjugate
      return conjugate


class Group:

  def __init__(self, matrix, MODE):
    self.MODE = MODE
    # functions
    if MODE == COXETER:
      # naming
      self.modeName = "coxeter"
      # obj:
      self.GroupObj = Coxeter(matrix)
      # attributes
      self.GENERATORS = self.GroupObj.GENERATORS
      self.RELATORS = self.GroupObj.RELATORS
      self.ALL_RELATORS = self.GroupObj.ALL_RELATORS
      # functs
      self.get_generators = Coxeter.get_generators
      self.get_relators = Coxeter.get_relators
      self.reduceVisible = Coxeter.reduceVisible
      self.expand_trivial_word = self.GroupObj.expand_trivial_word
    elif MODE == ARTIN:
      # naming 
      self.modeName = "artin"
      # obj:
      self.GroupObj = Artin(matrix)
      # attributes:
      self.GENERATORS = self.GroupObj.GENERATORS
      self.RELATORS = self.GroupObj.RELATORS
      self.ALL_RELATORS = self.GroupObj.ALL_RELATORS    # TODO: implement this correctly with sub_b
      # functs
      self.get_generators = Artin.get_generators
      self.get_relators = Artin.get_relators
      self.reduceVisible = Artin.reduceVisible
      self.expand_trivial_word = self.GroupObj.expand_trivial_word
    
    # run edge check, error will be raised if code not possible
    self._edgeCaseCheck()

  def _edgeCaseCheck(self):
    # goal: edgewase check runs once, wordElongater quickly generates code from then on
    # Check edge case where subroutine B wouldn't work (only 1 valid relator that only ever uses 2 generators)
    # get at least 2 relators with at least 2 generators 
    uniqueRels = []
    if self.MODE == COXETER:
      for rel in self.RELATORS: 
        if len(rel) <= 2:  # skip relators that are too short
          continue
        uniqueGens = set()
        for gen in rel: 
          uniqueGens.add(gen)
        if len(uniqueGens) >= 2:
          uniqueRels.append(rel)
      # check if number of unique relators is less than 2
      if len(uniqueRels) == 1:
        rel = uniqueRels[0]
        if random.random() < 0.5:
          rel = rel[::-1]  # reverse the relator with 50% probability
        raise ValueError("TODO: find better fix for invalid generator besides rotating under this condition")
        #return uniqueRels[0] * (minWordLength // len(uniqueRels[0]))   # TODO, create fixed bool for this task using this function, so that this return can be ran if possible repeatedly
      elif len(uniqueRels) == 0:
        raise ValueError("Not enough valid relators with at least 2 generators to elongate the word.")
    elif self.MODE == ARTIN:
      if len(self.GENERATORS) == 1:
        raise ValueError("Not enough generators to elongate the word.")

  # NOTE: Both generation functions are non obviously trivial

  # instance variable, reliant on existing matrix with properties in order to use: expand_trivial_word function
  def generateTrivialWord(self, minWordLength, maxWordLength) -> List[int]:
    """
    goal: generate a trivial word between a min and max by making it longer using subroutineB then 
    removing 'aa' relations to make it less visibly reducible
    """
    ## Subroutine B: Elongating the word (T > t C t) where C = w R W
    tWord = []
    
    #keep elongating the word until it's at least as large as the minWordLength
    while( len(tWord) < minWordLength ):
      tWord = self.expand_trivial_word(tWord, maxWordLength)

    ## Subroutine A: removing the 'aa' visible trivial parts of a word post creating the word with subroutine B
    tWord = self.reduceVisible(tWord)

    #check that the generated word is within the desired range (if not then recall this function)
    if len(tWord) < minWordLength:
      tWord = self.generateTrivialWord(minWordLength, maxWordLength)
    #make sure word doesn't pass maxWordLength
    if len(tWord) > maxWordLength:
      tWord = self.generateTrivialWord(minWordLength, maxWordLength)
    
    return tWord
  
  # obvious reducibility is different for coxeter and artin groups (NOTE only use for nontrivial word generation for now..) 
  def generateNontrivialWord(self, length: int):
      word = []
      lastGen = 0
      while len(word) < length:
          newGen = random.choice(self.GENERATORS)
          while True:
              if self.MODE == COXETER:
                  condition = newGen == lastGen
              elif self.MODE == ARTIN:
                  condition = newGen == -lastGen
              else:
                  condition = False
              if not condition:
                  break
              newGen = random.choice(self.GENERATORS)
          word.append(newGen)
          lastGen = newGen    # word[-1]
      return word
    
# NOTE: need functions for word length datasets (made using subroutines) and also relator datasets (if len 2 we need a special way of generating these)