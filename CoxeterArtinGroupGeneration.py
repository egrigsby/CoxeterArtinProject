#Run this preamble to import some libraries that are available in google colab that are often useful.
#Numpy is good for efficiently working with array/vector/matrix data.
#Random is good for generating random numbers according to some (discrete or continuous) distribution
#Matplotlib is good for plotting
#Torch is PyTorch, which is the standard python library for creating and training ML models
#You may need to call other libraries for your code

import numpy as np
import random
import matplotlib.pyplot as plt
#import torch

from operator import neg  #for negating integers
import time
import logging
logger = logging.getLogger(__name__)

########################################################
## Base functions for getting generators and relators ##
########################################################

def cox_gen(matrix):
  n = np.sum(matrix == 1)            #masks for 1s and sums all true values
  generators = list(range(1,n+1))    #generates integers from 1 through n
  return generators 

def cox_rel(matrix):
  #generate pairs to check cases where s != s'
  generators = cox_gen(matrix)
  pairs = [(generators[i], generators[j]) for i in range(len(generators)) for j in range(i + 1, len(generators))]
  relators = []

  #getting relators of form s^2
  for g in generators:
    relators.append([g,g])

  #getting braid relators
  for p in pairs:
    m = matrix[p[0]-1,p[1]-1]       #subtracting 1 retrieves the correct row and column, eg what we call row 1 is actually indexed as row 0
    if np.any(np.isinf(m)):         #skipping to the next pair if m(s,s') = infinity
      continue
    relators.append(p*(int(m)))     #otherwise, appends the pair p m times, representing the relation (ss')^m(s,s') = e
  return relators

def artin_gen(matrix):
  n = np.sum(matrix == 1)            #masks for 1s and sums all true values
  generators = list(range(-n,n+1))   #generates integers from -n through n
  generators.remove(0)
  return generators

def artin_rel(matrix):
  #generate pairs to check cases where s != s'
  generators = cox_gen(matrix)
  pairs = [(generators[i], generators[j]) for i in range(len(generators)) for j in range(i + 1, len(generators))]

  relators = []

  #retrieving length m from m(s,s')
  for p in pairs:
    m = matrix[p[0]-1,p[1]-1]
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

##################################################################################
## Subroutine Functions to reduce a coxeter/artin word and to create using conjugates ##
##################################################################################

### Display Functions for Generators and Relators in String Format ##
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
def coxeter_word_to_string(w):
  return "".join(f's{i}'.translate(SUB) for i in w)

def artin_word_to_string(w):
  result = []
  for i in w:
    index = abs(i)
    gen = f's{str(index).translate(SUB)}'
    if i < 0:
      gen += '⁻¹'
    result.append(gen)
  return "".join(result)

### Checks if the coxeter matrix is a valid one 
def is_coxeter_matrix(n, m):

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

### Subroutine A: reduces a coxeter word by removing adjacent equal generators iteratively.
def reduce_coxeter_word(w):
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

def reduce_artin_word(w):
  stack = []
  for x in w:
    if stack and stack[-1] == -x:
      stack.pop()
    else:
      stack.append(x)
  return stack

### Subroutine B: creates a coxeter word using conjugates of the generators 
from re import sub
# helper functions for subroutine b (coxeter and artin)
def coxeter_inverse(w):
  return w[::-1]
def artin_inverse(w):
  return [-g for g in reversed(w)]

# Note: If there are only 2 or less generators present in the finite relators of a Coxeter Group the following function won't work
##### If there are 2 generators present in the finite relators, the only valid trivial word is alternating between the generators (ie s1s2s1s2s1s2 repeating or s2s1s2s1s2s1 repeating)
##### If there is only 1 generator present in the finite relators, abandon the activity as it is impossible to generate visually unreducable trivial words

# TODO: move this function into class, adding extra parameters into this and coxeter for now
def subroutine_b_cox(t, set_of_generators, set_of_relators, maxWordLen):
  t = reduce_coxeter_word(t)
  size_of_trivial = len(t)
  insertion_point = random.randint(0, size_of_trivial)

  #letters between insertion point (T = ttttt a (conjugate) b tttt)
  a = None
  b = None
  if size_of_trivial > 0:
    if insertion_point > 0:
        a = t[insertion_point - 1]
    if insertion_point < size_of_trivial:
        b = t[insertion_point]


  ## Get list of ALL relators 

  all_relators = []
  validlySized_relators = [rel for rel in set_of_relators if len(rel) != 2]
  for rel in validlySized_relators:
    rel_inv = coxeter_inverse(rel)
    all_relators.extend([rel, rel_inv])

  ## Generate W (generated semi randomly)
  
  w = []  
  w_max_length = (maxWordLen - max([len(x) for x in set_of_relators]))  // 2        # max word len - max relator length // 2
  w_length = random.randint(0,w_max_length)     # TODO debug, make sure this works
  for i in range(w_length):
    w.append(random.choice(set_of_generators))    # TODO make function that definitely generates a non-obviously trivial word

  # reduce w using subroutine A
  w = reduce_coxeter_word(w)

  # IF W has length of 0, insert a random relator into t (which could be empty)   # TODO LOGIC DIFFERS (2): no collision check with insertion into t
  if len(w) == 0:
    t[insertion_point:insertion_point] = random.choice(all_relators)
    return t

  # calculate w inverse(note its for coxeter group)
  w_inv = coxeter_inverse(w)

  # TODO LOGIC DIFFERS (2): this collision check is not done for len(w) = 0
  # make sure word w does not create visually reducable words
  #if (a is not None and a == w[0]) or (b is not None and w_inv[-1] == b):
  #  return subroutine_b_cox(t, set_of_generators, set_of_relators, maxWordLen)



  ####### pick a relator that is not visually reducable given w
  ####### Choose a relator that avoids obvious cancellation with w and w_inv
  candidates = []
  for rel in validlySized_relators:
      rel_inv = coxeter_inverse(rel)
      for rel_tuple in [rel, rel_inv]:
          # TODO LOGIC DIFFERS (2): removing check 
          #if w[-1] != rel_tuple[0] and rel_tuple[-1] != w_inv[0]:
          candidates.append(rel_tuple)

  # rerun if no viable relator
  if len(candidates) == 0:
      return subroutine_b_cox(t, set_of_generators, set_of_relators, maxWordLen)

  # select a relator to use for trivial conjugate 
  r = random.choice(candidates)

  ## Insert conjugate: w + r + w_inv
  conjugate = w + list(r) + w_inv
  t[insertion_point:insertion_point] = conjugate
  return t

# Note: good to run unless only 1 generator. Then, abandon.
def subroutine_b_artin(t, set_of_generators, set_of_relators, maxWordLen):
    t = reduce_artin_word(t)
    size_of_trivial = len(t)
    insertion_point = random.randint(0, size_of_trivial)

    # Get neighbors to avoid cancellation at insertion boundaries
    a = t[insertion_point - 1] if insertion_point > 0 else None
    b = None
    if len(t) > 0:
      b = t[insertion_point] if insertion_point < size_of_trivial else None

    ## Get list of ALL relators 
    
    all_relators = []
    validlySized_relators = [rel for rel in set_of_relators if len(rel) != 2] # TODO replace with len(rel) != 2 (NOTE remove this todo, change done)
    for rel in validlySized_relators:
      inv_rel = artin_inverse(rel)         # specific to artin group 
      all_relators.extend([rel, inv_rel])

    ## Generate W (generated semi randomely)
    
    # max w len = max word len - max relator length // 2  
    w_max_length = (maxWordLen - max([len(x) for x in set_of_relators]))  // 2
    w_length = random.randint(0,w_max_length)     # TODO debug, make sure this works (seems to work)
    w = []
    for i in range(w_length):
      w.append(random.choice(set_of_generators))    # TODO IMPORTANT make function that definitely generates a non-obviously trivial word w? (NOTE would prevent accidental w length of 0)

    # reduce w using subroutine A
    w = reduce_artin_word(w)    
    # TODO LOGIC DIFFERS (1): add boundary check on realtor (like code for standard non zero w length case below)?
    if len(w) == 0:
      t[insertion_point:insertion_point] = random.choice(all_relators)
      return t

    # calculate w inverse
    w_inv = artin_inverse(w)

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
        return subroutine_b_artin(t, set_of_generators, set_of_relators, maxWordLen)

    r = random.choice(valid_relators)

    ####### Form conjugate and insert
    conjugate = w + r + w_inv
    t[insertion_point:insertion_point] = conjugate
    return conjugate


# Additional
from typing import List, Tuple

####################################################
### Generate a trvial word using both subroutines ##
####################################################


# Function generating the trivial words
# TODO: consider moving into the DataGenerator object..
def wordElongater(generators, relators, minWordLength, maxWordLength, mode="coxeter") -> List[int]:
  """
  goal: generate a trivial word of between a min and max by making it longer using subroutineB then removing 'aa' relations to make it less visibly reducible
  """
  word_creation_routine = None    #subroutine_b
  reduce_visible_routine = None   #subroutine_a

  if mode == "coxeter":
    word_creation_routine = subroutine_b_cox
    reduce_visible_routine = reduce_coxeter_word
  elif mode == "artin":
    word_creation_routine = subroutine_b_artin
    reduce_visible_routine = reduce_artin_word

  # TODO move this edgecase check, before doing word elongater, to not have to rerun it several times
  # Before proceeding to using the subroutines:
  # Check edge case where subroutine B wouldn't work (only 1 valid relator that only ever uses 2 generators)
  # get at least 2 relators with at least 2 generators 
  uniqueRels = []
  if mode == "coxeter":
    for rel in relators: 
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
      return uniqueRels[0] * (minWordLength // len(uniqueRels[0])) 
    elif len(uniqueRels) == 0:
      raise ValueError("Not enough valid relators with at least 2 generators to elongate the word.")
  elif mode == "artin":
    if len(generators) == 1:
      raise ValueError("Not enough generators to elongate the word.")

  ## Subroutine B: Elongating the word (w T w)
  
  #initialize the empty word
  tWord = [] 
  # 1st pass required
  tWord = word_creation_routine(tWord, generators, relators, maxWordLength)
  #keep elongating the word until it's at least as large as the minWordLength
  while( len(tWord) < minWordLength ):
    tWord = word_creation_routine(tWord, generators, relators, maxWordLength)

  ## Subroutine A: removing the 'aa' visible trivial parts of a word post creating the word with subroutine B
  tWord = reduce_visible_routine(tWord)

  #check that the generated word is within the desired range (if not then recall this function)
  if len(tWord) < minWordLength:
    tWord = wordElongater(generators, relators, minWordLength, maxWordLength, mode=mode)
  #make sure word doesn't pass maxWordLength
  if len(tWord) > maxWordLength:
    tWord = wordElongater(generators, relators, minWordLength, maxWordLength, mode=mode)
  
  return tWord


## IO operations to write raw "datasets" to files 
import os

# create a folder to store the datasets
DATA_FOLDER = "generated_datasets"
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

def getTimestamp():
  """format: YYYY-MM-DD"""
  return datetime.now().strftime("%Y-%m-%d")

from datetime import datetime
# TODO delete, no longer in use
def createFileReturnPath(fileName, fileExtension=".txt", timestamp=None):
  if timestamp is None:
    timestamp = getTimestamp()
  
  # current date is passed through
  if not fileName.endswith(fileExtension):
    fileName += fileExtension
  file_path = f"{DATA_FOLDER}/{timestamp}_{fileName}"
  # if the file already exists, append a number to the file name
  if os.path.exists(file_path):
    base, ext = os.path.splitext(file_path)
    counter = 1
    while os.path.exists(file_path):
      file_path = f"{DATA_FOLDER}/{counter}_{timestamp}-{fileName}"   #fileName include ext (.txt) 
      counter += 1  
  return file_path

# both raw datasets are timestamped and saved in the DATA_FOLDER directory

def padWord(word_as_tuple:tuple, fixedWordLength):
  fill = [0] * (fixedWordLength - len(word_as_tuple))
  return list(word_as_tuple) + fill

# Seperated into function since this is used in several places
def getTrueWordLength(word):
  """gets actual word length (ignores 0's used for padding)"""
  try: 
    lenWord = word.index(0)
  except ValueError:
    lenWord = len(word)
  return lenWord

def readDataset(filepath:str):
  """Reads dataset from a file into memory (takes out padding)"""
  words = []
  with open(filepath) as fileObj:
    for line in fileObj:
      raw_list = line.split(" ")   #note: last gen has \n char as well
      gen_list = list(map(int, raw_list))
      # get length of word without padding 
      lenWord = getTrueWordLength(gen_list)
      # appends the relevant part of the "list of generatrs" (excludes padding given true length)
      words.append(gen_list[0:lenWord])
  return words

def getWordLengthFrequencies(dataset) -> List[Tuple[int,int]]:
  frequencies = {}
  for word in dataset:
    wordLen = len(word)
    if wordLen in frequencies:
      frequencies[wordLen] += 1
    else:
      frequencies[wordLen] = 1
  return frequencies

# Create a plot for the frequencies dictionary
import plotly.express as px

def plotFrequencies(datasetList, wordType=""):
  """takes in a list of words (represented as lists), note expects no padding using readDataset to achieve this"""
  # Turn dataset into list of word lengths
  wordLengths = [len(word) for word in datasetList]
  
  # Create interactive histogram
  fig = px.histogram(
      x=wordLengths,
      nbins=max(wordLengths) - min(wordLengths) + 1,
      labels={'x': 'Word Length'},
      title=f'Distribution of {wordType} Word Lengths'
  )
  fig.update_layout(
      xaxis_title='Word Length',
      yaxis_title='Frequency',
      bargap=0.1
  )
  fig.show()


################################################################################
## Actual Functions to generate and manipulate csv and dataframes of datasets ##
################################################################################
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
def loadDataset(datasetName:str):
    df = pd.read_csv(datasetName)  # ex: 'train.csv' or 'test.csv'

    # Convert the 'tokens' column back to lists
    df['tokens'] = df['tokens'].apply(ast.literal_eval)

    return df


class DataGenerator:
  def __init__(self, coxeterMatrix=None, mode=None, dataDir="generated_datasets", timestamp=None, min_wordLength=None, max_wordLength=None, fixed_wordLength=None, datasetSize=None, train_size=None, groupName="", BR='¦'):
    # what path can you expect all datasets to be in 
    self.groupName = groupName
    self.BR = BR               # param spliting character
    self.dataDir = dataDir      # parent folder containing all datasets (all subfolders)
    self.datasetPath = None     # defined after all params have been set
    # timestamp 
    self.timestamp = timestamp
    
    self.coxeterMatrix = coxeterMatrix
    # only give the class your matrix, later functions feed it more parameters
    self.mode = mode
    # set word sizes 
    self.setSizes(min_wordLength, max_wordLength, fixed_wordLength)
    # set dataset size 
    self.datasetSize = datasetSize
    self.fileSize = None   # let it be initialized in makeDataset function, half of s.datasetSize
    # split 
    self.train_size = train_size
    
    # other uninitialized variables (mode specific)
    self.generators = None 
    self.relators = None
    self.createWord_routine = None
    self.reduceVisible_routine = None 
    
    # "constant" variables (conventional file names)
    self.trivialFile = "trivialWords.txt" 
    self.nonTrivialFile = "nontrivialWords.txt" 
    self.trainFile = "train.csv" 
    self.testfile = "test.csv"
    
       
  def setSizes(self, min_wordLength, max_wordLength, fixed_wordLength):
    self.min_wordLength =  min_wordLength
    self.max_wordLength = max_wordLength
    self.fixed_wordLength = fixed_wordLength
  
  def generateFolderName(s):
    """MUST RUN in order to get dynamically generated folder with description of dataset"""
    BR = s.BR
    generationRunName = f"{s.groupName} {BR} '{s.mode}' {BR} {s.min_wordLength}-{s.max_wordLength} {BR} pad {s.fixed_wordLength} {BR} size {s.datasetSize:,} {BR} split {int(s.train_size * 100)} {int((1-s.train_size) * 100)}"
    # get list of folders with this exact name         
    matches = s._matches(generationRunName)
    count = len(matches) # just len since 1st folder has index 0
    s.folderName = f"{count} {BR} {generationRunName}"
           
    # finally set datasetPath for this specific task 
    s.datasetPath = os.path.join(s.dataDir, s.folderName)
    
    return s.folderName
  
    
  
  # helper functions:  
  def _matches(s, runName):
    folders = [name for name in os.listdir(s.dataDir)
            if os.path.isdir(os.path.join(s.dataDir, name))]
    folderTypes = [folder[4:] for folder in folders]
 
    matches = []
    for i, folderType in enumerate(folderTypes):
      if runName == folderType: 
        matches.append(folders[i])
             
    return matches

  def initializeModeVariables(s):
    """
    Initializes
    generators, relators
    createWord_routine, reduceVisible_routine
    """
    # assumes coxeterMatrix and mode are both defined 
    if s.mode == "coxeter":
      s.generators = cox_gen(s.coxeterMatrix)
      s.relators = cox_rel(s.coxeterMatrix)
      s.createWord_routine = subroutine_b_cox
      s.reduceVisible_routine = reduce_coxeter_word
    elif s.mode == "artin":
      s.generators = artin_gen(s.coxeterMatrix)
      s.relators = artin_rel(s.coxeterMatrix)
      s.createWord_routine = subroutine_b_artin
      s.reduceVisible_routine = reduce_coxeter_word

  # TODO option to union the batches of these unique trivial sets (unique by length), would have to unlink from reliance on while loop to stop by file size
  # TODO make it so that max time is variable based on the current targeted word length (5 is a bit much for words 6,8,10, 12 but too little for 20+) 
  def writeRawTrivial_Partial(s, current_wordLength, startTime, maxTime, trivialWordsSet):
    """
    function that generates all trivial words of length current_wordLength
    trivialWordsSet : set of unique trivial words into this function and then updated
    """
    # loop until size of dataset has been reached (or the more likely timeout scenario)
    oldSize = len(trivialWordsSet)
    while len(trivialWordsSet) != s.fileSize:
      if time.time() - startTime > maxTime:
        logger.info(f"Word size {current_wordLength:<3}done | Time alloted {maxTime:<6.4f}| Words Generated {len(trivialWordsSet) - oldSize}")
        return
      
      # NOTE: fixing word length, ignoring class min and max (incremented outside of function)
      word_as_tuple = wordElongater(s.generators, s.relators, current_wordLength, current_wordLength, mode=s.mode)
      
      # attempt to add newly generated trivial word and verify that it has been added (is unique)
      oldLen = len(trivialWordsSet)
      trivialWordsSet.add(tuple(word_as_tuple))
      if len(trivialWordsSet) > oldLen:
        #logger.info(f"Number of Unique Trivial Words: {len(trivialWordsSet)}")   # DEBUG, too many prints for now TODO print every 100 unique words? or a progress bar?
        pass
    
    # log last sized amount of words collected
    logger.info(f"Word size {current_wordLength:<3}done | Time alloted {time.time()- startTime:}| Words Generated {len(trivialWordsSet) - oldSize}")


  def writeRawTrivialDataset(s):
    """ 
    writes trivial dataset based on parameters provided 
    generators: list of generators based on matrix 
    relators: list of relators based on matrix
    datasetSize: number of trivial words to generate (for this particular dataset)
    desiredWordLength: minimum length of each word to shoot for
    fixedWordLength: fixed word length that all words will have, usually desired word length + some extra amount 
    returns file path contianing list of trivial words of WordLength (includes padding )
    """
    if s.timestamp == None:  
      s.timestamp = getTimestamp()
    
    # create and open file 
    file_path = os.path.join(s.datasetPath, s.trivialFile)
    fileObj = open(file_path, mode="w")

    # get desiredWordLength value in between: minWordLen,maxWordLen, fixedWordLen    
    
    # TODO: (NOTE outdated, remove todo) explore other fixes for creating more unique trivialWords (for example, re-examine how subroutine b's possible solutions could all be maximally used)
    trivialWords = set()    # saving as unique set of trivial words
    
    ## TODO make sure this works (adds )   (could modify subroutine b instead)  NOTE we're not doing this todo, went with alternative method again 
    #if s.addRelators == True:
    #  for relator in s.relators: 
    #    trivialWords.add(relator)

    # TODO increment time in relation to "currSize" variable, since word length's (and time to generate complete set of said words) grows 'exponentially'    
    maxTime = 5   # 5 seconds
    # TODO double check this new implementation (incrementing currentWord len by 2 starting from 6 (or min), )
    
    # Bruteforce getting a complete dataset for a fixed word length size starting from the minimum and going up by two until the maximum is reached
    for currSize in range(s.min_wordLength, s.max_wordLength + 2, 2):
      maxTime = currSize * 5
      startTime = time.time()
      s.writeRawTrivial_Partial(currSize, startTime, maxTime, trivialWords)
            
    # TODO: delete this (moved this into it's own function called in the above for loop)
#    while len(trivialWords) < s.fileSize:
#      if time.time() - startTime > maxTime:
#        # print("timeout")    #debug
#        break
#      g
#      word_as_tuple = wordElongater(s.generators, s.relators, s.min_wordLength, s.max_wordLength, mode=s.mode)
#      
#      # old length
#      oldLen = len(trivialWords)
#      trivialWords.add(tuple(word_as_tuple))
#      if len(trivialWords) > oldLen:
#        print(f"New Length: {len(trivialWords)}")
#      
    
    # write all unique words from the set into trivial text file
    for word in trivialWords:
      paddedWord = padWord(word, s.fixed_wordLength)
      fileObj.write(" ".join(str(item) for item in paddedWord) + "\n")
    fileObj.close()

    return file_path 

  def writeRawNontrivialDataset(s, trivialDataset):
    """
    trivialDataset: list of trivial words (each word is a list of generators) 
    generators: list of generators based on matrix 
    fixedWordLength: fixed word length where padded with 0's are done at the end
    
    returns the file path of the nontrivial words written to a file
    note: mode is implied based on the generators given 
    """
    # create matching likely non trivial word based on length of each trivial word it reads in a loop 
    #i = 0
    #nontrivialDataset = {}
    #while i < len(trivialDataset):
    #  trivialWord = trivialDataset[i]
    #  nontrivialWord = []   #build this up

    nontrivialDataset = []
    for trivialWord in trivialDataset:     
      nontrivialWord = []
      
      #get ACTUAL length of the trivial word (find len before a 0 is found)
      try: 
        lenTrivialWord = trivialWord.index(0)
      except ValueError:
        lenTrivialWord = len(trivialWord)
        
      prevWord = 0
      randomGen = s.generators[random.randint(0, len(s.generators)-1)]
      for i in range(lenTrivialWord):
        while prevWord == randomGen: 
          randomGen = s.generators[random.randint(0, len(s.generators)-1)]
        #add some different generator that doesn't match the last one
        nontrivialWord.append(randomGen)
        prevWord = randomGen
        
      #add nonTrivial word to list within this loop    
      nontrivialDataset.append(nontrivialWord)

    # create fileObj with a timestamp 
    file_path = os.path.join(s.datasetPath, s.nonTrivialFile)
    fileObj = open(file_path, mode="w")

    # add the words to the nonTrivialWords.txt file
    for word_as_list in nontrivialDataset:
      word_as_list = padWord(word_as_list, s.fixed_wordLength)
      fileObj.write(" ".join(str(item) for item in word_as_list) + "\n")
    return file_path  

  def createTrainTestSplitData(s, rawTrivialPath, rawNontrivialPath, random_state=42):
      """
      helper function called by 'makeData()' that returns the dataframes according to parameters you give it 
      returns (trainDF, testDF) 
      """
      # Step 1: Read the raw data 
      def loadRaw(filename, label):
          with open(filename, 'r') as file:
              lines = file.readlines()
          # Each line is a list of tokens separated by spaces
          return pd.DataFrame({
              'tokens': [line.strip().split() for line in lines],
              'label': label
          })

      # Load data from both classes
      raw_tDF = loadRaw(rawTrivialPath, '0') #raw trivial dataframe
      raw_ntDF = loadRaw(rawNontrivialPath, '1') #raw non-trivial dataframe

      # combines both raw datasets into a single pandas dataframe
      raw_df = pd.concat([raw_tDF, raw_ntDF]).sample(frac=1, random_state=random_state).reset_index(drop=True)

      # creating 2 separate training and testing dataframes (modify test_size param)
      train_size = s.train_size
      test_size = 1 - s.train_size
      train_df, test_df = train_test_split(raw_df, test_size=test_size, train_size=train_size, random_state=42, stratify=raw_df['label'])

      # Optional: print out details of both dataframes
      print("Training set size:", len(train_df))
      print("Testing set size:", len(test_df))

      # Save to CSV and return as well
      train_path = os.path.join(s.datasetPath, s.trainFile)
      test_path = os.path.join(s.datasetPath, s.testfile)
      train_df.to_csv(train_path, index=False)
      test_df.to_csv(test_path, index=False)
      
      return (train_df, test_df)

  def makeDataset(s, userDatasetPath=None, random_state=1):
    """returns (trainDF, testDF)"""
    if s.timestamp == None:
      s.timestamp = getTimestamp()

    # get generators and relators updated, also subroutine functions set by mode
    s.initializeModeVariables()
    s.fileSize = s.datasetSize//2
    
    # generate files 
    if userDatasetPath != None: 
        s.datasetPath = os.path.join(s.dataDir, userDatasetPath)

    #TODO add check based on how many files are in the subfolders (if not 4 than clean and delete the invalid folders)
    # create path folder
    os.makedirs(s.datasetPath, exist_ok=True)
    
    # TODO write empty file that just has the date (could contain details about the dataset inside)
    
    # write raw trivial dataset 
    rawTrivialPath = s.writeRawTrivialDataset()
    trivialDataset = readDataset(rawTrivialPath)    #TODO can be more efficient
    # write raw non trivial dataset 
    rawNontrivialPath = s.writeRawNontrivialDataset(trivialDataset)
    
    # create split 
    trainDF, testDF = s.createTrainTestSplitData(rawTrivialPath, rawNontrivialPath, random_state=random_state)
    
    return trainDF, testDF
  
# import from notebook if logging is to be enabled
def setup_logging(level=logging.INFO):
    if not logger.hasHandlers():  # Prevent adding multiple handlers in Jupyter
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', "%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    
def debug():
  BR = "."    # break character
  # get timestamp (for job)
  timestamp = getTimestamp()  #format: YYYY-MM-DD

  coxeterMatrix = np.array([
      [1, 3, 3],
      [3, 1, 3],
      [3, 3, 1],
  ])
  dg = DataGenerator(coxeterMatrix, dataDir="datasets", timestamp=timestamp, BR=BR)
  dg.groupName = "A2_tilde"
  dg.mode = 'coxeter'
  dg.timestamp = timestamp

  # define word length, dataset size, splits 
  min_wordLen = 6
  max_wordLen =  22
  fixed_wordLen = max_wordLen
  dg.datasetSize = 6000 * 2
  dg.train_size = 0.3
  dg.setSizes(min_wordLen, max_wordLen, fixed_wordLen)

  # generate folder name for dataset using dataset features (updates folderPath)
  folderName = dg.generateFolderName()
  print(f"Unique folder name for dataset:\n{folderName}")
  # define directory path (defined via generation or manually)
  trainDF, testDF = dg.makeDataset(userDatasetPath=folderName, random_state=1)

if __name__ == "__main__":
  setup_logging(level=logging.INFO)  
  # enable logging if file is being run as main
  debug()   # run as main