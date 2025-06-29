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

# Note: If there are only 2 or less generators present in the finite relators of a Coxeter Group the following function won't work
##### If there are 2 generators present in the finite relators, the only valid trivial word is alternating between the generators (ie s1s2s1s2s1s2 repeating or s2s1s2s1s2s1 repeating)
##### If there is only 1 generator present in the finite relators, abandon the activity as it is impossible to generate visually unreducable trivial words

def subroutine_b_cox(t, set_of_generators, set_of_relators):
  t = reduce_coxeter_word(t)
  size_of_trivial = len(t)
  insertion_point = random.randint(0, size_of_trivial)

  #letters between insertion point
  a = None
  b = None
  if size_of_trivial > 0:
    if insertion_point > 0:
        a = t[insertion_point - 1]
    if insertion_point < size_of_trivial:
        b = t[insertion_point]

  ####### randomly generate word w
  w_length = random.randint(1,10)
  w = []
  for i in range(w_length):
    w.append(random.choice(set_of_generators))

  # reduce w using subroutine A
  w = reduce_coxeter_word(w)

  if len(w) == 0:
    return subroutine_b_cox(t, set_of_generators, set_of_relators)

  # calculate w inverse(note its for coxeter group)
  w_inv = w[::-1]

  # make sure word w does not create visually reducable words
  if (a is not None and a == w[0]) or (b is not None and w_inv[-1] == b):
    return subroutine_b_cox(t, set_of_generators, set_of_relators)



####### pick a relator that is not visually reducable
####### Choose a relator that avoids obvious cancellation with w and w_inv
  non_squares = [rel for rel in set_of_relators if rel[0] != rel[1]]
  candidates = []

  for rel in non_squares:
      for rel_tuple in [rel, rel[::-1]]:
          if w[-1] != rel_tuple[0] and rel_tuple[-1] != w_inv[0]:
              candidates.append(rel_tuple)

  if not candidates:
      return subroutine_b_cox(t, set_of_generators, set_of_relators)

  r = random.choice(candidates)

  ####### Insert conjugate: w + r + w_inv
  conjugate = w + list(r) + w_inv
  t[insertion_point:insertion_point] = conjugate
  return t

# Note: good to run unless only 1 generator. Then, abandon.
def subroutine_b_artin(t, set_of_generators, set_of_relators):
    t = reduce_artin_word(t)
    size_of_trivial = len(t)
    insertion_point = random.randint(0, size_of_trivial)

    # Get neighbors to avoid cancellation at insertion boundaries
    a = t[insertion_point - 1] if insertion_point > 0 else None
    b = None
    if len(t) > 0:
      b = t[insertion_point] if insertion_point < size_of_trivial else None

    ####### Generate random reduced word w
    w_length = random.randint(1, 10)
    w = [random.choice(set_of_generators) for _ in range(w_length)]
    w = reduce_artin_word(w)
    if not w:
        return subroutine_b_artin(t, set_of_generators, set_of_relators)

    w_inv = [-g for g in reversed(w)]

    # Early check: avoid reduction at boundaries with t
    if (a is not None and w and a == -w[0]) or (b is not None and w_inv and w_inv[-1] == -b):
        return subroutine_b_artin(t, set_of_generators, set_of_relators)


    ####### Choose a non-reducing relator
    valid_relators = []
    for rel in set_of_relators:
        # Skip trivial relators like (g, -g)
        if len(rel) == 2 and rel[0] == -rel[1]:
            continue
        for rel_tuple in [list(rel), [-g for g in reversed(rel)]]:
            if w and rel_tuple and w[-1] == -rel_tuple[0]:
                continue  # would cancel with end of w
            if rel_tuple and w_inv and rel_tuple[-1] == -w_inv[0]:
                continue  # would cancel with start of w_inv
            rel_reduced = reduce_artin_word(rel_tuple)
            if not rel_reduced:
              continue
            valid_relators.append(rel_tuple)

    if not valid_relators:
        return subroutine_b_artin(t, set_of_generators, set_of_relators)

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
  goal: generate a trivial word of length N by making it longer using subroutineB then removing 'aa' relations to make it less visibly reducible
  """
  word_creation_routine = None    #subroutine_b
  reduce_visible_routine = None   #subroutine_a

  if mode == "coxeter":
    word_creation_routine = subroutine_b_cox
    reduce_visible_routine = reduce_coxeter_word
  elif mode == "artin":
    word_creation_routine = subroutine_b_artin
    reduce_visible_routine = reduce_artin_word

  #initialize the empty word
  tWord = []

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

  ## Subroutine B: Elongating the word
  #run until desired size is reached (tWord will be of length: >= N)
  tWord = word_creation_routine(tWord, generators, relators)  #1st pass
  while( len(tWord) < minWordLength ):
    tWord = word_creation_routine(tWord, generators, relators)

  ## Subroutine A: removing the 'aa' visible trivial parts of a word
  #tWord=subroutineA(tWord)
  tWord = reduce_visible_routine(tWord)

  #check that it's long enough, if it is then return tWord, if not then call again
  if len(tWord) < minWordLength:
    tWord = wordElongater(generators, relators, minWordLength, maxWordLength, mode=mode)
  #now at least of length 20
  if len(tWord) > maxWordLength: #desiredWordLength derived from N+(some val)
    tWord = wordElongater(generators, relators, minWordLength, maxWordLength, mode=mode)
  #now at least of length 35, cuts of really long words, uses fixedWordLength max to create in between desiredWordLength
  
  # padding being done in file writing functions

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

## Read dataset from file into memory 
def readDataset(filepath:str):
  words = []
  with open(filepath) as fileObj:
    for line in fileObj:
      raw_list = line.split(" ")   #note: last gen has \n char as well
      gen_list = list(map(int, raw_list))
      lenWord = len(gen_list)
      words.append(gen_list)
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
def plotFrequencies(dataset):
  #turn dataset into list of lengths
  wordLengths = [len(word) for word in dataset]

  plt.hist(wordLengths)


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
  def __init__(self, coxeterMatrix=None, mode=None, dataDir="generated_datasets", timestamp=None, min_wordLength=None, max_wordLength=None, fixed_wordLength=None, datasetSize=None, train_size=None):
    # what path can you expect all datasets to be in 
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
    generationRunName = f"mode '{s.mode}' | range {s.min_wordLength} - {s.max_wordLength} | pad {s.fixed_wordLength} | size {s.datasetSize:,} | split {int(s.train_size * 100)} {int((1-s.train_size) * 100)}"
    # get list of folders with this exact name         
    matches = s._matches(generationRunName)
    count = len(matches) # just len since 1st folder has index 0
    s.folderName = f"{count} | {generationRunName}"
           
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
    # create and open file 
    if s.timestamp == None:  
      s.timestamp = getTimestamp()
    
    file_path = os.path.join(s.datasetPath, s.trivialFile)
    fileObj = open(file_path, mode="w")
    
    # get desiredWordLength value in between: minWordLen,maxWordLen, fixedWordLen
    
    # first save as a set (of unique elements)
    # TODO: explore other fixes for creating more unique trivialWords (for example, re-examine how subroutine b's possible solutions could all be maximally used)
    trivialWords = set()
    while len(trivialWords) != s.fileSize:
      word_as_tuple = wordElongater(s.generators, s.relators, s.min_wordLength, s.max_wordLength, mode=s.mode)
      trivialWords.add(tuple(word_as_tuple))
    
    # then write to file 
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
    
def debug():
  # get timestamp (for job)
  timestamp = getTimestamp()  #format: YYYY-MM-DD
  coxeterMatrix = np.array([
      [1, 3, 2],
      [3, 1, 3],
      [2, 3, 1],
  ])
  # create object for generating data 
  dg = DataGenerator(coxeterMatrix, mode="coxeter", dataDir="datasets", timestamp=timestamp)
  # define group type (mode)
  dg.mode = 'coxeter'
  dg.timestamp = timestamp

  # define word length, dataset size, splits 
  min_wordLen = 16
  max_wordLen =  16
  fixed_wordLen = max_wordLen
  dg.datasetSize = 15000
  dg.train_size = 0.3
  dg.setSizes(min_wordLen, max_wordLen, fixed_wordLen)

  # generate folder name for dataset using dataset features (updates folderPath)
  folderName = dg.generateFolderName()
  print(f"Unique folder name for dataset: {folderName}")
  # define directory path (defined via generation or manually)
  trainDF, testDF = dg.makeDataset(userDatasetPath=folderName, random_state=1)

if __name__ == "__main__":
  debug()