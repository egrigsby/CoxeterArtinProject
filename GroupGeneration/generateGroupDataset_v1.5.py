import logging
from pathlib import Path
import random
import time
import os
from operator import neg  # for negating integers
from typing import List, Tuple
import numpy as np
from groups import Group, COXETER, ARTIN
import pandas as pd
from sklearn.model_selection import train_test_split
import utils

logger = logging.getLogger(__name__)

def padWord(word_as_tuple:tuple, fixedWordLength):
  fill = [0] * (fixedWordLength - len(word_as_tuple))
  return list(word_as_tuple) + fill

################################################################################
## Actual Functions to generate and manipulate csv and dataframes of datasets ##
################################################################################

class DataGenerator:
  def __init__(self, coxeterMatrix=None, mode=None, dataDir="datasets", min_wordLength=None, max_wordLength=None, fixed_wordLength=None, datasetSize=None, train_size=None, groupName="", BR='¦'):
    # what path can you expect all datasets to be in 
    self.groupName = groupName
    self.BR = BR               # param spliting character
    self.dataDir = Path(dataDir)      # parent folder containing all datasets (all subfolders)
    self.datasetPath = None     # defined after all params have been set
    
    self.coxeterMatrix = coxeterMatrix
    # only give the class your matrix, later functions feed it more parameters
    self.mode = mode
    # set word sizes 
    self._setSizes(min_wordLength, max_wordLength, fixed_wordLength)
    # set total target dataset size 
    self.datasetSize = datasetSize
    
    #self.fileSize = datasetSize//2   # let it be initialized in makeDataset function, half of s.datasetSize
    
    # TODO tentative more important variable than fileSize, techincally the same thing but used as an indicator for timeout loop
    self.target_trivial_word_count = None# datasetSize//2
    self.totalTrivialWords_num = None# None
    
    # split (DONT IMMEDIATLY CREATE SPLIT) 
    self.train_size = train_size
    
    # other uninitialized variables (mode specific)
    self.group = Group(self.coxeterMatrix, self.mode)
    self.generators = self.group.GENERATORS
    self.relators = self.group.RELATORS
    self.reduceVisible_routine = self.group.reduceVisible
    self.createWord_routine = self.group.expand_trivial_word

    ## REVAMP: more than one file name 
    """
    relator dataset files:
    2-relator.txt ..... "#-relator.txt"
    2-relatorCounter.txt  --> "matching the relator file with nontrivial words 
    
    word length dataset files:
    4-trivial.txt ..... "#-dataset.txt"
    4-trivialCounter.txt
    """
    # "constant" variables (conventional file names)
    # "constant" variables (suffix file names)
    self.trivial_suffix = "trivial.txt" 
    self.trivialCounter_suffix = "trivialCounter.txt" 
    self.trainFile = "train.csv" 
    self.testfile = "test.csv"
    
  
  def generateFolderName(s) -> Path:
    """MUST RUN in order to get dynamically generated folder with description of dataset"""
    BR = s.BR
    generationRunName = f"{s.groupName} {BR} '{s.mode}' {BR} {s.min_wordLength}-{s.max_wordLength} {BR} pad {s.fixed_wordLength} {BR} size {s.datasetSize:,} {BR} split {int(s.train_size * 100)} {int((1-s.train_size) * 100)}"
    # get list of folders with this exact name         
    matches = s._matches(generationRunName)
    count = len(matches) # just len since 1st folder has index 0
    s.folderName = f"{count} {BR} {generationRunName}"
    # finally set datasetPath for this specific task 
    s.datasetPath = Path(s.dataDir / s.folderName)
   
  # helper functions:         
  def _setSizes(self, min_wordLength, max_wordLength, fixed_wordLength):
    self.min_wordLength =  min_wordLength
    self.max_wordLength = max_wordLength
    self.fixed_wordLength = fixed_wordLength
  def _matches(s, runName):
    folders = [name for name in os.listdir(s.dataDir)
            if os.path.isdir(os.path.join(s.dataDir, name))]
    folderTypes = [folder[4:] for folder in folders]
 
    matches = []
    for i, folderType in enumerate(folderTypes):
      if runName == folderType: 
        matches.append(folders[i])
             
    return matches

  def writeRawTrivial_Partial(s, setWordLen, startTime, maxTime):
    """
    function that generates all trivial words of length setWordLen (currWordLen)
    trivialWordsSet : set of unique trivial words into this function and then updated
    """
    trivialWordsSet = set()    
    lenTrivialSet = len(trivialWordsSet)
    
    # Timeout loop for A PART of the whole dataset being generated

    max_idle_time = 0.5
    last_addition_time = time.time()
    while True:
      # timeout condition
      B_overallTimeoutReached = time.time() - startTime > maxTime
      B_noProgressTimeoutReached = time.time() - last_addition_time > max_idle_time
      B_completedTrivialSetSize = s.totalTrivialWords_num + lenTrivialSet >= s.target_trivial_word_count
      if B_overallTimeoutReached or B_noProgressTimeoutReached or B_completedTrivialSetSize:
          logger.info(f"Word size {setWordLen:<3} done | "
                      f"Time Used {time.time()-startTime:<8.4f} | "
                      f"Words Generated {len(trivialWordsSet)}")
          break
      # cache the current number of trivial words
      lenTrivialSet = len(trivialWordsSet)
 
      trivialWord = s.group.generateTrivialWord(setWordLen, setWordLen)
      trivialWordsSet.add(tuple(trivialWord))
      
      # "Progress bar" for number of unique words being added:
      #if len(trivialWordsSet) > lenTrivialSet and len(trivialWordsSet) % 100 == 0:
      #  logger.info(f"Reached {len(trivialWordsSet)} unique words")
 
      if len(trivialWordsSet) > lenTrivialSet:
        # update addition time          
        last_addition_time = time.time()

    # outside of loop:

    # log last sized amount of words collected
    #logger.info(f"Word size {setWordLen:<3}done | "
    #            f"Time Used {time.time()-startTime:<6.4f}| "
    #            f"Words Generated {len(trivialWordsSet) - lenTrivialSet}")
    # return unique set, all of some fixed word length
    return trivialWordsSet

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
    file_path = s.datasetPath / s.trivial_suffix
    fileObj = open(file_path, mode="w")

    # TODO: getting rid of this general thing
    trivialWords = set()    # saving as unique set of trivial words
    
    ## TODO make sure this works (adds )   (could modify subroutine b instead)  NOTE we're not doing this todo, went with alternative method again 
    #if s.addRelators == True:
    #  for relator in s.relators: 
    #    trivialWords.add(relator)
    
    # Bruteforce getting a complete dataset for a fixed word length size starting from the minimum and going up by two until the maximum is reached
    for currSize in range(s.min_wordLength, s.max_wordLength + 2, 2):
      maxTime = utils.allotTime(currSize)
      startTime = time.time()
      newTrivialWords = s.writeRawTrivial_Partial(currSize, startTime, maxTime)
      trivialWords.update(newTrivialWords)
      # update current size of ALL trivial words dataset
      s.totalTrivialWords_num = len(trivialWords)
    
    
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
    nontrivialDataset = set()
    for trivialWord in trivialDataset:           
      lenTrivialWord = utils.getTrueWordLength(trivialWord)    
      nontrivialWord = s.group.generateNontrivialWord(lenTrivialWord)
      nontrivialDataset.add(tuple(nontrivialWord))

    # create fileObj with a timestamp 
    file_path = s.datasetPath / f"{s.trivialCounter_suffix}"
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
      train_df, test_df = train_test_split(raw_df, train_size=train_size, random_state=42, stratify=raw_df['label'])
      
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
    
    #s.fileSize = s.datasetSize//2    # TODO: remove, outdated
    s.target_trivial_word_count = s.datasetSize // 2
    s.totalTrivialWords_num = 0

    #TODO add check based on how many files are in the subfolders (if not 4 than clean and delete the invalid folders)
    # create path folder
    os.makedirs(s.datasetPath, exist_ok=True)
    
    # TODO write empty file that just has the date (could contain details about the dataset inside)
    
    # write raw trivial dataset 
    rawTrivialPath = s.writeRawTrivialDataset()
    trivialDataset = utils.readDataset(rawTrivialPath)    #TODO can be more efficient
    # write raw non trivial dataset 
    rawNontrivialPath = s.writeRawNontrivialDataset(trivialDataset)
    
    # create split 
    trainDF, testDF = s.createTrainTestSplitData(rawTrivialPath, rawNontrivialPath, random_state=random_state)
    
    return trainDF, testDF
  
  def makeDatasets(s, userDatasetPath=None, random_state=1):
    """
    creates several txt files for relator and counter relator datasets as well as N-length trivial and nontrivial word datasets 
    
    returns nothing 
    """
    # ini dir
    os.makedirs(s.datasetPath, exist_ok=True)

    
    # 1. create relator datasets
    
    
    # 2. create N-length datasets
    s.target_trivial_word_count = s.datasetSize // 2 
    s.totalTrivialWords_num = 0
    
    
    
    pass
  
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

  coxeterMatrix = np.array([
      [1, 3, 3],
      [3, 1, 3],
      [3, 3, 1],
  ])
  dg = DataGenerator(coxeterMatrix, dataDir="datasets", mode=COXETER, BR=BR)
  dg.groupName = "A2_tilde"


  # define word length, dataset size, splits 
  min_wordLen = 6
  max_wordLen =  22
  fixed_wordLen = max_wordLen
  dg.datasetSize = 129300  #6000 * 2
  dg.train_size = 0.3
  dg._setSizes(min_wordLen, max_wordLen, fixed_wordLen)

  # generate folder name for dataset using dataset features (updates folderPath)
  dg.generateFolderName()
  folderName = dg.folderName
  print(f"Unique folder name for dataset:\n{folderName}")
  # define directory path (defined via generation or manually)
  trainDF, testDF = dg.makeDataset(userDatasetPath=dg.datasetPath, random_state=1)

if __name__ == "__main__":
  setup_logging(level=logging.INFO)  
  # enable logging if file is being run as main
  debug()   # run as main