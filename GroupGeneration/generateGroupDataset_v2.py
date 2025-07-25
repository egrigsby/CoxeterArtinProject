# NEVER PUSH THIS, ssh has latest
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
from collections import defaultdict

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
    self.relator_suffix = "relator.txt"
    self.relatorCounter_suffix = "relatorCounter.txt"
    
    self.trivial_suffix = "trivial.txt" 
    self.trivialCounter_suffix = "trivialCounter.txt" 
    self.train_suffix = "train.csv" 
    self.test_suffix = "test.csv"
    
  
  def generateFolderName(s) -> Path:
    """MUST RUN in order to get dynamically generated folder with description of dataset"""
    BR = s.BR
    generationRunName = f"{s.groupName} {BR} '{s.group.modeName}' {BR} {s.min_wordLength}-{s.max_wordLength} {BR} pad {s.fixed_wordLength} {BR} size {s.datasetSize:,} {BR} split {int(s.train_size * 100)} {int((1-s.train_size) * 100)}"
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

  def generateTrivialSet(s, setWordLen, startTime, maxTime):
    """
    function that generates all trivial words of length setWordLen (currWordLen)
    trivialWordsSet : set of unique trivial words into this function and then updated
    """
    trivialWordsSet = set()    
    lenTrivialSet = len(trivialWordsSet)
    
    # Timeout loop for A PART of the whole dataset being generated

    max_idle_time = 10
    last_addition_time = time.time()
    while True:
      # timeout condition
      B_overallTimeoutReached = time.time() - startTime > maxTime
      B_noProgressTimeoutReached = time.time() - last_addition_time > max_idle_time
      B_completedTrivialSetSize = s.totalTrivialWords_num + lenTrivialSet >= s.target_trivial_word_count
      if B_overallTimeoutReached or B_completedTrivialSetSize or B_noProgressTimeoutReached:
        timeoutMsg = (
            "(Completed)" if B_completedTrivialSetSize else
            "(Max Time)" if B_overallTimeoutReached else
            "(Stagnant)" if B_noProgressTimeoutReached else
            "")
        logger.info(f"{timeoutMsg:<12}| "
                    f"Word size {setWordLen:<3}| "
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

  def writeRawTrivialDataset(s, setWordLen:int, prefix=""):
    """ 
    writes a trivial dataset of a certain word length 
    setWordLen: current word length being targeted
    returns the file path containing a list of trivial words of WordLength (includes padding) as well as a count of trivial words generated
    """
    
    # create and open file 
    file_path = s.datasetPath / f"{prefix}{s.trivial_suffix}"
    fileObj = open(file_path, mode="w")
        
    # Bruteforce method for getting "complete" datasets by word lengths, with 2 step increments
    maxTime = utils.allotTime(setWordLen)
    startTime = time.time()
    trivialWords = s.generateTrivialSet(setWordLen, startTime, maxTime)    
    
    # write all unique words from the set into trivial text file
    for word in trivialWords:
      paddedWord = padWord(word, s.fixed_wordLength)
      fileObj.write(" ".join(str(item) for item in paddedWord) + "\n")
    fileObj.close()

    return file_path, len(trivialWords)

  def writeRawNontrivialDataset(s, lenTrivialWord, trivialDatasetSize, prefix=""):
    """
    datasets are passed here where all words are a fixed size
    
    lenTrivialWord: size of the trivial dataset to mimic 
    trivialDatasetSize: total trivial words to match with
        
    returns the file path of the nontrivial words written to a file
    """
    nontrivialWords = set()
    while len(nontrivialWords) < trivialDatasetSize:
      nontrivialWord = s.group.generateNontrivialWord(lenTrivialWord)
      nontrivialWords.add(tuple(nontrivialWord))

    # write to file
    file_path = s.datasetPath / f"{prefix}{s.trivialCounter_suffix}"
    fileObj = open(file_path, mode="w")

    # add the words to the nonTrivialWords.txt file
    for word in nontrivialWords:
      paddedWord = padWord(word, s.fixed_wordLength)
      fileObj.write(" ".join(str(item) for item in paddedWord) + "\n")
    return file_path

  # TODO: move some logic in makeDatasets into here
  def writeAllRelatorDatasets(s):
    
    pass



  def createTrainTestSplitData(s, rawTrivialPath, rawNontrivialPath, random_state):
      """
      helper function called by 'makeData()' that returns the dataframes according to parameters you give it 
      returns (trainDF, testDF) 
      """
      def getDatasetName(datasetPath):
        name, _ = os.path.splitext(os.path.basename(datasetPath))
        return name
            
      # Step 1: Read the raw data 
      # get the properties from the type of dataset based off the name (last subfolder)
      trivialDatasetName:str = getDatasetName(rawTrivialPath)
      nontrivialDatasetName = getDatasetName(rawNontrivialPath)
      wordLength = trivialDatasetName.split("-")[0]

      # Load data from both classes
      raw_tDF = utils.loadRaw(rawTrivialPath, '0') #raw trivial dataframe
      raw_ntDF = utils.loadRaw(rawNontrivialPath, '1') #raw non-trivial dataframe

      # combines both raw datasets into a single pandas dataframe
      raw_df = pd.concat([raw_tDF, raw_ntDF]).sample(frac=1, random_state=random_state).reset_index(drop=True)

      # creating 2 separate training and testing dataframes (modify test_size param)
      train_size = s.train_size
      train_df, test_df = train_test_split(raw_df, train_size=train_size, random_state=random_state, stratify=raw_df['label'])
      
      # Optional: print out details of both dataframes
      print("Training set size:", len(train_df))
      print("Testing set size:", len(test_df))

      # Save to CSV and return as well
      train_path = os.path.join(s.datasetPath, f"{wordLength}-length-{s.train_suffix}")
      test_path = os.path.join(s.datasetPath, f"{wordLength}-{s.test_suffix}")
      train_df.to_csv(train_path, index=False)
      test_df.to_csv(test_path, index=False)
      
      return (train_df, test_df)


  # NOTE NOT FUNCTIONAL RN
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
    rawTrivialPath = s.writeRawTrivialDataset(16)
    trivialDataset = utils.readDataset(rawTrivialPath)    #TODO can be more efficient
    # write raw non trivial dataset 
    rawNontrivialPath = s.writeRawNontrivialDataset(trivialDataset)
    
    # create split 
    return trainDF, testDF
  
  def makeDatasets(s, userDatasetPath=None, random_state=1):
    """
    creates several txt files for relator and counter relator datasets as well as N-length trivial and nontrivial word datasets 
    
    returns nothing 
    """
    # ini dir
    os.makedirs(s.datasetPath, exist_ok=True)

    # TODO NOTE: append all relators to test.csv dataset at the very end.. (adding uniquely, not repeatedly)
    
    # 1. create relator datasets
    relators = s.group.ALL_RELATORS
    datasets_by_length = defaultdict(list)
    for rel in relators:
        datasets_by_length[len(rel)].append(rel)
    datasets_by_length = dict(sorted(datasets_by_length.items()))
 
    relator_sets = {}   # length → set of tupled relators
    relator_counts = {} # length → number of relators

    # FIRST PASS
    for length in sorted(datasets_by_length):
        prefix = f"{length}-"
        filePath_relator = s.datasetPath / f"{prefix}{s.relator_suffix}"
        relators = datasets_by_length[length]

        relator_set = set()
        with open(filePath_relator, mode="w") as f_rel:
            for relator in relators:
                padded = padWord(relator, s.fixed_wordLength)
                f_rel.write(" ".join(str(item) for item in padded) + "\n")
                relator_set.add(tuple(padded))

        relator_sets[length] = relator_set
        relator_counts[length] = len(relator_set)
    # 2nd PASS
    for length in sorted(relator_counts):
        prefix = f"{length}-"
        filePath_counterRelator = s.datasetPath / f"{prefix}{s.relatorCounter_suffix}"
        num_needed = relator_counts[length]
        existing = relator_sets[length]
        generated = set()

        with open(filePath_counterRelator, mode="w") as f_nonrel:
            while len(generated) < num_needed:
                word = s.group.generateNontrivialWord(length)
                padded = tuple(padWord(word, s.fixed_wordLength))
                if padded not in existing and padded not in generated:
                    generated.add(padded)
                    f_nonrel.write(" ".join(str(item) for item in padded) + "\n")
    
    
    # 2. create N-length datasets
    s.target_trivial_word_count = s.datasetSize // 2 
    s.totalTrivialWords_num = 0
    
    # dictionary to get any 2 paths later   # TODO: make function that constructs a similar dictionary given a valid dataset directory..  
    fetchDatasetPath = {}
    # loop to make all files    
    for currSize in range(s.min_wordLength, s.max_wordLength + 2, 2):
      filePath_trivial, wordsAdded = s.writeRawTrivialDataset(currSize, prefix=f"{currSize}-")
      s.totalTrivialWords_num += wordsAdded
      filePath_nontrivial = s.writeRawNontrivialDataset(currSize, wordsAdded, prefix=f"{currSize}-")
      fetchDatasetPath[currSize] = (filePath_trivial, filePath_nontrivial)
  
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
  random_state = 1

  # OLD COXETER MATRIX: A2~
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
  dg.train_size = 0.4
  dg._setSizes(min_wordLen, max_wordLen, fixed_wordLen)

  # generate folder name for dataset using dataset features (updates folderPath)
  dg.generateFolderName()
  folderName = dg.folderName
  print(f"Unique folder name for dataset:\n{folderName}")
  # define directory path (defined via generation or manually)
  
  # DEBUG
  dg.makeDatasets(userDatasetPath=dg.datasetPath, random_state=1)
  #dg.datasetPath = dg.dataDir / "1 . A2_tilde . 'coxeter' . 6-22 . pad 22 . size 129,300 . split 30 70"
  
  
  # get the paths as dictionaries:
  relator_dict, trivial_dict = utils.group_files_by_type(dg.datasetPath)
  
  # Creating CSV's for relators and their matches 
  ## Create Full CSV of relators and their counterparts (this whole CSV is for training)
  print(f"creating complete CSV for Relators:")
  relatorDataset_df_dicts = {}
  
  all_relator_dfs = []
  for k in sorted(relator_dict.keys()):
    relatorDataset_df_dicts[k] = (utils.loadRaw(relator_dict[k][0], label=0), utils.loadRaw(relator_dict[k][1], label=1))
    # add dataframes into one list
    all_relator_dfs.extend(relatorDataset_df_dicts[k]) # add tuple as 2 separate elements
  
  # combine all relator datasets into one 
  relators_df = pd.concat(all_relator_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
  relators_df.to_csv(dg.datasetPath / "relators.csv")
  
  # overwrite text files for n-length trivial words, by removing duplicate words that exist in relators
  for k in sorted(relator_dict.keys()):
    pass
    # TODO 

  
  # Creating CSV's for Trivial/Nontrivial Word Length Mix 
  trainDFS = []
  testDFS = []
  for k in sorted(trivial_dict.keys()):
    print(f"Creating CSV's for Word Length: {k}")  
    trainDF, testDF = dg.createTrainTestSplitData(trivial_dict[k][0], trivial_dict[k][1], random_state=random_state)     #split is implicit 
    trainDFS.append(trainDF)
    testDFS.append(testDF)
    
  # combine Test CSV's 
  testing_df = pd.concat(testDFS).sample(frac=1, random_state=random_state).reset_index(drop=True)
  testing_df.to_csv(dg.datasetPath / "test.csv")
  
  
  #trainDF, testDF = dg.makeDataset(userDatasetPath=dg.datasetPath, random_state=1)

if __name__ == "__main__":
  setup_logging(level=logging.INFO)  
  # enable logging if file is being run as main
  debug()   # run as main
  
  
    
    ## TODO make sure this works (adds )   (could modify subroutine b instead)  NOTE we're not doing this todo, went with alternative method again 
    #if s.addRelators == True:
    #  for relator in s.relators: 
    #    trivialWords.add(relator)
