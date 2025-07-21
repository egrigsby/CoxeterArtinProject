# Create a plot for the frequencies dictionary
import plotly.express as px
from typing import List, Tuple
import pandas as pd
import ast

# NOTE not used at all currently
def getWordLengthFrequencies(dataset) -> List[Tuple[int,int]]:
  frequencies = {}
  for word in dataset:
    wordLen = len(word)
    if wordLen in frequencies:
      frequencies[wordLen] += 1
    else:
      frequencies[wordLen] = 1
  return frequencies

# Seperated into function since this is used in several places
def getTrueWordLength(word):
  """gets actual word length (ignores 0's used for padding)"""
  try: 
    lenWord = word.index(0)
  except ValueError:
    lenWord = len(word)
  return lenWord

# Plotting:
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

# Time bounds
def allotTime(word_size:int) -> float:
  if word_size <= 14:
    return 5.0
  else:
    return word_size * 5.0

# Reading from Files
def loadCSV(datasetName:str):
    df = pd.read_csv(datasetName)  # ex: 'train.csv' or 'test.csv'
    # Convert the 'tokens' column back to lists
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    return df

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

# TODO
def loadRaw(filePath, label):
    with open(filePath, 'r') as file:
        lines = file.readlines()
    # Each line is a list of tokens separated by spaces
    return pd.DataFrame({
        'tokens': [line.strip().split() for line in lines],
        'label': label
    })


# file IO stuff

from pathlib import Path
from typing import Dict, Tuple
import re

def group_files_by_type(directory: Path) -> Tuple[
    Dict[int, Tuple[Path, Path]],  # relator_dict
    Dict[int, Tuple[Path, Path]]   # trivial_dict
]:
  relator_dict = {}
  trivial_dict = {}
  
  for file in directory.glob("*.txt"):
      match = re.match(r"(\d+)-(relator|trivial)(Counter)?\.txt", file.name)
      if not match:
          continue  # Skip files that don't match expected pattern

      length = int(match.group(1))
      file_type = match.group(2)  # "relator" or "trivial"
      is_counter = match.group(3) is not None  # True if "Counter" exists

      if file_type == "relator":
          relator_dict.setdefault(length, [None, None])
          if is_counter:
              relator_dict[length][1] = file
          else:
              relator_dict[length][0] = file
      elif file_type == "trivial":
          trivial_dict.setdefault(length, [None, None])
          if is_counter:
              trivial_dict[length][1] = file
          else:
              trivial_dict[length][0] = file
              
  # Convert lists to tuples
  relator_dict = {k: tuple(v) for k, v in relator_dict.items() if None not in v}
  trivial_dict = {k: tuple(v) for k, v in trivial_dict.items() if None not in v}

  return relator_dict, trivial_dict