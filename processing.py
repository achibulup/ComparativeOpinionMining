import glob
from VnCoreNLP import VnCoreNLP
from VnCoreNLP import Annotations
import json
import os
from typing import TypedDict
import problem_spec

class InputData:
  def __init__(self, tokenized_sentences: list[str], sentences_words: list[str], tokenized_words: list[str], annotations: list[Annotations]):
    self.tokenized_sentences = tokenized_sentences 
    """list of sentences, each sentence is a string of tokenized words"""
    self.sentences_words = sentences_words 
    """list of raw words in the sentence, e.g. ["Tôi", "là", "sinh", "viên"]"""
    self.tokenized_words = tokenized_words 
    """list of words after tokenization e.g. ["Tôi", "là", "sinh_viên"]"""
    self.annotations = annotations 
    """list of annotations for each sentence"""

def aggregateMask(BMEO_mask: list[list[int]]) -> list[int]:
  """aggregate a list of BMEO masks into one mask"""
  if type(BMEO_mask[0][0]) != int:
    raise Exception("BMEO_mask must be a 2d list of integers")
  result = []
  for i in range(len(BMEO_mask[0])):
    cur_mask = 0
    for j in range(len(BMEO_mask)):
      cur_mask = cur_mask | BMEO_mask[j][i] # a | a = a; O | a = a; B | E = M; a | M = M
    result.append(cur_mask)
  return result

Quintuple = TypedDict("Quintuple", {
  "subject": tuple[tuple[int, int], list[int]], # start end index and BMEO mask
  "object": tuple[tuple[int, int], list[int]], 
  "aspect": tuple[tuple[int, int], list[int]], 
  "predicate": tuple[tuple[int, int], list[int]], 
  "label": int
})

class LabelData:
  def __init__(self, quintuples: list[Quintuple]):
    self.is_comparative = len(quintuples) > 0
    self.quintuples = quintuples
    if self.is_comparative:
      self.aggregatedBMEO = TypedDict("AggregatedBMEO", {
        "subject": list[int],
        "object": list[int], 
        "aspect": list[int], 
        "predicate": list[int]
      })()
      for element in self.quintuples[0].keys():
        if element != "label":
          self.aggregatedBMEO[element] = aggregateMask(
              [quintuple[element][1] for quintuple in self.quintuples])


def parseDataset(tokenizer: VnCoreNLP, data_path: str) -> list[tuple[InputData, LabelData]] :
  result = []
  cur_input: InputData = None
  cur_quintuples: list[Quintuple] = None

  for file in globDataFiles(data_path):
    with open(file, 'r', encoding="utf-8") as f:
      for line in f.readlines():
        line = line.strip()
        if line == "":
          continue

        if not line.startswith("{"): #sentence line
          if cur_input != None:
            result.append((
              cur_input, 
              LabelData(cur_quintuples)
            ))
          sentence: str = getSentenceFromLine(line)
          cur_input = mineInputSentence(tokenizer, sentence)
          cur_quintuples = []

        else: #quintuple line
          quintuple = mineQuintuple(cur_input, line)
          cur_quintuples.append(quintuple)

      # end of for line in f.readlines()
      # last datapoint
      if cur_input != None:
        result.append((
          cur_input, 
          LabelData(cur_quintuples)
        ))

  return result



def globDataFiles(data_path: str) -> list[str]:
  return glob.glob(data_path + "*.txt") if os.path.isdir(data_path) else [data_path]

def getSentenceFromLine(line: str) -> str:
  # the data is in the format of "untokenized_sentence \t tokenized_sentence"
  return line.split("\t")[1]

def mineInputSentence(tokenizer: VnCoreNLP, sentence: str) -> InputData:
  """Populate the InputData object"""
  annotations = tokenizer.annotate(sentence)
  tokenized_sentences = []
  tokenized_words = []
  for sentence_annotation in annotations:
    tokenized_sentences.append(" ".join(sentence_annotation["word_form"]))
    tokenized_words.extend(sentence_annotation["word_form"])
  sentences_words = sentence.split(" ")

  return InputData(tokenized_sentences, sentences_words, tokenized_words, annotations)

def mineQuintuple(lookup: InputData, line: str) -> Quintuple:
  """Populate the Quintuple object"""
  raw_quintuple = json.loads(line)
  quintuple = Quintuple()
  for element in raw_quintuple.keys():
    if element != "label":
      original_index = parseStartEndIndex(raw_quintuple[element])
      remapped_index = remapToTokenizedIndex(
          original_index, 
          lookup.sentences_words, 
          lookup.tokenized_words
      )
      mask = createBMEOMask(remapped_index, len(lookup.tokenized_words))
      quintuple[element] = (remapped_index, mask)
    else:
      quintuple[element] = mapLabelStrToInt(raw_quintuple[element])
  return quintuple

def parseStartEndIndex(indexRangeArray: list[str]) -> tuple[int, int]:
  """["3$$a", "4$$b", "5$$c"] -> (3-1, 5) = (2, 5)"""
  if len(indexRangeArray) == 0:
    return (-1, -1)
  else:
    begin = int(indexRangeArray[0].split("&&")[0]) - 1
    end = int(indexRangeArray[-1].split("&&")[0]) - 1
    return (begin, end)
  
def remapToTokenizedIndex(index: tuple[int, int], original: list[str], tokenized: list[str]) -> tuple[int, int]:
  if index == (-1, -1):
    return index
  original_index = 0
  tokenized_index = 0
  cur_token_match_len = 0

  while original_index < index[0]:
    cur_token_match_len += len(original[original_index])
    if cur_token_match_len == len(tokenized[tokenized_index]):
      cur_token_match_len = 0
      tokenized_index += 1
    else :
      cur_token_match_len += 1 # account for '_' character
    original_index += 1
  begin = tokenized_index

  while original_index <= index[1]: # do the same thing
    cur_token_match_len += len(original[original_index])
    if cur_token_match_len == len(tokenized[tokenized_index]):
      cur_token_match_len = 0
      tokenized_index += 1
    else :
      cur_token_match_len += 1 # account for '_' character
    original_index += 1
  end = tokenized_index - 1

  return (begin, end)

def createBMEOMask(index: tuple[int, int], num_words: int) -> list[int]:
  """(2, 6) -> [*O*, O, O, B, M, M, E, O,....,*O*]
  the mask include additional O at the beginning and end to account for the [CLS] and [SEP] tokens
  """
  result = [0] * num_words
  if index != (-1, -1):
    result[index[1] - 1] = 2
    result[index[0]] = 1
    for i in range(index[0] + 1, index[1] - 1):
      result[i] = 3
  result = [0] + result + [0]
  return result

def mapLabelStrToInt(label: str) -> int:
  return problem_spec.LABELS.index(label)

def mapLabelIntToStr(label: int) -> str:
  return problem_spec.LABELS[label]

#remove all characters that are not alphabetic, numeric, or common punctuation
#should be as fast as possible
# def cleanText(text: str) -> str:
