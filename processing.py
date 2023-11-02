import glob
from VnCoreNLP import VnCoreNLP
from VnCoreNLP import Annotations
import json
import os
from typing import TypedDict
import problem_spec
from problem_spec import ELEMENTS, ELEMENTS_NO_LABEL, LABELS
import torch
from transformers import PreTrainedTokenizer

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
          cur_input = mineInputSentence(sentence, tokenizer)
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

def mineInputSentence(sentence: str, tokenizer: VnCoreNLP) -> InputData:
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

  while original_index < index[1]: # do the same thing
    cur_token_match_len += len(original[original_index])
    if cur_token_match_len == len(tokenized[tokenized_index]):
      cur_token_match_len = 0
      tokenized_index += 1
    else :
      cur_token_match_len += 1 # account for '_' character
    original_index += 1
  end = tokenized_index

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

def decode(mask: list[int]) -> tuple[int, int]:
  """[*O*, O, O, B, M, M, E, O,....,*O*] -> (2, 6) 
  the mask include additional O at the beginning and end to account for the [CLS] and [SEP] tokens
  """
  begin = 0
  end = 0
  for i in range(len(mask)):
    if mask[i] != 0:
      if begin == 0:
        begin = i
      end = i
  if end == len(mask) - 1:
    end -= 1
  return (begin - 1, end - 1)

def mapLabelStrToInt(label: str) -> int:
  return problem_spec.LABELS.index(label)

def mapLabelIntToStr(label: int) -> str:
  return problem_spec.LABELS[label]











def transformData(data: tuple[InputData, LabelData], tokenizer: PreTrainedTokenizer):
    inp, label = data
    encoded_input = tokenizer.encode(inp.tokenized_words)
    annotations = inp.annotations
    bmeo_mask_list : list[list[int]]= []
    for elem in ELEMENTS_NO_LABEL:
      if label.is_comparative:
        bmeo_mask_list.append(label.quintuples[0][elem][1])
      else:
        bmeo_mask_list.append([0] * len(encoded_input))
    return (encoded_input, annotations, 
        label.is_comparative, bmeo_mask_list, label.quintuples[0]["label"] if label.is_comparative else None)

def collate_fn(batch: list[tuple[
        list[int], list[Annotations], bool, list[list[int]], int | None
    ]]):
  batch_size = len(batch)
  input_ids, annotations, is_comp, elem_bmeo_masks, labels = zip(*batch)
  max_len = max([len(input_id) for input_id in input_ids])
  padded_input_ids = []
  padded_elem_bmeo_masks = []
  attn_masks = []
  for i in range(batch_size):
    this_len = len(input_ids[i])
    num_padded = max_len - this_len
    padded_input_ids.append(input_ids[i] + [0] * num_padded)
    attn_masks.append([1] * this_len + [0] * num_padded)
    padded_mask = [ori_mask + [0] * num_padded for ori_mask in elem_bmeo_masks[i]]
    padded_elem_bmeo_masks.append(padded_mask)

  input_ids = torch.tensor(padded_input_ids)
  attn_masks = torch.tensor(attn_masks, dtype=torch.bool)
  is_comp = torch.tensor(is_comp)
  elem_bmeo_masks = torch.tensor(padded_elem_bmeo_masks)
  labels = torch.tensor([label if label is not None else -1 for label in labels])
  return input_ids, attn_masks, annotations, is_comp, elem_bmeo_masks, labels

def transformBatch(batch: list[tuple[InputData, LabelData]], tokenizer: PreTrainedTokenizer):
  return collate_fn([transformData(data, tokenizer) for data in batch])

def detransformResult(result_tuple:tuple, lookup:list[tuple[InputData, LabelData]]) -> list:
  batch_size = len(lookup)
  is_comparative_prob, elem_output, sentence_class_prob = result_tuple
  result = []
  for i in range(batch_size):
    is_comparative = bool(is_comparative_prob[i] >= 0.5)
    elements = None
    label = None
    if is_comparative:
      elements = dict()
      for j, elem in enumerate(problem_spec.ELEMENTS_NO_LABEL):
        index = decode(elem_output[j][0][i])
        elements[elem] = lookup[i][0].tokenized_words[index[0]:index[1] + 1]
      label = problem_spec.LABELS[int(torch.argmax(sentence_class_prob[i]))]
    result.append((is_comparative, elements, label))
  return result
