import config
import problem_spec
from VnCoreNLP import VnCoreNLP
from VnCoreNLP import Annotations
from problem_spec import ELEMENTS, ELEMENTS_NO_LABEL, LABELS

import os
import json
import glob
import torch
import itertools
from typing import TypedDict, Iterable
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
    end = int(indexRangeArray[-1].split("&&")[0])
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

def mapToOriginalIndex(index: tuple[int, int], original: list[str], tokenized: list[str]) -> tuple[int, int]:
  if index == (-1, -1):
    return index
  original_index = 0
  tokenized_index = 0
  cur_token_match_len = 0

  while tokenized_index < index[0]:
    cur_token_match_len += len(original[original_index])
    if cur_token_match_len == len(tokenized[tokenized_index]):
      cur_token_match_len = 0
      tokenized_index += 1
    else :
      cur_token_match_len += 1 # account for '_' character
    original_index += 1
  begin = original_index

  while tokenized_index < index[1]: # do the same thing
    cur_token_match_len += len(original[original_index])
    if cur_token_match_len == len(tokenized[tokenized_index]):
      cur_token_match_len = 0
      tokenized_index += 1
    else :
      cur_token_match_len += 1 # account for '_' character
    original_index += 1
  end = original_index

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
  for i in range(1, len(mask) - 1):
    if mask[i] != 0:
      if begin == 0:
        begin = i
      end = i + 1
  if (begin == 0):
    return (-1, -1)
  return (begin - 1, end - 1)

def decodeList(mask: list[int]) -> list[tuple[int, int]]:
  """[*O*, O, B, E, O, B, M, E, O,....,*O*] -> [(1, 3), (4, 7)]
  the mask include additional O at the beginning and end to account for the [CLS] and [SEP] tokens
  """
  result = []
  begin = None
  end = None
  for i in range(1, len(mask)):
    if i == len(mask) - 1 or mask[i] == 0 or mask[i] == 1:
      if begin is not None:
        result.append((begin - 1, end - 1))
        begin = None
        end = None
    if i != len(mask) - 1 and mask[i] != 0:
      if begin is None:
        begin = i
      end = i + 1
  return result

def mapLabelStrToInt(label: str) -> int:
  return problem_spec.LABELS.index(label)

def mapLabelIntToStr(label: int) -> str:
  return problem_spec.LABELS[label]











def transformData(data: tuple[InputData, LabelData], tokenizer: PreTrainedTokenizer):
    inp, label = data
    encoded_input = tokenizer.encode(inp.tokenized_words)
    annotations = inp.annotations
    bmeo_mask_list : list[list[int]] = []
    for elem in ELEMENTS_NO_LABEL:
      if label.is_comparative:
        bmeo_mask_list.append(label.aggregatedBMEO[elem])
      else:
        bmeo_mask_list.append([0] * len(encoded_input))
    transformed_label = []
    for quintuple in label.quintuples:
      transformed_quintuple = []
      for elem in ELEMENTS_NO_LABEL:
        transformed_quintuple.append(quintuple[elem][0])
      transformed_quintuple.append(quintuple["label"])
      transformed_label.append(transformed_quintuple)
    return (encoded_input, annotations, 
        label.is_comparative, bmeo_mask_list, transformed_label)

def collate_fn(batch: list[tuple[
        list[int], list[Annotations], bool, list[list[int]], int
    ]]):
  batch_size = len(batch)
  input_ids, annotations, is_comp, elem_bmeo_masks, labels = zip(*batch)
  max_len = max([len(input_id) for input_id in input_ids])
  padded_input_ids = []
  padded_elem_bmeo_masks = []
  attn_masks: list[list[int]] = []
  for i in range(batch_size):
    this_len = len(input_ids[i])
    num_padded = max_len - this_len
    padded_input_ids.append(input_ids[i] + [0] * num_padded)
    attn_masks.append([1] * this_len + [0] * num_padded)
    padded_mask = [ori_mask + [0] * num_padded for ori_mask in elem_bmeo_masks[i]]
    padded_elem_bmeo_masks.append(padded_mask)
  return padded_input_ids, attn_masks, annotations, is_comp, padded_elem_bmeo_masks, labels

def toTensor(batch: tuple[
        list[list[int]], list[list[int]], list[list[Annotations]], list[bool], list[list[list[int]]], list[int]
    ]):
  input_ids, attn_masks, annotations, is_comp, elem_bmeo_masks, labels = batch
  input_ids = torch.tensor(input_ids, device=config.DEVICE)
  attn_masks = torch.tensor(attn_masks, dtype=torch.bool, device=config.DEVICE)
  is_comp = torch.tensor(is_comp, device=config.DEVICE)
  elem_bmeo_masks = torch.tensor(elem_bmeo_masks, device=config.DEVICE)
  return input_ids, attn_masks, annotations, is_comp, elem_bmeo_masks, labels

def transformBatch(batch: list[tuple[InputData, LabelData]], tokenizer: PreTrainedTokenizer):
  return toTensor(collate_fn([transformData(data, tokenizer) for data in batch]))


def part1Postprocess(result_tuple:tuple) -> list:
  is_comparative_prob, elem_output, token_embedding = result_tuple
  batch_size = len(is_comparative_prob)
  result = [None] * batch_size
  for i in range(batch_size):
    is_comparative = bool(is_comparative_prob[i] >= 0.5)
    elements = dict()
    for j, elem in enumerate(problem_spec.ELEMENTS_NO_LABEL):
      elem_masks, elem_costs = elem_output[j]
      indexes = decodeList(elem_masks[i])
      elements[elem] = indexes
    result[i] = (is_comparative, elements)
  return result

def probArgMax(class_prob: list[list[list[float]]]) -> list[list[int]]:
  result = []
  for i in range(len(class_prob)):
    result.append([])
    for j in range(len(class_prob[i])):
      class_pred = torch.argmax(class_prob[i][j]).item()
      if class_pred == len(LABELS):
        class_pred = -1
      result[i].append(class_pred)
  return result

def part2Postprocess(indexes: list[list[tuple[tuple[int, int], ...]]], class_prob: list[list[list[float]]]) -> list[list[tuple]]:
  collapsed_class_prob = probArgMax(class_prob)
  result = []
  for i in range(len(indexes)):
    result.append([])
    for j, index in enumerate(indexes[i]):
      if collapsed_class_prob[i][j] == -1:
        continue
      result[i].append(index + (collapsed_class_prob[i][j],))
  return result

def postprocess(result: tuple, lookup: InputData) -> dict:
  result_dict = dict()
  for i, elem in enumerate(problem_spec.ELEMENTS_NO_LABEL):
    elem_index = result[i]
    elem_index = mapToOriginalIndex(elem_index, lookup.sentences_words, lookup.tokenized_words)
    result_dict[elem] = [str(i+1)+"&&"+lookup.sentences_words[i] for i in range(elem_index[0], elem_index[1])]
  result_dict["label"] = problem_spec.LABELS[result[-1]]
  return result_dict



def generateCandiateQuadEmbedding(candidate_quads_indexes: list[tuple[tuple[int, int],...]], token_embedding: list[list[float]]):
  candidate_quads_embedding = []
  for quad in candidate_quads_indexes:
    quad_embedding = [token_embedding[0, :]]
    for idx in quad:
      if idx != (-1, -1):
        seq_embedding = token_embedding[idx[0] + 1 : idx[1] + 1, :]
        seq_embedding = torch.mean(seq_embedding, dim=0)
      else:
        seq_embedding = torch.zeros(token_embedding.shape[1], device=config.DEVICE)
      quad_embedding.append(seq_embedding)
    quad_embedding = torch.cat(quad_embedding)
    candidate_quads_embedding.append(quad_embedding)
  if len(candidate_quads_embedding) != 0:
    candidate_quads_embedding = torch.stack(candidate_quads_embedding)
  return candidate_quads_embedding

def isOverlapping(a: tuple[int, int], b: tuple[int, int]) -> bool:
  nulla = a == (-1, -1)
  nullb = b == (-1, -1)
  if nulla != nullb:
    return False
  if nulla and nullb:
    return True
  return min(a[1], b[1]) > max(a[0], b[0])

def generateCandidateQuadLabel(candidate_quads_indexes: list[tuple[tuple[int, int],...]], labels: list[tuple]):
  candidate_quads_label: list[int] = []
  for quad in candidate_quads_indexes:
    matches = False
    for label in labels:
      matches = all([isOverlapping(quad[i], label[i]) for i in range(len(quad))])
      if matches:
        candidate_quads_label.append(label[-1])
        break
    if not matches:
      candidate_quads_label.append(len(LABELS))
  return candidate_quads_label


def formatResult(result: tuple[bool, dict[str, tuple[int, int]] | None, int | None], 
                 lookup: InputData) -> str | None:
  is_comparative, elements, label = result
  if not is_comparative:
    return None
  else:
    result_dict = dict()
    for elem in elements.keys():
      index = mapToOriginalIndex(elements[elem], lookup.sentences_words, lookup.tokenized_words)
      result_dict[elem] = [str(i+1)+"&&"+lookup.sentences_words[i] for i in range(index[0], index[1])]
    result_dict["label"] = problem_spec.LABELS[label] if label != -1 else None
    return json.dumps(result_dict, ensure_ascii=False)