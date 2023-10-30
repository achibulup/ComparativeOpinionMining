import glob
from VnCoreNLP import VnCoreNLP
import json
import os
from typing import TypedDict

Label = TypedDict("Label", {
  "subject": list[int, int], 
  "object": list[int, int], 
  "aspect": list[int, int], 
  "predicate": list[int, int], 
  "label": str
})

class InputData:
  tokenized_sentences: list[str]
  sentences_words: list[str]
  tokenized_words: list[str]

def parseDataset(tokenizer: VnCoreNLP, data_path: str) -> list[(InputData, (bool, list[Label]))] :
  train_dataset = []
  cur_sentence = None
  cur_tokenized_sentences = None
  cur_tokenized_words = None
  cur_label = []
  files = glob.glob(data_path + "*.txt") if os.path.isdir(data_path) else [data_path]
  for file in files:
    with open(file, 'r', encoding="utf-8") as f:
      for line in f.readlines():
        line = line.strip()
        if line == "":
          continue

        if line.startswith("{"): #label line
          label = json.loads(line)
          for key in label.keys():
            if key != "label":
              original_index = parseStartEndIndex(label[key])
              label[key] = remapToTokenizedIndex(
                  original_index, 
                  cur_sentence, 
                  cur_tokenized_words
              )
          cur_label.append(label)

        else: #sentence line
          if cur_sentence != None:
            input = InputData()
            input.tokenized_sentences = cur_tokenized_sentences
            input.sentences_words = cur_sentence
            input.tokenized_words = cur_tokenized_words
            train_dataset.append((
              input, 
              (int(len(cur_label) != 0), cur_label)
            ))
            cur_label = []
          cur_sentence = line.split("\t")[1]
          try:
            annotations = tokenizer.annotate(cur_sentence)
          except Exception as e:
            print(file)
            print(cur_sentence)
            raise e
          cur_tokenized_sentences = []
          cur_tokenized_words = []
          for sentence_annotation in annotations:
            cur_tokenized_sentences.append(" ".join(sentence_annotation["word_form"]))
            cur_tokenized_words.extend(sentence_annotation["word_form"])
          cur_sentence = cur_sentence.split(" ")

      # end of for line in f.readlines()
      if cur_sentence != None:
        input = InputData()
        input.tokenized_sentences = cur_tokenized_sentences
        input.sentences_words = cur_sentence
        input.tokenized_words = cur_tokenized_words
        train_dataset.append((
          input, 
          (int(len(cur_label) != 0), cur_label)
        ))

  return train_dataset

def parseStartEndIndex(indexRangeArray: list[str]) -> (int, int):
  if len(indexRangeArray) == 0:
    return (-1, -1)
  else:
    begin = int(indexRangeArray[0].split("&&")[0]) - 1
    end = int(indexRangeArray[-1].split("&&")[0]) - 1
    return (begin, end)
  
def remapToTokenizedIndex(index: (int, int), original: list[str], tokenized: list[str]) -> (int, int):
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
  while original_index <= index[1]:
    cur_token_match_len += len(original[original_index])
    if cur_token_match_len == len(tokenized[tokenized_index]):
      cur_token_match_len = 0
      tokenized_index += 1
    else :
      cur_token_match_len += 1 # account for '_' character
    original_index += 1
  end = tokenized_index - 1
  return (begin, end)

#remove all characters that are not alphabetic, numeric, or common punctuation
#should be as fast as possible
# def cleanText(text: str) -> str:
