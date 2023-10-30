from torch.utils import data
from processing import InputData, Label
from transformers import AutoTokenizer 
import torch

class ClassDataSet(data.Dataset):
  def __init__(self, data: list[(InputData, (bool, list[Label]))]):
    super(ClassDataSet, self).__init__()
    self.data = data
    self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

  def __getitem__(self, index: int):
    row = self.data[index]
    input_sentence = " ".join(row[0].tokenized_sentences)
    encoded_input = self.tokenizer.encode(input_sentence)
    return encoded_input, [0, 1] if row[1][0] else [1, 0]

  def __len__(self):
    return len(self.data)

def collate_fn(batch: list[(list[int], list[int])]):
  input_ids, labels = zip(*batch)
  max_len = max([len(input_id) for input_id in input_ids])
  padded_input_ids = []
  attn_masks = []
  for input_id in input_ids:
    padded_input_ids.append(input_id + [0] * (max_len - len(input_id)))
    attn_masks.append([1] * len(input_id) + [0] * (max_len - len(input_id)))
  input_ids = torch.tensor(padded_input_ids)
  attn_masks = torch.tensor(attn_masks)
  labels = torch.tensor(labels, dtype=torch.float32)
  return input_ids, attn_masks, labels

class ClassDataLoader(data.DataLoader):
  def __init__(self, dataset: ClassDataSet, batch_size: int, shuffle=False):
    super(ClassDataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn=collate_fn)