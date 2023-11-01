from torch.utils import data
from processing import InputData, LabelData, Quintuple
from transformers import AutoTokenizer 
import torch
from VnCoreNLP import Annotations
import random
from problem_spec import ELEMENTS_NO_LABEL, LABELS


def oneHot(index: int, size: int) -> list[int]:
  if type(index) == bool:
    index = int(index)
  ret = [0] * size
  ret[index] = 1
  return ret

class COMDataSet(data.Dataset):
  def __init__(self, data: list[tuple[InputData, LabelData]]):
    super(COMDataSet, self).__init__()
    self.data = data
    self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

  def __getitem__(self, index: int):
    inp, label = self.data[index]
    encoded_input = self.tokenizer.encode(inp.tokenized_words)
    annotations = inp.annotations
    bmeo_mask_list : list[list[int]]= []
    for elem in ELEMENTS_NO_LABEL:
      if label.is_comparative:
        bmeo_mask_list.append(label.quintuples[0][elem][1])
      else:
        bmeo_mask_list.append([0] * len(encoded_input))
    return (encoded_input, annotations, 
        label.is_comparative, bmeo_mask_list, label.quintuples[0]["label"] if label.is_comparative else None)

  def __len__(self):
    return len(self.data)
  
class BalancedSampler(data.Sampler):
  def __init__(self, dataset: COMDataSet, *, positive_rate: float = 0.5, seed: int = random.randint(0, 1000)):
    super(BalancedSampler, self).__init__(dataset)
    self.dataset = dataset
    self.positive_rate = positive_rate
    self.rng = random.Random(seed)
    self.negatives = []
    self.positives = []
    for i in range(len(self.dataset)):
      if self.dataset.data[i][1].is_comparative:
        self.positives.append(i)
      else:
        self.negatives.append(i)
    self.cur_neg = []
    self.cur_pos = []

  def __iter__(self):
    ret = []
    for i in range(len(self.dataset)):
      if self.rng.random() < 1 - self.positive_rate:
        if (len(self.cur_neg) == 0):
          self.cur_neg = self.negatives.copy()
          self.rng.shuffle(self.cur_neg)
        ret.append(self.cur_neg.pop())
      else:
        if (len(self.cur_pos) == 0):
          self.cur_pos = self.positives.copy()
          self.rng.shuffle(self.cur_pos)
        ret.append(self.cur_pos.pop())
    return iter(ret)

  def __len__(self):
    return len(self.dataset)

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

class ClassDataLoader(data.DataLoader):
  def __init__(self, dataset: COMDataSet, batch_size: int, shuffle: bool = None, sampler: data.Sampler = None):
    super(ClassDataLoader, self).__init__(dataset, batch_size, shuffle=shuffle, sampler=sampler, collate_fn=collate_fn)