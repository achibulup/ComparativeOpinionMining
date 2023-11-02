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

  def __getitem__(self, index: int):
    return self.data[index]
    
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

def identity(x):
  return x

class ClassDataLoader(data.DataLoader):
  def __init__(self, dataset: COMDataSet, batch_size: int, shuffle: bool = None, sampler: data.Sampler = None):
    super(ClassDataLoader, self).__init__(dataset, batch_size, shuffle=shuffle, sampler=sampler, collate_fn=identity)