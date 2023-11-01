import torch
from torch import nn
from transformers import AutoModel
from torchcrf import CRF

class BertCell(nn.Module):
  def __init__(self):
    super(BertCell, self).__init__()
    self.bert = AutoModel.from_pretrained("vinai/phobert-base-v2")
    self.dense = nn.Linear(768, 2)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, input_ids: list[list[int]], attn_mask: list[list[int]]):
    with torch.no_grad():
      sequence_output, pooled_output = self.bert(input_ids, attention_mask=attn_mask)[:2]
    slice = sequence_output[:, 0, :]
    dense_output = self.dense(slice)
    return self.softmax(dense_output)
  
class CRFCell(nn.Module):
  def __init__(self, num_tags: int):
    super(CRFCell, self).__init__()
    self.crf = CRF(num_tags, batch_first=True)

  def forward(self, input_ids: list[list[int]], attn_mask: list[list[int]], labels: list[list[int]] | None):
    if labels is None:
      return self.crf.decode(input_ids, mask=attn_mask)
    return self.crf(input_ids, labels, mask=attn_mask).neg()