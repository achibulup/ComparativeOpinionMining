import torch
from torch import nn
from transformers import AutoModel
from torchcrf import CRF
from VnCoreNLP import Annotations
from problem_spec import ELEMENTS, ELEMENTS_NO_LABEL, LABELS

BERT_HIDDEN_SIZE = 768

class BertCell(nn.Module):
  def __init__(self):
    super(BertCell, self).__init__()
    self.bert = AutoModel.from_pretrained("vinai/phobert-base-v2")

  def forward(self, input_ids: list[list[int]], attn_mask: list[list[int]]):
    with torch.no_grad():
      return self.bert(input_ids, attention_mask=attn_mask)[:2]
  
class CRFCell(nn.Module):
  def __init__(self, num_tags: int):
    super(CRFCell, self).__init__()
    self.crf = CRF(num_tags, batch_first=True)

  def forward(self, input_ids: list[list[list[int]]], attn_mask: list[list[int]]):
    return self.crf.decode(input_ids, mask=attn_mask)
  
  def loss(self, input_ids: list[list[list[int]]], 
              attn_mask: list[list[int]], bmeo_mask_target: list[list[int]]):
    return self.crf(input_ids, bmeo_mask_target, mask=attn_mask, reduction="none").neg()
  

class BertCrfCell(nn.Module):
  def __init__(self, bert_model = None):
    global BERT_HIDDEN_SIZE
    super(BertCrfCell, self).__init__()
    self.bert = BertCell() if bert_model is None else bert_model
    # self.dropout = nn.Dropout(0.1)
    self.identification = nn.Sequential(
      nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE // 2),
      nn.Sigmoid(),
      nn.Linear(BERT_HIDDEN_SIZE // 2, 1),
      nn.Sigmoid()
    )
    self.element_linear = nn.ModuleList()
    for i in range(4):
      self.element_linear.append(nn.Linear(BERT_HIDDEN_SIZE, len("BMEO")))
    self.label_from_sentence_linear = torch.nn.Sequential(
      torch.nn.Linear(BERT_HIDDEN_SIZE, len(LABELS)),
    )
    self.crf = nn.ModuleList()
    for i in range(4):
      self.crf.append(CRFCell(len("BMEO")))

  def forward(self, input_ids: list[list[int]], attn_mask: list[list[int]], 
      annotations: Annotations, elem_bmeo_mask:list[list[list[int]]]=None):
    token_embedding, pooled_output = self.bert(input_ids, attn_mask)
    batch_size, sequence_length, _ = token_embedding.size()
    # token_embedding = self.dropout(token_embedding)
    # final_embedding = self.embedding_dropout(token_embedding)
    # class_embedding = self.embedding_dropout(pooled_output)
    is_comparative_prob = self.identification(pooled_output)
    element_prob = [w(token_embedding) for w in self.element_linear]
    elem_output = []
    for index in range(4):
      if elem_bmeo_mask is None:
        elem_output.append((self.crf[index](element_prob[index], attn_mask), None))
      else:
        elem_output.append((self.crf[index](element_prob[index], attn_mask), 
            self.crf[index].loss(element_prob[index], attn_mask, elem_bmeo_mask[:, index, :])))
    return is_comparative_prob, elem_output, token_embedding

class ClassificationCell(nn.Module):
  def __init__(self):
    super(ClassificationCell, self).__init__()
    self.classification = nn.Sequential(
      nn.Linear(BERT_HIDDEN_SIZE * 5, BERT_HIDDEN_SIZE * 2),
      nn.Sigmoid(),
      nn.Linear(BERT_HIDDEN_SIZE * 2, len(LABELS) + 1),
    )
  
  def forward(self, candidate_quads: list[list[list[float]]]):
    result = []
    torch.TensorType
    for i in range(len(candidate_quads)):
      result.append(self.classification(candidate_quads[i]) if len(candidate_quads[i]) != 0 else [])
    return result
  
class BertCrfExtractor(nn.Module):
  def __init__(self, bertcrf_cell = None, classification_cell = None):
    super(BertCrfExtractor, self).__init__()
    self.bertcrf = BertCrfCell() if bertcrf_cell is None else bertcrf_cell
    self.classification = ClassificationCell() if classification_cell is None else classification_cell