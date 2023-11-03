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
    # self.dense = nn.Linear(768, 2)
    # self.softmax = nn.Softmax(dim=1)

  def forward(self, input_ids: list[list[int]], attn_mask: list[list[int]]):
    with torch.no_grad():
      sequence_output, pooled_output = self.bert(input_ids, attention_mask=attn_mask)[:2]
    return sequence_output, pooled_output
  
class CRFCell(nn.Module):
  def __init__(self, num_tags: int):
    super(CRFCell, self).__init__()
    self.crf = CRF(num_tags, batch_first=True)

  def forward(self, input_ids: list[list[list[int]]], 
              attn_mask: list[list[int]], bmeo_mask_target: list[list[int]] | None):
    result = self.crf.decode(input_ids, mask=attn_mask)
    loss = None
    if bmeo_mask_target is not None:
      loss = self.crf(input_ids, bmeo_mask_target, mask=attn_mask, reduction="none").neg()
    return result, loss
  

class TheModel(nn.Module):
  def __init__(self, bert_model = None):
    global BERT_HIDDEN_SIZE
    super(TheModel, self).__init__()
    self.bert = BertCell() if bert_model is None else bert_model
    self.identification = nn.Sequential(
      nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE // 2),
      nn.Sigmoid(),
      nn.Linear(BERT_HIDDEN_SIZE // 2, 1),
      nn.Sigmoid()
    )
    self.element_linear = nn.ModuleList()
    for i in range(4):
      self.element_linear.append(nn.Linear(BERT_HIDDEN_SIZE, len("BMEO")))
    self.label_linear = torch.nn.Sequential(
      torch.nn.Linear(BERT_HIDDEN_SIZE, len(LABELS)),
    )
    self.crf = nn.ModuleList()
    for i in range(4):
      self.crf.append(CRFCell(len("BMEO")))

  def forward(self, input_ids: list[list[int]], attn_mask: list[list[int]], 
      annotations: Annotations, elem_bmeo_mask:list[list[list[int]]]=None):
    token_embedding, pooled_output = self.bert(input_ids, attn_mask)
    batch_size, sequence_length, _ = token_embedding.size()
    # final_embedding = self.embedding_dropout(token_embedding)
    # class_embedding = self.embedding_dropout(pooled_output)
    is_comparative_prob = self.identification(pooled_output)
    element_prob = [w(token_embedding) for w in self.element_linear]
    sentence_class_prob = self.label_linear(pooled_output)
    elem_output = []
    for index in range(4):
      if elem_bmeo_mask is None:
        elem_output.append(self.crf[index](element_prob[index], attn_mask, None))
      else:
        elem_output.append(self.crf[index](element_prob[index], attn_mask, elem_bmeo_mask[:, index, :]))
    # elem_output = torch.cat(elem_output, dim=0).view(3, batch_size, sequence_length).permute(1, 0, 2)
    # elem_feature = element_prob
    # elem_feature = [elem_feature[index].unsqueeze(0) for index in range(len(elem_feature))]
    return is_comparative_prob, elem_output, sentence_class_prob

# _, sent_output = torch.max(torch.softmax(sent_class_prob, dim=1), dim=1)
# sent_loss = F.cross_entropy(is_comparative_prob, comparative_label.view(-1))
# crf_loss = sum(elem_output)