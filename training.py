import models
import torch
import data
from config import DEVICE, DO_TRAIN_PART1, DO_TRAIN_PART2

class Metric:
  def __init__(self):
    self.mean_loss: float | None = None
    self.accuracy: float | None = None
    self.precision: float | None = None
    self.recall: float | None = None
    self.f1: float | None = None
  def __str__(self):
    return f"mean_loss: {self.mean_loss}, accuracy: {self.accuracy}, precision: {self.precision}, recall: {self.recall}, f1: {self.f1}"

def trainClassifier(
    model: models.TheModel, train_dataloader: data.ClassDataLoader, val_dataloader: data.ClassDataLoader | None, 
    *, epochs: int = 10, optimizer = None, metric_callback = None):
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001) if optimizer is None else optimizer
  for i in range(epochs):
    train_metric = trainOneEpochClassifier(model, train_dataloader, i, optimizer=optimizer)
    if val_dataloader is not None:
      val_metric = validateClassifier(model, val_dataloader, i)
    if metric_callback is not None:
      metric_callback(i, train_metric, val_metric)

def trainOneEpochClassifier(
      model: models.TheModel, train_dataloader: data.ClassDataLoader,
      epoch_index: int = 0, *, optimizer = None) -> Metric:
  return trainOneEpochOrValidateClassifier(model, train_dataloader, do_train=True, optimizer=optimizer, epoch_index=epoch_index)

def validateClassifier(
      model: models.TheModel, val_dataloader: data.ClassDataLoader,
      epoch_index: int = 0) -> Metric:
  return trainOneEpochOrValidateClassifier(model, val_dataloader, do_train=False, epoch_index=epoch_index)

def trainOneEpochOrValidateClassifier(
    model: models.TheModel, dataloader: data.ClassDataLoader, do_train = True,
    *, epoch_index: int = 0, optimizer = None) -> Metric:
  if do_train:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) if optimizer is None else optimizer
  sum_loss = 0
  corrects = 0
  confusion_matrix = [[0, 0], [0, 0]]
  for data in dataloader:
    batch_size = len(data[0])
    input_id, attn_mask, annotation, is_comp, elem_bmeo_mask, label = data
    outputs = model(input_id, attn_mask, annotation, elem_bmeo_mask)
    is_comparative_prob, elem_output, sentence_class_prob = outputs

    # print(elem_output)
    
    for i in range(batch_size):
      pred = float(is_comparative_prob[i]) >= 0.5
      is_correct = pred == bool(is_comp[i])
      print(is_comparative_prob[i], ":", is_comp[i], ":", is_correct)
      if is_correct:
        corrects += 1
      confusion_matrix[is_correct][pred] += 1
    # for i in range(batch_size):
    #   print (sentence_class_prob[i], ":", int(label[i]))
    #   print ("correct" if int(torch.argmax(sentence_class_prob[i])) == int(label[i]) else "wrong")
    
    if do_train:
      optimizer.zero_grad()
      if DO_TRAIN_PART1:
        is_comparative_cost = torch.nn.BCELoss()(is_comparative_prob[:,0], is_comp.float())
        is_comparative_cost.backward(retain_graph=DO_TRAIN_PART2)
      if DO_TRAIN_PART2:
        sum_positive = float(torch.sum(is_comp))
        for i in range(batch_size):
          if is_comp[i]:
            for elem in range(4):
              elem_pred, elem_cost = elem_output[elem]
              (elem_cost[i] / (4 * sum_positive)).backward(retain_graph=True)
            ce = torch.nn.CrossEntropyLoss()
            (ce(sentence_class_prob[i], label[i]) / sum_positive).backward(retain_graph=True)
      optimizer.step()

  tp = confusion_matrix[1][1]
  fp = confusion_matrix[0][1]
  fn = confusion_matrix[0][0]

  mt = Metric()
  mt.mean_loss = sum_loss / len(dataloader)
  mt.accuracy = corrects / len(dataloader.dataset)
  mt.precision = tp / (tp + fp) if tp + fp != 0 else None
  mt.recall = tp / (tp + fn) if tp + fn != 0 else None
  if mt.precision is None or mt.recall is None:
    mt.f1 = None
  else:
    mt.f1 = 2 * mt.precision * mt.recall / (mt.precision + mt.recall)
  return mt