import models
import torch
import data

class Metric:
  mean_loss: float | None
  accuracy: float | None
  precision: float | None
  recall: float | None
  f1: float | None
  def __init__(self):
    self.mean_loss = None
    self.accuracy = None
    self.precision = None
    self.recall = None
    self.f1 = None
  def __str__(self):
    return f"mean_loss: {self.mean_loss}, accuracy: {self.accuracy}, precision: {self.precision}, recall: {self.recall}, f1: {self.f1}"

def trainClassifier(
    model: models.ClassifierModule, train_dataloader: data.ClassDataLoader, val_dataloader: data.ClassDataLoader | None, 
    *, epochs: int = 10, optimizer = None, loss_fn = torch.nn.BCELoss(), metric_callback = None):
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001) if optimizer is None else optimizer
  for i in range(epochs):
    train_metric = trainOneEpochClassifier(model, train_dataloader, i, optimizer=optimizer, loss_fn=loss_fn)
    if val_dataloader is not None:
      val_metric = validateClassifier(model, val_dataloader, i, loss_fn=loss_fn)
    if metric_callback is not None:
      metric_callback(i, train_metric, val_metric)

def trainOneEpochClassifier(
      model: models.ClassifierModule, train_dataloader: data.ClassDataLoader,
      epoch_index: int = 0, *, optimizer = None, loss_fn = torch.nn.BCELoss()) -> Metric:
  return trainOneEpochOrValidateClassifier(model, train_dataloader, do_train=True, optimizer=optimizer, loss_fn=loss_fn, epoch_index=epoch_index)

def validateClassifier(
      model: models.ClassifierModule, val_dataloader: data.ClassDataLoader,
      epoch_index: int = 0, *, loss_fn = torch.nn.BCELoss()) -> Metric:
  return trainOneEpochOrValidateClassifier(model, val_dataloader, do_train=False, loss_fn=loss_fn, epoch_index=epoch_index)

def trainOneEpochOrValidateClassifier(
    model: models.ClassifierModule, dataloader: data.ClassDataLoader, do_train = True,
    *, epoch_index: int = 0, optimizer = None, loss_fn = torch.nn.BCELoss()) -> Metric:
  if do_train:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) if optimizer is None else optimizer
  sum_loss = 0
  corrects = 0
  confusion_matrix = [[0, 0], [0, 0]]
  for data in dataloader:
    input_ids, attn_mask, labels = data
    outputs = model(input_ids, attn_mask)
    print(outputs)
    print(labels)
    pred = torch.argmax(outputs, dim=1)
    for i in range(len(pred)):
      is_correct = int(labels[i][pred[i]])
      corrects += is_correct
      confusion_matrix[is_correct][pred[i]] += 1
      
    if do_train:
      optimizer.zero_grad()

    loss = loss_fn(outputs, labels)
    sum_loss += loss.item()
    print(loss)
    
    if do_train:
      loss.backward()
      optimizer.step()

  tp = confusion_matrix[1][1]
  fp = confusion_matrix[0][1]
  fn = confusion_matrix[1][0]

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