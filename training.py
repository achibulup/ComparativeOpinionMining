import models
import torch
import data
import processing
import config
from transformers import AutoTokenizer
from problem_spec import LABELS
from metric import MetricRecord, BinaryMetric, MultiClassMetric


tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

def predict(model: models.BertCrfCell, sentence: str, nlptokenizer) -> str:
  global tokenizer
  model.eval()
  processed_input = processing.mineInputSentence(sentence, nlptokenizer)
  dummy_label = processing.LabelData([])
  batch = processing.transformBatch([(processed_input, dummy_label)], tokenizer)
  input_id, attn_mask, annotation, is_comp, elem_bmeo_mask, label = batch

  outputs = model(input_id, attn_mask, annotation, elem_bmeo_mask)
  transformed_output = processing.part1Postprocess(outputs)[0]
  return processing.formatResult(transformed_output, processed_input)


def identificationLoss(result: list[float], target: list[bool]):
  if (len(result) != len(target)):
    raise Exception("Result's length must be equal to target's length")
  batch_size = len(result)
  weight=[]
  for i in range(batch_size):
    weight.append(1.2 if target[i] else 0.8) 
  comp_loss_fn = torch.nn.BCELoss(torch.tensor(weight).to(config.DEVICE))      
  return comp_loss_fn(result, target.float())

def extractionLoss(crf_output: list[tuple[list, list[int]]], identification_label: list[bool]):
  if (len(crf_output[0][1]) != len(identification_label)):
    raise Exception("crf_output's 2nd dimension must be equal to identification_label's length")
  batch_size = len(crf_output)
  sum_loss = None
  sum_positive = 0
  for i in range(batch_size):
    if identification_label[i]:
      sum_positive += 1
      for elem in range(4):
        elem_pred, elem_cost = crf_output[elem]
        sum_loss = sum_loss + elem_cost[i] if sum_loss is not None else elem_cost[i]
  if sum_loss is not None:
    sum_loss = sum_loss / (sum_positive * 4)
  return sum_loss

def classificationLoss(sentence_class_prob: list[list[float]], label: list[int]):
  if (len(sentence_class_prob) != len(label)):
    raise Exception("sentence_class_prob's length must be equal to label's length")
  ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.0967, 0.4514, 1.1151, 0.2872, 0.0610, 4.7393, 6.3191, 0.2562]).to(config.DEVICE))
  batch_size = len(sentence_class_prob)
  loss = None
  sum_positive = 0
  for i in range(batch_size):
    if label[i] != -1:
      sum_positive += 1
      add = ce(sentence_class_prob[i], label[i])
      loss = loss + add if loss is not None else add
  if loss is not None:
    loss = loss / sum_positive
  return loss


def trainClassifier(
    model: models.BertCrfCell, train_dataloader: data.ClassDataLoader, val_dataloader: data.ClassDataLoader | None, 
    *, epochs: int = 10, optimizer = None, metric_callback = None):
  
  optimizer = torch.optim.Adam(model.parameters(), lr=config.LR) if optimizer is None else optimizer

  for i in range(epochs):
    train_metric = trainOneEpochClassifier(model, train_dataloader, i, optimizer=optimizer)
    if val_dataloader is not None:
      val_metric = validateClassifier(model, val_dataloader, i)
    if metric_callback is not None:
      metric_callback(i, train_metric, val_metric)

def trainOneEpochClassifier(
      model: models.BertCrfCell, train_dataloader: data.ClassDataLoader,
      epoch_index: int = 0, *, optimizer = None):
  return trainOneEpochOrValidateClassifier(model, train_dataloader, do_train=True, optimizer=optimizer, epoch_index=epoch_index)

def validateClassifier(
      model: models.BertCrfCell, val_dataloader: data.ClassDataLoader,
      epoch_index: int = 0):
  return trainOneEpochOrValidateClassifier(model, val_dataloader, do_train=False, epoch_index=epoch_index)


def trainOneEpochOrValidateClassifier(
    model: models.BertCrfCell, dataloader: data.ClassDataLoader, do_train = True,
    *, epoch_index: int = 0, optimizer = None):
  global tokenizer

  if do_train and optimizer is None:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

  sum_loss = 0
  bin_class_metric = BinaryMetric()
  class_metrics = MultiClassMetric(len(LABELS))

  for batch_index, raw_batch in enumerate(dataloader):
    batch = processing.transformBatch(raw_batch, tokenizer)
    batch_size = len(batch[0])

    #
    input_id, attn_mask, annotation, is_comp, elem_bmeo_mask, label = batch
    outputs = model(input_id, attn_mask, annotation, elem_bmeo_mask)
    is_comparative_prob, elem_output, sentence_class_prob = outputs
    transformed_output = processing.part1Postprocess(outputs)
    #

    if batch_index == 0 and True:# epoch_index % 5 == 4:
      print(outputs)
      for i, out in enumerate(transformed_output):
        print(out)
        print(processing.formatResult(out, raw_batch[i][0]))
      print("---")

    sum_positive = int(torch.sum(is_comp))
    
    is_comp_corrects = 0
    for i in range(batch_size):
      pred = transformed_output[i][0]
      is_correct = pred == bool(is_comp[i])
      bin_class_metric.addSample(is_correct, pred)
      is_comp_corrects += int(is_correct)
    if config.LOG_PROGRESS:
      print("is_comp_corrects:", is_comp_corrects, "/", batch_size)
    
    class_corrects = 0
    for i in range(batch_size):
      if bool(is_comp[i]):
        pred = transformed_output[i][2]
        actual = int(label[i])
        class_metrics.addSample(actual, pred)
        class_corrects += int(pred == actual)
    if config.LOG_PROGRESS:
      print("class_corrects:", class_corrects, "/", sum_positive)

    
    if do_train:
      optimizer.zero_grad()

    identifcation_loss = identificationLoss(is_comparative_prob[:, 0], is_comp)
    sum_loss += identifcation_loss.item()
    if do_train and config.DO_TRAIN_PART1:
      identifcation_loss.backward(retain_graph=config.DO_TRAIN_PART2)
      
    extraction_loss = extractionLoss(elem_output, is_comp)
    classification_loss = classificationLoss(sentence_class_prob, label)
    part2_loss = None
    if extraction_loss is not None:
      part2_loss = extraction_loss
    if classification_loss is not None:
      part2_loss = part2_loss + classification_loss if part2_loss is not None else classification_loss
    if part2_loss is not None:
      sum_loss += part2_loss.item()
      if do_train and config.DO_TRAIN_PART2:
        part2_loss.backward()

    if do_train:
      optimizer.step()
 
  avg_loss = sum_loss / len(dataloader.dataset)

  return avg_loss, bin_class_metric, class_metrics