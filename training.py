import models
import torch
import data
import processing
import config
from processing import InputData
from transformers import AutoTokenizer
from problem_spec import LABELS
from metric import MetricRecord, BinaryMetric, MultiClassMetric

import time
import itertools
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

def predict(model: models.BertCrfExtractor, input_data: InputData) -> tuple[bool, list[dict]]:
  global tokenizer
  model.eval()
  dummy_label = processing.LabelData([])
  batch = processing.transformBatch([(input_data, dummy_label)], tokenizer)
  # input_id, attn_mask, annotation, is_comp, elem_bmeo_mask, label = batch

  # outputs = model(input_id, attn_mask, annotation, elem_bmeo_mask)
  # transformed_output = processing.part1Postprocess(outputs)[0]

  input_id, attn_mask, annotation, is_comp, elem_bmeo_mask, label = batch
  bertcrf_output = model.bertcrf(input_id, attn_mask, annotation, elem_bmeo_mask)
  is_comparative_prob, elem_output, token_embedding = bertcrf_output
  part1_output = processing.part1Postprocess(bertcrf_output)
  candidate_indexes: list[list[tuple[int, int]]] = []
  candidate_embedding: list[list[list[list[float]]]] = []
  quads_label: list[list[int]] = []
  for i in range(1):
    is_comparative, elements = part1_output[i]
    for elem, indexes in elements.items():
      if elem == "subject" or elem == "object" or elem == "aspect":
        indexes.append((-1, -1))
    candidate_indexes.append(list(itertools.product(*(elements.values()))))
    candidates = processing.generateCandiateQuadEmbedding(candidate_indexes[i], token_embedding[i, :, :])
    candidates_label = processing.generateCandidateQuadLabel(candidate_indexes[i], label[i])
    candidate_embedding.append(candidates)
    quads_label.append(torch.tensor(candidates_label, device=config.DEVICE))
  sentence_class_prob = model.classification(candidate_embedding)
  part2_output = processing.part2Postprocess(candidate_indexes, sentence_class_prob)[0]

  # input_id, attn_mask, annotation, is_comp, elem_bmeo_mask, label = batch
  # bertcrf_output = model.bertcrf(input_id, attn_mask, annotation, elem_bmeo_mask)
  # is_comparative_prob, elem_output, token_embedding = bertcrf_output
  # part1_output = processing.part1Postprocess(bertcrf_output)
  
  # is_comparative, elements = part1_output[0]
  # candidate_indexes = [itertools.product(*(elements.values()))]
  # candidates = processing.generateCandiateQuadEmbedding(candidate_indexes, token_embedding[0, :, :])
  # candidates_label = processing.generateCandidateQuadLabel(candidate_indexes, label[0])
  # candidate_embedding = [candidates]
  # quads_label = [candidates_label]
  
  # sentence_class_prob = model.classification(candidate_embedding)
  # part2_output = processing.part2Postprocess(candidate_indexes, sentence_class_prob)[0]

  return is_comparative, [processing.postprocess(out, input_data) for out in part2_output]


def identificationLoss(result: list[float], target: list[bool]):
  if (len(result) != len(target)):
    raise Exception("Result's length must be equal to target's length")
  batch_size = len(result)
  weight = torch.where(target, 1.2, 0.8)
  # weight = [None] * batch_size
  # for i in range(batch_size):
  #   weight[i] = 1.2 if target[i] else 0.8
  comp_loss_fn = torch.nn.BCELoss(weight)#torch.tensor(weight, device=config.DEVICE))      
  return comp_loss_fn(result, target.float())

def extractionLoss(crf_output: list[tuple[list, list[int]]], identification_label: list[bool]):
  if (len(crf_output[0][1]) != len(identification_label)):
    raise Exception("crf_output's 2nd dimension must be equal to identification_label's length")
  batch_size = len(crf_output[0][1])
  sum_loss = None
  sum_positive = 0
  for i in range(batch_size):
    # if identification_label[i]:
      sum_positive += 1
      for elem in range(4):
        elem_pred, elem_cost = crf_output[elem]
        sum_loss = sum_loss + elem_cost[i] if sum_loss is not None else elem_cost[i]
  if sum_loss is not None:
    sum_loss = sum_loss / (sum_positive * 4)
  return sum_loss

def classificationLoss(sentence_class_prob: list[list[list[float]]], label: list[list[int]], ident_label: list[bool]):
  if (len(sentence_class_prob) != len(label)):
    raise Exception("sentence_class_prob's length must be equal to label's length")
  ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.0967, 0.4514, 1.1151, 0.2872, 0.0610, 4.7393, 6.3191, 0.2562, 0.1], device=config.DEVICE))
  batch_size = len(sentence_class_prob)
  loss = None
  sum_positive = 0
  for i in range(batch_size):
    # if bool(ident_label[i]):
      sum_positive += len(label[i]) * int(ident_label[i])
      scale = 1 if ident_label[i] else 0.4
      if len(sentence_class_prob[i]) != 0:
        add = ce(sentence_class_prob[i], label[i]) * scale
        loss = loss + add if loss is not None else add
  if loss is not None and sum_positive != 0:
    loss = loss / sum_positive
    return loss
  else:
    return None


def trainClassifier(
    model: models.BertCrfExtractor, train_dataloader: data.ClassDataLoader, val_dataloader: data.ClassDataLoader | None, 
    *, epochs: int = 10, optimizer = None, metric_callback = None):
  
  optimizer = torch.optim.Adam(model.parameters(), lr=config.LR) if optimizer is None else optimizer

  for i in range(epochs):
    train_metric = trainOneEpochClassifier(model, train_dataloader, i, optimizer=optimizer)
    if val_dataloader is not None:
      val_metric = validateClassifier(model, val_dataloader, i)
    if metric_callback is not None:
      metric_callback(i, train_metric, val_metric)

def trainOneEpochClassifier(
      model: models.BertCrfExtractor, train_dataloader: data.ClassDataLoader,
      epoch_index: int = 0, *, optimizer = None):
  return trainOneEpochOrValidateClassifier(model, train_dataloader, do_train=True, optimizer=optimizer, epoch_index=epoch_index)

def validateClassifier(
      model: models.BertCrfExtractor, val_dataloader: data.ClassDataLoader,
      epoch_index: int = 0):
  return trainOneEpochOrValidateClassifier(model, val_dataloader, do_train=False, epoch_index=epoch_index)


def trainOneEpochOrValidateClassifier(
    model: models.BertCrfExtractor, dataloader: data.ClassDataLoader, do_train = True,
    *, epoch_index: int = 0, optimizer = None):
  global tokenizer

  if do_train and optimizer is None:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

  sum_loss = 0
  identify_metric = BinaryMetric()
  extract_metrics = [0 for _ in range(4)]
  class_metrics = MultiClassMetric(len(LABELS) + 1)
  all_positive = 0

  tt = time.perf_counter()
  def printPerf(name):
    if not config.LOG_PERF:
      return
    nonlocal tt
    newtt = time.perf_counter()
    print(name + ": ", newtt - tt)
    tt = newtt

  for batch_index, raw_batch in enumerate(dataloader):
    tt = time.perf_counter()
    batch = processing.transformBatch(raw_batch, tokenizer)
    printPerf("transformBatch")
    batch_size = len(batch[0])

    with torch.autograd.detect_anomaly():
      # evaluation
      input_id, attn_mask, annotation, is_comp, elem_bmeo_mask, label = batch
      bertcrf_output = model.bertcrf(input_id, attn_mask, annotation, elem_bmeo_mask)
      is_comparative_prob, elem_output, token_embedding = bertcrf_output
      printPerf("bertcrf")
      part1_output = processing.part1Postprocess(bertcrf_output)
      printPerf("part1Postprocess")
      if config.DO_TRAIN_PART2:
        candidate_indexes: list[list[tuple[int, int]]] = []
        candidate_embedding: list[list[list[list[float]]]] = []
        quads_label: list[list[int]] = []
        for i in range(batch_size):
          is_comparative, elements = part1_output[i]
          tmp = []
          for elem, indexes in elements.items():
            if elem == "subject" or elem == "object" or elem == "aspect":
              tmp.append(indexes + [(-1, -1)])
            else:
              tmp.append(indexes)
          candidate_indexes.append(list(itertools.product(*tmp)))
          candidates = processing.generateCandiateQuadEmbedding(candidate_indexes[i], token_embedding[i, :, :])
          candidates_label = processing.generateCandidateQuadLabel(candidate_indexes[i], label[i])
          candidate_embedding.append(candidates)
          quads_label.append(torch.tensor(candidates_label, device=config.DEVICE))
        printPerf("generateCandidateQuadEmbedding")
        sentence_class_prob = model.classification(candidate_embedding)
        printPerf("classification")
        part2_output = processing.part2Postprocess(candidate_indexes, sentence_class_prob, keep_negative=True)
        printPerf("part2Postprocess")
      #

      elem_bmeo_mask = elem_bmeo_mask.to("cpu")

      # learning
      if do_train:
        optimizer.zero_grad()

      identification_loss = identificationLoss(is_comparative_prob[:, 0], is_comp)
      is_comp = is_comp.to("cpu")
      extraction_loss = extractionLoss(elem_output, is_comp)
      printPerf("loss")
        
      part1_loss = identification_loss
      if extraction_loss is not None:
        part1_loss += extraction_loss
      sum_loss += part1_loss.item()
      if do_train and config.DO_TRAIN_PART1:
        part1_loss.backward(retain_graph=config.DO_TRAIN_PART2)
      printPerf("backward")


      if config.DO_TRAIN_PART2:
        part2_loss = classificationLoss(sentence_class_prob, quads_label, is_comp)
        printPerf("loss")
        if part2_loss is not None:
          sum_loss += part2_loss.item()
          if do_train:
            part2_loss.backward()
            printPerf("backward")

      if do_train:
        optimizer.step()
      #
      printPerf("learning")



      # metrics and logging
      # if batch_index == 0 and epoch_index % 10 == 5:
      #   print(bertcrf_output)
      #   for i, out in enumerate(part1_output):
      #     print(out)
      #   print("---")

      sum_positive = int(torch.sum(is_comp))
      all_positive += sum_positive
      is_comp_corrects = 0
      extract_corrects = [0 for _ in range(4)]
      class_corrects = 0
      sum_quads = 0
      for i in range(batch_size):
        binary_pred, elems = part1_output[i]
        is_correct = binary_pred == bool(is_comp[i])
        identify_metric.addSample(is_correct, binary_pred)
        is_comp_corrects += int(is_correct)

        if bool(is_comp[i]):
          for j, (elem, indexes) in enumerate(elems.items()):
            extract_target = processing.decodeList(elem_bmeo_mask[i, j, :])
            # print(extract_pred, extract_target)
            is_correct = indexes == extract_target
            extract_metrics[j] += int(is_correct)
            extract_corrects[j] += int(is_correct)
        
        if config.DO_TRAIN_PART2:
          sum_quads += len(quads_label[i])
          for pred, actual in zip(part2_output[i], quads_label[i]):
            class_metrics.addSample(actual, pred[-1])
            class_corrects += int(pred[-1] == actual) 
      printPerf("metrics")
      if config.LOG_PROGRESS:
        print("is_comp_corrects", is_comp_corrects, "/", batch_size)
        print("extract_corrects", extract_corrects, "/", sum_positive)
        print("class_corrects", class_corrects, "/", sum_quads)
      #
 
  avg_loss = sum_loss / len(dataloader.dataset)
  extract_metrics = [m / all_positive for m in extract_metrics]

  return avg_loss, identify_metric, extract_metrics, class_metrics