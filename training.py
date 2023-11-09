import models
import torch
import data
import processing
import config
from processing import InputData
from transformers import AutoTokenizer
from problem_spec import LABELS
from metric import MetricRecord, BinaryMetric, MultiClassMetric

import itertools

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
    quads_label.append(torch.tensor(candidates_label).to(config.DEVICE))
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
  weight=[]
  for i in range(batch_size):
    weight.append(1.2 if target[i] else 0.8) 
  comp_loss_fn = torch.nn.BCELoss(torch.tensor(weight).to(config.DEVICE))      
  return comp_loss_fn(result, target.float())

def extractionLoss(crf_output: list[tuple[list, list[int]]], identification_label: list[bool]):
  if (len(crf_output[0][1]) != len(identification_label)):
    raise Exception("crf_output's 2nd dimension must be equal to identification_label's length")
  batch_size = len(crf_output[0][1])
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

def classificationLoss(sentence_class_prob: list[list[list[float]]], label: list[list[int]], ident_label: list[bool]):
  if (len(sentence_class_prob) != len(label)):
    raise Exception("sentence_class_prob's length must be equal to label's length")
  ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.0967, 0.4514, 1.1151, 0.2872, 0.0610, 4.7393, 6.3191, 0.2562, 3]).to(config.DEVICE))
  batch_size = len(sentence_class_prob)
  loss = None
  sum_positive = 0
  for i in range(batch_size):
    if bool(ident_label[i]):
      sum_positive += len(label[i])
      if len(sentence_class_prob[i]) != 0:
        add = ce(sentence_class_prob[i], label[i]) * len(label[i])
        loss = loss + add if loss is not None else add
  if loss is not None:
    loss = loss / sum_positive
  return loss


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
  class_metrics = MultiClassMetric(len(LABELS))
  all_positive = 0

  for batch_index, raw_batch in enumerate(dataloader):
    batch = processing.transformBatch(raw_batch, tokenizer)
    batch_size = len(batch[0])

    # evaluation
    input_id, attn_mask, annotation, is_comp, elem_bmeo_mask, label = batch
    bertcrf_output = model.bertcrf(input_id, attn_mask, annotation, elem_bmeo_mask)
    is_comparative_prob, elem_output, token_embedding = bertcrf_output
    part1_output = processing.part1Postprocess(bertcrf_output)
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
        quads_label.append(torch.tensor(candidates_label).to(config.DEVICE))
      sentence_class_prob = model.classification(candidate_embedding)
      part2_output = processing.part2Postprocess(candidate_indexes, sentence_class_prob)
    #


    # learning
    if do_train:
      optimizer.zero_grad()

    identification_loss = identificationLoss(is_comparative_prob[:, 0], is_comp)
    extraction_loss = extractionLoss(elem_output, is_comp)
      
    part1_loss = identification_loss
    if extraction_loss is not None:
      part1_loss += extraction_loss
    sum_loss += part1_loss.item()
    if do_train and config.DO_TRAIN_PART1:
      part1_loss.backward(retain_graph=config.DO_TRAIN_PART2)

    if config.DO_TRAIN_PART2:
      part2_loss = classificationLoss(sentence_class_prob, quads_label, is_comp)
      if part2_loss is not None:
        sum_loss += part2_loss.item()
        if do_train:
          part2_loss.backward()

    if do_train:
      optimizer.step()
    #



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
    for i in range(batch_size):
      binary_pred, elems = part1_output[i]
      is_correct = binary_pred == bool(is_comp[i])
      identify_metric.addSample(is_correct, binary_pred)
      is_comp_corrects += int(is_correct)

      if bool(is_comp[i]):
        for j, (elem, indexes) in enumerate(elems.items()):
          extract_target = processing.decodeList(elem_bmeo_mask[i, j, :])
          extract_pred = elems[elem]
          # print(extract_pred, extract_target)
          is_correct = extract_pred == extract_target
          extract_metrics[j] += int(is_correct)
          extract_corrects[j] += int(is_correct)
        
      #   class_pred = part1_output[i][2]
      #   actual = int(label[i])
      #   class_metrics.addSample(actual, class_pred)
      #   class_corrects += int(pred == actual)
    if config.LOG_PROGRESS:
      print("is_comp_corrects:", is_comp_corrects, "/", batch_size)
      print("extract_corrects:", extract_corrects, "/", sum_positive)
      # print("class_corrects:", class_corrects, "/", sum_positive)
    #
 
  avg_loss = sum_loss / len(dataloader.dataset)
  extract_metrics = [m / all_positive for m in extract_metrics]

  return avg_loss, identify_metric, extract_metrics