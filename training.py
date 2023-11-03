import models
import torch
import data
import processing
import config
from transformers import AutoTokenizer
from problem_spec import LABELS
from metric import MetricRecord, BinaryMetric, MultiClassMetric


tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

def predict(model: models.TheModel, sentence: str, nlptokenizer) -> str:
  global tokenizer
  model.eval()
  processed_input = processing.mineInputSentence(sentence, nlptokenizer)
  dummy_label = processing.LabelData([])
  batch = processing.transformBatch([(processed_input, dummy_label)], tokenizer)
  input_id, attn_mask, annotation, is_comp, elem_bmeo_mask, label = batch

  outputs = model(input_id, attn_mask, annotation, elem_bmeo_mask)
  is_comparative_prob, elem_output, sentence_class_prob = outputs
  transformed_output = processing.detransformResult(outputs, [(processed_input, dummy_label)])[0]
  return processing.formatResult(transformed_output, processed_input)

def trainClassifier(
    model: models.TheModel, train_dataloader: data.ClassDataLoader, val_dataloader: data.ClassDataLoader | None, 
    *, epochs: int = 10, optimizer = None, metric_callback = None):
  
  optimizer = torch.optim.Adam(model.parameters(), lr=config.LR) if optimizer is None else optimizer

  for i in range(epochs):
    train_metric = trainOneEpochClassifier(model, train_dataloader, i, optimizer=optimizer)
    if val_dataloader is not None:
      val_metric = validateClassifier(model, val_dataloader, i)
    if metric_callback is not None:
      metric_callback(i, train_metric, val_metric)

def trainOneEpochClassifier(
      model: models.TheModel, train_dataloader: data.ClassDataLoader,
      epoch_index: int = 0, *, optimizer = None):
  return trainOneEpochOrValidateClassifier(model, train_dataloader, do_train=True, optimizer=optimizer, epoch_index=epoch_index)

def validateClassifier(
      model: models.TheModel, val_dataloader: data.ClassDataLoader,
      epoch_index: int = 0):
  return trainOneEpochOrValidateClassifier(model, val_dataloader, do_train=False, epoch_index=epoch_index)

def trainOneEpochOrValidateClassifier(
    model: models.TheModel, dataloader: data.ClassDataLoader, do_train = True,
    *, epoch_index: int = 0, optimizer = None):
  global tokenizer

  if do_train:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR) if optimizer is None else optimizer

  g_sum_loss = 0
  bin_class_metric = BinaryMetric()
  class_metrics = MultiClassMetric(len(LABELS))
  first_batch = True
  for raw_batch in dataloader:
    batch = processing.transformBatch(raw_batch, tokenizer)
    batch_size = len(batch[0])

    #
    input_id, attn_mask, annotation, is_comp, elem_bmeo_mask, label = batch
    outputs = model(input_id, attn_mask, annotation, elem_bmeo_mask)
    is_comparative_prob, elem_output, sentence_class_prob = outputs
    #

    if first_batch and epoch_index % 5 == 4:
      print(outputs)
      transformed_output = processing.detransformResult(outputs, raw_batch)
      for out in transformed_output:
        print(out)
      print("---")

    sum_positive = int(torch.sum(is_comp))
    
    is_comp_corrects = 0
    for i in range(batch_size):
      pred = float(is_comparative_prob[i]) >= 0.5
      is_correct = pred == bool(is_comp[i])
      # print(is_comparative_prob[i].item(), ":", is_comp[i], ":", is_correct)
      bin_class_metric.addSample(is_correct, pred)
      is_comp_corrects += int(is_correct)
    print("is_comp_corrects:", is_comp_corrects, "/", batch_size)
    
    class_corrects = 0
    for i in range(batch_size):
      if bool(is_comp[i]):
        pred = int(torch.argmax(sentence_class_prob[i]))
        actual = int(label[i])
        class_metrics.addSample(actual, pred)
        if pred == actual:
          class_corrects += 1
    print("class_corrects:", class_corrects, "/", sum_positive)

    
    #weight=torch.tensor([0.9, 1.1]))
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.0967, 0.4514, 1.1151, 0.2872, 0.0610, 4.7393, 6.3191, 0.2562]).to(config.DEVICE))
    
    if do_train:
      optimizer.zero_grad()

    weight=[]
    for i in range(batch_size):
      weight.append(1.2 if is_comp[i] else 0.8) 
    comp_loss_fn = torch.nn.BCELoss(torch.tensor(weight).to(config.DEVICE))      
    is_comparative_cost = comp_loss_fn(is_comparative_prob[:,0], is_comp.float())
    g_sum_loss += is_comparative_cost.item()
    if do_train and config.DO_TRAIN_PART1:
      is_comparative_cost.backward(retain_graph=config.DO_TRAIN_PART2)
      
    sum_loss = None
    for i in range(batch_size):
      if is_comp[i]:
        for elem in range(4):
          elem_pred, elem_cost = elem_output[elem]
          sum_loss = sum_loss + elem_cost[i] if sum_loss is not None else elem_cost[i]
        add = ce(sentence_class_prob[i], label[i]) * 3
        sum_loss = sum_loss + add if sum_loss is not None else add
    if sum_loss is not None:
      sum_loss = sum_loss / (sum_positive * 4)
      g_sum_loss += sum_loss.item()
      if do_train and config.DO_TRAIN_PART2:
        sum_loss.backward()

    if do_train:
      optimizer.step()

    first_batch = False
 
  avg_loss = g_sum_loss / len(dataloader.dataset)

  return g_sum_loss, bin_class_metric, class_metrics