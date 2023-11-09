import models
import data
import training
import config
import generate_result
from problem_spec import ELEMENTS, ELEMENTS_NO_LABEL, LABELS
from VnCoreNLP import VnCoreNLP
from processing import parseDataset

import os
import torch
import argparse
import numpy as np
from torchcrf import CRF
from transformers import AutoTokenizer, AutoModel

if __name__ == '__main__':
  config.loadConfig(config.makeParser().parse_args())
  torch.manual_seed(config.SEED)
  np.random.seed(config.SEED)

  train_data_path = config.DATA_PATH + "train/" + ("train_0001.txt" if config.IS_PROTOTYPE else "")
  val_data_path = config.DATA_PATH + "val/" + ("dev_0001.txt" if config.IS_PROTOTYPE else "")
  test_data_path = config.DATA_PATH + "test/" + ("test_0001.txt" if config.IS_PROTOTYPE else "")
  result_path = config.DATA_PATH + "result/"
  
  if config.MODE == "train":
    with VnCoreNLP(config.VNCORENLP_JAR_PATH) as vncorenlp:
      train_dataset = parseDataset(vncorenlp, train_data_path)
      train_dataset = data.COMDataSet(train_dataset)
      val_dataset = parseDataset(vncorenlp, val_data_path)
      val_dataset = data.COMDataSet(val_dataset)
    # test_dataset = parseDataset(tokenizer, test_data_path)
    # test_dataset = data.ClassDataSet(test_dataset)

    # class_count = [0] * len(LABELS)
    # for datapoint in train_dataset:
    #   if datapoint[2]:
    #     class_count[datapoint[4]] += 1
    # for datapoint in val_dataset:
    #   if datapoint[2]:
    #     class_count[datapoint[4]] += 1
    # print(class_count)

    # # print(len(train_dataset) + len(val_dataset) + len(test_dataset))

    train_dataloader = data.ClassDataLoader(train_dataset, config.BATCH_SIZE, sampler=data.BalancedSampler(train_dataset, positive_rate=config.POSTIVE_RATE, seed=config.SEED))
    val_dataloader = data.ClassDataLoader(val_dataset, 16)

    model = models.BertCrfExtractor().to(config.DEVICE)
    if config.LOAD_MODEL:
      if os.path.exists(config.LOAD_MODEL_PATH):
        model.load_state_dict(torch.load(config.LOAD_MODEL_PATH, map_location=config.DEVICE))
      else:
        raise Exception("Model path: " + config.LOAD_MODEL_PATH + " does not exist")

    max_f1 = 0
    def process_metric(epoch, train, val):
      global max_f1
      print(f"""{{Epoch {epoch}: 
train metric: ({", ".join([str(mt) for mt in train])})
val metric: ({", ".join([str(mt) for mt in val])})
}}""")
      if config.SAVE_MODEL and (config.DO_TRAIN_PART2 or val[1].f1 is not None and val[1].f1 > max_f1):
        torch.save(model.state_dict(), config.SAVE_MODEL_PATH)
        max_f1 = val[1].f1
    training.trainClassifier(model, train_dataloader, val_dataloader, epochs=config.EPOCHS, metric_callback=process_metric)

  elif config.MODE == "result":
    model = models.BertCrfExtractor().to(config.DEVICE)
    if os.path.exists(config.LOAD_MODEL_PATH):
      model.load_state_dict(torch.load(config.LOAD_MODEL_PATH, map_location=config.DEVICE))
    else:
      raise Exception("Model path: " + config.LOAD_MODEL_PATH + " does not exist")
    with VnCoreNLP(config.VNCORENLP_JAR_PATH) as vncorenlp:
      generate_result.generateResult(test_data_path, result_path, model=model, vncorenlp=vncorenlp)
  elif config.MODE == "demo":
    vncorenlp = VnCoreNLP(config.VNCORENLP_JAR_PATH)
    model = models.BertCrfExtractor().to(config.DEVICE)
    if config.LOAD_MODEL:
      if os.path.exists(config.LOAD_MODEL_PATH):
        model.load_state_dict(torch.load(config.LOAD_MODEL_PATH, map_location=config.DEVICE))
      else:
        raise Exception("Model path: " + config.LOAD_MODEL_PATH + " does not exist")
  else:
    raise Exception("Invalid mode")

else:
  raise Exception("This file was not created to be imported")