from dotenv import load_dotenv
from VnCoreNLPTokenizer import VnCoreNLPTokenizer
import os
import glob
from processing import parseTrainDataset
import models
import data

from transformers import AutoModel, AutoTokenizer
import torch
import training

if __name__ == '__main__':
  load_dotenv()
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  data_path = os.getenv('DATA_PATH')
  train_data_path = data_path + "train/"
  train_dataset = parseTrainDataset(train_data_path)
  train_dataset = data.ClassDataSet(train_dataset)
  train_dataloader = data.ClassDataLoader(train_dataset, 4, shuffle=True)
  model = models.ClassifierModule().to(device)
  # print(training.trainClassifierOneEpoch(model, train_dataloader, 0))

else:
  raise Exception("This file was not created to be imported")