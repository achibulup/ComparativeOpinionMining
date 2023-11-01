from dotenv import load_dotenv
from VnCoreNLP import VnCoreNLP
import os
from processing import parseDataset
import models
import data
import torch
import training
from torchcrf import CRF
from transformers import AutoTokenizer, AutoModel
from config import DEVICE, SEED, VNCORENLP_JAR_PATH, DATA_PATH
import numpy as np

if __name__ == '__main__':
  # load_dotenv()
  # data_path = os.getenv('DATA_PATH')
  # vncorenlp_path = os.getenv('VNCORENLP_JAR_PATH')
  torch.manual_seed(SEED)
  np.random.seed(SEED)

  train_data_path = DATA_PATH + "train/"
  val_data_path = DATA_PATH + "val/"
  test_data_path = DATA_PATH + "test/"
  
  with VnCoreNLP(VNCORENLP_JAR_PATH) as vncorenlp:
    train_dataset = parseDataset(vncorenlp, train_data_path)
    train_dataset = data.COMDataSet(train_dataset)
    val_dataset = parseDataset(vncorenlp, val_data_path)
    val_dataset = data.COMDataSet(val_dataset)
  # # test_dataset = parseDataset(tokenizer, test_data_path)
  # # test_dataset = data.ClassDataSet(test_dataset)

  # # print(len(train_dataset) + len(val_dataset) + len(test_dataset))

  train_dataloader = data.ClassDataLoader(train_dataset, 16, sampler=data.BalancedSampler(train_dataset, positive_rate=0.4, seed=SEED))
  val_dataloader = data.ClassDataLoader(val_dataset, 16)

  model = models.TheModel().to(DEVICE)
  # if os.path.exists("model.pt"):
  #   model.load_state_dict(torch.load("model.pt"))

  training.trainClassifier(model, train_dataloader, val_dataloader, epochs=20, metric_callback=lambda epoch, train, val: 
    print(f"Epoch {epoch}: \n train metric: {train} \n val metric: {val}")
  )
# , loss_fn=torch.nn.BCELoss(weight=torch.tensor([0.3, 1.7]))
else:
  raise Exception("This file was not created to be imported")