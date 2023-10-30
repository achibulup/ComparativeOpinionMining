from dotenv import load_dotenv
from VnCoreNLP import VnCoreNLP
import os
from processing import parseDataset
import models
import data
import torch
import training

if __name__ == '__main__':
  load_dotenv()
  data_path = os.getenv('DATA_PATH')
  vncorenlp_path = os.getenv('VNCORENLP_JAR_PATH')

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  tokenizer = VnCoreNLP(vncorenlp_path)
  train_data_path = data_path + "train/"
  val_data_path = data_path + "val/train_0049.txt"

  train_dataset = parseDataset(tokenizer, train_data_path)
  train_dataset = data.ClassDataSet(train_dataset)
  val_dataset = parseDataset(tokenizer, val_data_path)
  val_dataset = data.ClassDataSet(val_dataset)

  train_dataloader = data.ClassDataLoader(train_dataset, 10, shuffle=True)
  val_dataloader = data.ClassDataLoader(val_dataset, 10)

  model = models.ClassifierModule().to(device)

  training.trainClassifier(model, train_dataloader, val_dataloader, epochs=2, metric_callback=lambda epoch, train, val: 
    print(f"Epoch {epoch}: \n train metric: {train} \n val metric: {val}")
  )

else:
  raise Exception("This file was not created to be imported")