import models
import config
import os
import torch

config.loadConfig(config.makeParser().parse_args())

model = models.BertCrfExtractor().to(config.DEVICE)
if os.path.exists(config.LOAD_MODEL_PATH):
  if os.path.exists(config.LOAD_MODEL_PATH + ".classify"):
    model.load_state_dict(torch.load(config.LOAD_MODEL_PATH, map_location=config.DEVICE))
    model.classification.load_state_dict(torch.load(config.LOAD_MODEL_PATH + ".classify", map_location=config.DEVICE))
    torch.save(model.state_dict(), config.SAVE_MODEL_PATH)
  else :
    raise Exception("Model path: " + config.LOAD_MODEL_PATH + ".classify does not exist")
else:
  raise Exception("Model path: " + config.LOAD_MODEL_PATH + " does not exist")
