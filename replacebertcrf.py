import models
import config
import os
import torch

config.loadConfig(config.makeParser().parse_args())

model = models.BertCrfExtractor().to(config.DEVICE)
if os.path.exists(config.LOAD_MODEL_PATH):
  if os.path.exists(config.LOAD_MODEL_PATH + ".bertcrf"):
    model.load_state_dict(torch.load(config.LOAD_MODEL_PATH, map_location=config.DEVICE))
    model.bertcrf.load_state_dict(torch.load(config.LOAD_MODEL_PATH + ".bertcrf", map_location=config.DEVICE))
    torch.save(model.state_dict(), config.SAVE_MODEL_PATH)
  else :
    raise Exception("Model path: " + config.LOAD_MODEL_PATH + ".bertcrf does not exist")
else:
  raise Exception("Model path: " + config.LOAD_MODEL_PATH + " does not exist")
