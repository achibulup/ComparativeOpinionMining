import models
import config

import os
import torch

config.loadConfig(config.makeParser().parse_args())

model = models.BertCrfExtractor().to(config.DEVICE)
if os.path.exists(config.LOAD_MODEL_PATH):
  model.load_state_dict(torch.load(config.LOAD_MODEL_PATH, map_location=config.DEVICE))
  torch.save(model.classification.state_dict(), config.SAVE_MODEL_PATH + ".classify")
else:
  raise Exception("Model path: " + config.LOAD_MODEL_PATH + " does not exist")
