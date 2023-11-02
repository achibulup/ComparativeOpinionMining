import torch
import argparse

DO_TRAIN_PART1: bool
DO_TRAIN_PART2: bool
BATCH_SIZE: int
EPOCHS: int
SEED: int
POSTIVE_RATE: float
LR: float
BINARY_WEIGHT: float
VNCORENLP_JAR_PATH: str
DATA_PATH: str
DEVICE: torch.device
SAVE_MODEL: bool
LOAD_MODEL: bool
LOAD_MODEL_PATH: str
SAVE_MODEL_PATH: str

def makeParser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", help="batch size", type=int, default=16)
  parser.add_argument("--epochs", help="number of epochs", type=int, default=20)
  parser.add_argument("--positive_rate", help="positive rate for balanced sampler", type=float, default=0.4)
  parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
  parser.add_argument("--binary_weight", help="weight ratio for binary classification", type=float, default=1.2/0.8)
  parser.add_argument("--seed", help="random seed", type=int, default=999)
  parser.add_argument("--do_train_part1", help="whether to train part 1", type=bool, default=True)
  parser.add_argument("--do_train_part2", help="whether to train part 2", type=bool, default=True)
  parser.add_argument("--vncorenlp_path", help="path to vncorenlp jar file", type=str, default="dependencies/VnCoreNLP/VnCoreNLP.jar")
  parser.add_argument("--data_path", help="path to data folder, which should contain train and dev subfolder containing .txt data files", type=str, default="data/VLSP2023/")
  parser.add_argument("--device", help="device to run on", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--save_model", help="whether to save model", type=bool, default=True)
  parser.add_argument("--load_model", help="whether to load model", type=bool, default=False)
  parser.add_argument("--model_path", help="path to load and save model", type=str, default="models/model.pt")
  parser.add_argument("--load_model_path", help="path to load model", type=str, default=None)
  parser.add_argument("--save_model_path", help="path to save model", type=str, default=None)
  return parser

def loadConfig(args):
  global DO_TRAIN_PART1, DO_TRAIN_PART2, BATCH_SIZE, EPOCHS, SEED, POSTIVE_RATE, LR, BINARY_WEIGHT, VNCORENLP_JAR_PATH, DATA_PATH, DEVICE, SAVE_MODEL, LOAD_MODEL, LOAD_MODEL_PATH, SAVE_MODEL_PATH
  VNCORENLP_JAR_PATH = args.vncorenlp_path
  DATA_PATH = args.data_path
  DEVICE = torch.device(args.device)
  SEED = args.seed
  SAVE_MODEL = args.save_model
  LOAD_MODEL = args.load_model
  LOAD_MODEL_PATH = args.model_path
  if (args.load_model_path is not None):
    LOAD_MODEL_PATH = args.load_model_path
  SAVE_MODEL_PATH = args.model_path
  if (args.save_model_path is not None):
    SAVE_MODEL_PATH = args.save_model_path
  DO_TRAIN_PART1 = args.do_train_part1
  DO_TRAIN_PART2 = args.do_train_part2
  BATCH_SIZE = args.batch_size
  EPOCHS = args.epochs
  POSTIVE_RATE = args.positive_rate
  LR = args.lr
  BINARY_WEIGHT = args.binary_weight

