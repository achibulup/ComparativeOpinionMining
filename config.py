import torch
import argparse

IS_PROTOTYPE: bool = False
LIGHT_VAL: bool = False
MODE: str = "train"
DO_TRAIN_PART1: bool = True
DO_TRAIN_PART2: bool = True
BATCH_SIZE: int = 16
EPOCHS: int = 20
SEED: int = 999
POSTIVE_RATE: float = 0.4
LOG_PROGRESS: bool = True
LOG_PERF: bool = False
LR: float = 0.001
BINARY_WEIGHT: float = 1.2/0.8
VNCORENLP_JAR_PATH: str = "dependencies/VnCoreNLP/VnCoreNLP.jar"
DATA_PATH: str = "data/VLSP2023/"
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODEL: bool = True
LOAD_MODEL: bool = False
LOAD_MODEL_PATH: str = "models/model.pt"
SAVE_MODEL_PATH: str = "models/model.pt"

def makeParser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--prototype", help="whether to use prototype", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument("--light-val", help="whether to use light val", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument("--mode", help="'train' or 'demo' or 'result'", type=str, default="train")
  parser.add_argument("--batch-size", help="batch size", type=int, default=16)
  parser.add_argument("--epochs", help="number of epochs", type=int, default=20)
  parser.add_argument("--positive-rate", help="positive rate for balanced sampler", type=float, default=0.4)
  parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
  parser.add_argument("--binary-weight", help="weight ratio for binary classification", type=float, default=1.2/0.8)
  parser.add_argument("--seed", help="random seed", type=int, default=999)
  parser.add_argument("--do-train-part1", help="whether to train part 1", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--do-train-part2", help="whether to train part 2", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--log-progress", help="whether to log ...", default=True, action=argparse.BooleanOptionalAction)
  parser.add_argument("--log-perf", help="whether to log performance", default=False, action=argparse.BooleanOptionalAction)
  parser.add_argument("--vncorenlp-path", help="path to vncorenlp jar file", type=str, default="dependencies/VnCoreNLP/VnCoreNLP.jar")
  parser.add_argument("--data-path", help="path to data folder, which should contain train and dev subfolder containing .txt data files", type=str, default="data/VLSP2023/")
  parser.add_argument("--device", help="device to run on", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--save-model", help="whether to save model", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--load-model", help="whether to load model", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument("--model-path", help="path to load and save model", type=str, default="models/model.pt")
  parser.add_argument("--load-model-path", help="path to load model", type=str, default=None)
  parser.add_argument("--save-model-path", help="path to save model", type=str, default=None)
  return parser

def loadConfig(args):
  global IS_PROTOTYPE, LIGHT_VAL, MODE, DO_TRAIN_PART1, DO_TRAIN_PART2, BATCH_SIZE, EPOCHS, SEED, LOG_PROGRESS, LOG_PERF, POSTIVE_RATE, LR, BINARY_WEIGHT, VNCORENLP_JAR_PATH, DATA_PATH, DEVICE, SAVE_MODEL, LOAD_MODEL, LOAD_MODEL_PATH, SAVE_MODEL_PATH
  if args.mode not in ["train", "demo", "result"]:
    raise Exception("Invalid mode")
  IS_PROTOTYPE = args.prototype
  LIGHT_VAL = args.light_val
  MODE = args.mode
  VNCORENLP_JAR_PATH = args.vncorenlp_path
  DATA_PATH = args.data_path
  DEVICE = torch.device(args.device)
  SEED = args.seed
  LOG_PROGRESS = args.log_progress
  LOG_PERF = args.log_perf
  LOAD_MODEL = args.load_model
  SAVE_MODEL = args.save_model
  if MODE == "result" or MODE == "demo" or args.load_model_path is not None:
    LOAD_MODEL = True
  if args.save_model_path is not None:
    SAVE_MODEL = True
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


