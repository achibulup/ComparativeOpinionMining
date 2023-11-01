import torch

VNCORENLP_JAR_PATH="dependencies/VnCoreNLP/VnCoreNLP.jar"
DATA_PATH="data/VLSP2023/"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SEED=1001
DO_TRAIN_PART1 = True
DO_TRAIN_PART2 = True