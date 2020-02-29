# model structure
MAX_LEN = 500
DROPOUT = 0.1
BEAM_WIDTH = 4
D_MODEL = 200
D_FF = 400
N = 6
H = 8


# training
BATCH_SIZE = 500


# load data
INPUT_TRAIN_FILE = "data/raw/fce/m2/fce.train.gold.bea19.m2"
OUTPUT_TRAIN_FILE = "data/processed/fce_train.json"

INPUT_DEV_FILE = "data/raw/fce/m2/fce.dev.gold.bea19.m2"
OUTPUT_DEV_FILE = "data/processed/fce_dev.json"

NMB_LINES = 100


# model
MODEL_PATH = "model/model_200_400"

