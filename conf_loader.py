# model structure
MAX_LEN = 500
DROPOUT = 0.1
BEAM_WIDTH = 4
D_MODEL = 200
D_FF = 400
N = 6
H = 8


# training
BATCH_SIZE = 100


# load data
INPUT_TRAIN_FILE = "data/raw/fce/json/fce.train.json"
OUTPUT_TRAIN_FILE = "data/processed/fce_processed_train.json"

INPUT_DEV_FILE = "data/raw/fce/json/fce.dev.json"
OUTPUT_DEV_FILE = "data/processed/fce_processed_dev.json"

NMB_LINES = 100


# model
MODEL_PATH = "model/model_200_400"

