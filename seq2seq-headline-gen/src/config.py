"""
Central config for the headline-generation seq2seq project.
Keep every tunable knob here so runs are reproducible and diffable.
"""
import torch

SEED = 42

# ---- Data ----
RAW_DATA_PATH = "data/news_summary.csv"     # article text + headline columns
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

MAX_SRC_LEN = 120     # article tokens (truncated)
MAX_TGT_LEN = 25       # headline tokens
MIN_VOCAB_FREQ = 2     # drop tokens seen fewer than this many times
VOCAB_SIZE_CAP = 30000

# ---- Model ----
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_ENCODER_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.3
ATTENTION_TYPE = "bahdanau"   # additive attention

# ---- Training ----
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
NUM_EPOCHS = 15
CLIP_GRAD_NORM = 1.0
TEACHER_FORCING_RATIO = 0.5
EARLY_STOP_PATIENCE = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Paths ----
CHECKPOINT_DIR = "checkpoints"
VOCAB_PATH = "checkpoints/vocab.json"

# ---- LLM baseline ----
OLLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M"   # or "mistral:7b-instruct"
LLM_TEST_SUBSET_SIZE = 1000   # set an int to subsample test set for cheaper LLM runs; None = full test set
