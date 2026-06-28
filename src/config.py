# [0] Base config — hyperparameters, file paths, constants. Imported by every other file.

from pathlib import Path
import torch


DATA_DIR = Path("data")
RAW_CSV = Path("dataset") / "nl2rx.csv"
TRAIN_JSON = DATA_DIR / "train.json"
VAL_JSON = DATA_DIR / "val.json"
TEST_JSON = DATA_DIR / "test.json"

d_model = 256       # dimension of your model
n_heads = 4         # This is number of heads in multi-head attention
d_ff = 1024 
n_layers = 4        # Number of encoder and decoder layers - 4 each
dropout = 0.2       # Original paper used 0.1, 0.2 here because we have less data. Prevents overfitting (regularization)

vocab_nl = 4000     # Natural Language vocabulary size for BPE
vocab_regex = 256   # Regex vocabulary size when using BPE
max_len_nl = 64     # Maximum length of natural language sequences (padding for batching)
max_len_regex = 32  # Same shit

batch_size = 32
max_lr = 5e-4       # learning rate bruv
warmup_steps = 4000
epochs = 30
label_smoothing = 0.1

pad_id_nl = 0
pad_id_regex = 0
sos_id_regex = 1
eos_id_regex = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
