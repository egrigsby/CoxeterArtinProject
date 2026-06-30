import os
from pathlib import Path

_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

LOAD_MODEL   = False
DATA_PATH    = _DIR
PTH_LOCATION = str(_DIR / "workspace/_scratch/model.pth")

DATA_SEED      = 598
TRAINING_SPLIT = 0.4
DATA_CSV       = "data.csv"

NUM_EPOCHS      = 100000  # Set to 50000 on Cluster
CHECKPOINT_STEP = 100

# --- Transformer Config (Config 1) ---
SEQUENCE_LENGTH           = 22
LAYERS                    = 2       # Slower, deeper structural tracking
HEADS                     = 4       
DIM_HEADS                 = 16      
DIM_MODEL                 = 64      # Balanced (HEADS * DIM_HEADS = 4 * 16)
DIM_MLP                   = 128     
TOKEN_TYPES               = 4
DIM_OUTPUT                = 3
TYPE                      = "relu"
INIT_WEIGHTS              = True
NUM_DEVICES               = 1
LENS_SEED                 = 999
ATTENTION_DIRECTION       = "causal"
NORMALIZATION             = "LNPre"
POSITIONAL_EMBEDDING_TYPE = "standard"

# --- Optimizer Config ---
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 0.05
BETAS         = (0.9, 0.98)
PATIENCE      = 15
