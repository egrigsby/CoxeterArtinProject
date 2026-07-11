import os
from pathlib import Path

# Directory containing this file — used to resolve DATA_PATH and PTH_LOCATION
_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Boolean Modes
# ---------------------------------------------------------------------------

LOAD_MODEL = False
CURRICULUM_MODE = True       # NEW: train shorter-word datasets first, then longer ones

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_PATH = _DIR
DATA_CSV = "data.csv"        # still used when CURRICULUM_MODE = False

CURRICULUM_DIR = _DIR / "curriculum_data"
PTH_LOCATION = str(_DIR / "workspace/_scratch/model.pth")
CURRICULUM_PTH_LOCATION = str(_DIR / "workspace/_scratch/curriculum_model_len10_len14.pth")

# ---------------------------------------------------------------------------
# Data Config
# ---------------------------------------------------------------------------

DATA_SEED = 598
TRAINING_SPLIT = 0.4

# For curriculum generation/training. Each stage file contains ONLY examples
# with true length exactly equal to that stage, padded to SEQUENCE_LENGTH.
CURRICULUM_STAGES = [10, 14]
CURRICULUM_LENGTHS_TO_GENERATE = CURRICULUM_STAGES
EXAMPLES_PER_LENGTH = 10000

# ---------------------------------------------------------------------------
# Training Loop Config
# ---------------------------------------------------------------------------

NUM_EPOCHS = 20000           # used when CURRICULUM_MODE = False
# Default epochs if a stage is not listed in EPOCHS_BY_STAGE.
EPOCHS_PER_STAGE = 20000     # used when CURRICULUM_MODE = True
# Partial-curriculum experiment: train length 10, then train length 14 longer.
EPOCHS_BY_STAGE = {
    10: 20000,
    14: 80000,
}
CHECKPOINT_STEP = 100

# ---------------------------------------------------------------------------
# Transformer Config
# ---------------------------------------------------------------------------

SEQUENCE_LENGTH = 22         # fixed maximum length; shorter words are padded to this
LAYERS = 1
HEADS = 4
DIM_HEADS = 64
DIM_MODEL = 256
DIM_MLP = 256
TOKEN_TYPES = 4              # 0 padding + generators 1,2,3
DIM_OUTPUT = 3               # one right-descent bit per generator
TYPE = "relu"
INIT_WEIGHTS = True
NUM_DEVICES = 1
LENS_SEED = 999
ATTENTION_DIRECTION = "causal"
NORMALIZATION = None
POSITIONAL_EMBEDDING_TYPE = "standard"

# ---------------------------------------------------------------------------
# Optimizer Config
# ---------------------------------------------------------------------------

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 2
BETAS = (0.9, 0.98)
PATIENCE = 15
