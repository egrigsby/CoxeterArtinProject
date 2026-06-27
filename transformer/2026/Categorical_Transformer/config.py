import os
from pathlib import Path

# Directory containing this file — used to resolve DATA_PATH and PTH_LOCATION
_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Boolean Modes
# ---------------------------------------------------------------------------

LOAD_MODEL  = False

# if LOAD_MODEL = True, loads from PTH_LOCATION to continue training,
# and the loop always restarts from epoch = 0 and runs the full NUM_EPOCHS again.

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_PATH    = _DIR
PTH_LOCATION = str(_DIR / "workspace/_scratch/model.pth")

# ---------------------------------------------------------------------------
# Data Config
# ---------------------------------------------------------------------------

# Random seed used for both dataset shuffling and the train/test split
DATA_SEED      = 598
# Fraction of the full dataset used for training (remainder becomes test set)
TRAINING_SPLIT = 0.4
# Single input CSV file — must be in the same directory as this config.
# Two columns: `word` (list-string of generator IDs, padded with 0) and
# `descents` (per-prefix right-descent bitmask, padded with -1). Built by
# build_descent_dataset.py.
DATA_CSV       = "data.csv"

# ---------------------------------------------------------------------------
# Training Loop Config
# ---------------------------------------------------------------------------

NUM_EPOCHS      = 50000
CHECKPOINT_STEP = 100   # save a weight snapshot every this many epochs

# ---------------------------------------------------------------------------
# Transformer Config
# ---------------------------------------------------------------------------

# These map directly to HookedTransformerConfig parameters.
SEQUENCE_LENGTH     = 22        # fixed word length; must match the number of columns per row in DATA_CSV
LAYERS              = 3         # Increased from 1 for more capacity/depth
HEADS               = 8         # Increased from 4 for more diverse attention patterns
DIM_HEADS           = 64
DIM_MODEL           = 512       # Adjusted to equal HEADS * DIM_HEADS (8 * 64)
DIM_MLP             = 2048      # Standardized to 4 * DIM_MODEL for standard transformer scaling
TOKEN_TYPES         = 4         # input vocab size: #generators + 1 padding token (A2~: 3 gens + pad = 4)
DIM_OUTPUT          = 3         # multi-label head: one independent sigmoid unit per generator (A2~: 3)
TYPE                = "relu"    # activation; relu breaks linearization (good for interpretability)
INIT_WEIGHTS        = True
NUM_DEVICES         = 1
LENS_SEED           = 999       # TransformerLens weight-init seed (separate from DATA_SEED)
ATTENTION_DIRECTION = "causal"  # causal: prediction at position i sees only the prefix s_1..s_i
NORMALIZATION       = "LNPre"   # Changed from None to Pre-Layer Norm to stabilize the deeper 3-layer network
POSITIONAL_EMBEDDING_TYPE = "standard" # Options: "standard", "rotary", "shortformer", "alibi".

# ---------------------------------------------------------------------------
# Optimizer Config
# ---------------------------------------------------------------------------
# Scheduler: ReduceLROnPlateau — halves lr when test loss stalls for PATIENCE epochs.
# NOTE: a small lr is recommended for full-batch training; accurate gradients don't need large steps.
LEARNING_RATE = 1e-4        # Bumped up slightly from 1e-5 to match the larger 3-layer model capacity
WEIGHT_DECAY  = 0.01        # Reduced from 2 to a standard 0.01 to prevent crushing the weights/gradients
BETAS         = (0.9, 0.98) # Kept standard AdamW betas (excellent for Transformers)
PATIENCE      = 15          # Lowered slightly from 20 to react faster to plateaus, given the deeper model
