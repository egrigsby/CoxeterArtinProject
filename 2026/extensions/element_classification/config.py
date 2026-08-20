"""
Finite A2 (= S3) element classification - 6-way softmax over group elements at every
prefix, NOT the multi-label descent task. Uses shared/Transformer_classification.py.
"""

import os
from pathlib import Path

# Directory containing this file — used to resolve DATA_PATH and PTH_LOCATION
_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Boolean Modes
# ---------------------------------------------------------------------------

LOAD_MODEL  = True   # continue-training run: resumes from workspace/_scratch/model.pth (10k-epoch checkpoint backed up as model_10k.pth)

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

# Random seed for torch (weight init reproducibility aside, LENS_SEED covers init;
# this seeds the training-loop RNG). The train/test split is fixed by the files below.
DATA_SEED      = 598
# Pre-split input CSV files — must be in the same directory as this config.
# Each has two columns: `word` (list-string of generator IDs) and `labels`
# (per-prefix ELEMENT ID, 0..5). Words are 4000 distinct random words of fixed
# length 18 over the two generators of FINITE A2 (= S3), adjacent repeats
# allowed; label i = which of the 6 elements the prefix s_1..s_i equals.
# Built and split 30/70 by build_element_dataset.py in this folder (the job
# script runs the build before training).
TRAIN_CSV      = "train.csv"
TEST_CSV       = "test.csv"

# ---------------------------------------------------------------------------
# Training Loop Config
# ---------------------------------------------------------------------------

NUM_EPOCHS      = 15000   # additional epochs for the continue run (first run was 10000)
CHECKPOINT_STEP = 100   # save a weight snapshot every this many epochs

# ---------------------------------------------------------------------------
# Transformer Config
# ---------------------------------------------------------------------------

# These map directly to HookedTransformerConfig parameters.
SEQUENCE_LENGTH     = 18        # max word length (= FIXED_LENGTH in build_element_dataset.py); must match the number of columns per row in TRAIN_CSV/TEST_CSV
LAYERS              = 1
HEADS               = 4
DIM_HEADS           = 64
DIM_MODEL           = 256       # should equal HEADS * DIM_HEADS
DIM_MLP             = 1024       # hidden dim of MLP block; typically 4 * DIM_MODEL
TOKEN_TYPES         = 3         # input vocab size: #generators + 1 padding token (finite A2: 2 gens + pad)
DIM_OUTPUT          = 6         # classification head: softmax over the 6 elements of S3
TYPE                = "relu"    # activation; relu breaks linearization (good for interpretability)
INIT_WEIGHTS        = True
NUM_DEVICES         = 1
LENS_SEED           = 999       # TransformerLens weight-init seed (separate from DATA_SEED)
ATTENTION_DIRECTION = "causal"  # causal: prediction at position i sees only the prefix s_1..s_i
NORMALIZATION       = None      # None, "LN", "LNPre", "RMS", "RMSPre"
POSITIONAL_EMBEDDING_TYPE = "standard" # Options: "standard", "rotary", "shortformer", "alibi".

# ---------------------------------------------------------------------------
# Optimizer Config
# ---------------------------------------------------------------------------

# Scheduler: ReduceLROnPlateau — halves lr when test loss stalls for PATIENCE epochs.
# (Note: scheduler.step() is currently disabled in the training loop, so the LR is constant.)
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 0.5
BETAS         = (0.9, 0.98)
PATIENCE      = 20
