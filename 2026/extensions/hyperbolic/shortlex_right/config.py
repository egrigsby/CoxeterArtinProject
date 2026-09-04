"""
Hyperbolic (4,3,3) - per-prefix RIGHT descent sets, ShortLex normal-form words.
"""

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

# Random seed for torch (weight init reproducibility aside, LENS_SEED covers init;
# this seeds the training-loop RNG). The train/test split is fixed by the files below.
DATA_SEED      = 598
# Pre-split input CSV files — must be in the same directory as this config.
# Each has two columns: `word` (list-string of generator IDs, padded with 0) and
# `descents` (per-prefix RIGHT-descent bitmask, padded with -1). Words and the
# 30/70 partition come from shortlex_right_descents.csv in this folder,
# split by resplit_train_test.py.
TRAIN_CSV      = "train.csv"
TEST_CSV       = "test.csv"

# ---------------------------------------------------------------------------
# Training Loop Config
# ---------------------------------------------------------------------------

NUM_EPOCHS      = 15000
CHECKPOINT_STEP = 100   # save a weight snapshot every this many epochs

# ---------------------------------------------------------------------------
# Transformer Config
# ---------------------------------------------------------------------------

# These map directly to HookedTransformerConfig parameters.
SEQUENCE_LENGTH     = 36        # max word length (= FIXED_LENGTH in build_left_descent_nf_dataset.py); must match the number of columns per row in TRAIN_CSV/TEST_CSV
LAYERS              = 1
HEADS               = 6
DIM_HEADS           = 64
DIM_MODEL           = 256       # should equal HEADS * DIM_HEADS
DIM_MLP             = 1024       # hidden dim of MLP block; typically 4 * DIM_MODEL
TOKEN_TYPES         = 4         # input vocab size: #generators + 1 padding token (A2~: 3 gens + pad = 4)
DIM_OUTPUT          = 3         # multi-label head: one independent sigmoid unit per generator (A2~: 3)
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
# These values reproduce the best-performing baseline (see REPORT.md, exp01):
# 1e-5 / WD=2 underfit badly; WD=0 overfits; 1e-3 / WD=0.5 is the validated operating point.
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 0.5
BETAS         = (0.9, 0.98)
PATIENCE      = 20
