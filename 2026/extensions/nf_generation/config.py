"""
Normal-form generation - next-token LM over ShortLex normal-form words of A2~.
Uses shared/Transformer_classification.py. Dataset: build_nf_lm_dataset.py.
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
# `labels` (per-position NEXT-token id: the following letter, 0 = STOP at the
# last letter, -1 at padding). Words are the exhaustive ShortLex normal-form
# words of length 1..36 (A2~, 1998 words), with words and 30/70 split inherited
# unchanged from "../Affine A2". Built by build_nf_lm_dataset.py in this folder
# (the job script runs it before training). After training, generate.py rolls
# the model out from seed prefixes and scores legality.
TRAIN_CSV      = "train.csv"
TEST_CSV       = "test.csv"

# ---------------------------------------------------------------------------
# Training Loop Config
# ---------------------------------------------------------------------------

NUM_EPOCHS      = 10000
CHECKPOINT_STEP = 100   # save a weight snapshot every this many epochs

# ---------------------------------------------------------------------------
# Transformer Config
# ---------------------------------------------------------------------------

# These map directly to HookedTransformerConfig parameters.
SEQUENCE_LENGTH     = 36        # max word length; must match the number of columns per row in TRAIN_CSV/TEST_CSV
LAYERS              = 1
HEADS               = 4
DIM_HEADS           = 64
DIM_MODEL           = 256       # should equal HEADS * DIM_HEADS
DIM_MLP             = 1024       # hidden dim of MLP block; typically 4 * DIM_MODEL
TOKEN_TYPES         = 4         # input vocab size: #generators + 1 padding token (A2~: 3 gens + pad)
DIM_OUTPUT          = 4         # next-token head: softmax over {0 = STOP, 1..3 = generators}
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
