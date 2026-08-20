"""
Normal-form multiplication - s x NF(w) -> NF(sw). Sequence output, cross-entropy.
Uses the local ../Transformer.py. Dataset: ../build_multiplication_datasets.py.
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
# Each has two columns: `word` (list-string of token IDs padded with 0:
# [w, MUL_s, product] where 1..3 are letters and 4..6 = MUL_s means "multiply
# by s") and `labels` (the product continuation: at the MUL position the first
# product letter, then each next letter, 0 = STOP at the last product letter;
# -1 on the w prefix and padding). LEFT multiplication: the product is the
# ShortLex normal form of s*w. Pairs are the exhaustive normal-form words of
# length 1..36 (A2~, 1998 words) x 3 generators, split 80/20 by base word
# (seed 0), identical words and partition as the sibling ../Right folder.
# Built once by ../build_multiplication_datasets.py (run it from the parent
# folder before submitting). After training, generate.py rolls the model out
# from every test pair's seed "w MUL_s" and scores exact match + legality.
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
SEQUENCE_LENGTH     = 74        # 36 (max w) + 1 (MUL token) + 37 (max product); must match the builder's FIXED_LENGTH
LAYERS              = 1
HEADS               = 4
DIM_HEADS           = 64
DIM_MODEL           = 256       # should equal HEADS * DIM_HEADS
DIM_MLP             = 1024       # hidden dim of MLP block; typically 4 * DIM_MODEL
TOKEN_TYPES         = 7         # input vocab: pad + 3 letters + 3 MUL_s markers (4..6)
DIM_OUTPUT          = 4         # product-continuation head: softmax over {0 = STOP, 1..3 = generators}
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
