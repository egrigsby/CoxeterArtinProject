# Categorical Transformer

Trains a [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) `HookedTransformer` on Coxeter group word data using an autoregressive (next-token prediction) objective. The model reads interleaved sequences of word tokens and descent labels and learns to predict each next token given all prior tokens.

## Files

| File | Purpose |
|---|---|
| `config.py` | **Single source of truth** for all configuration. Edit this file to change any setting. |
| `Transformer.py` | Training script. Imports config, trains the model, saves a checkpoint. |
| `Transformer.ipynb` | Analysis notebook. Imports config, loads the checkpoint, and runs interpretability analysis. |

## Quickstart

1. Place your dataset as `data.csv` in this directory (same folder as `config.py`).
2. Edit settings in `config.py` as needed.
3. Run: `python Transformer.py`

The trained model is saved to `workspace/_scratch/model.pth` when training finishes.

---

## Data Format

`data.csv` must be a headerless CSV where:
- Each **row** is one training example (a sequence of integer token IDs).
- Each **column** is one position in the sequence.
- Row length must match `SEQUENCE_LENGTH`.
- Token ID `0` is reserved as the **padding token** and is masked out during attention and loss computation.

The script shuffles the full dataset (using `DATA_SEED`), then splits it into train and test sets according to `TRAINING_SPLIT`. The notebook reproduces this exact split so that analysis always runs on the correct held-out test set.

---

## Boolean Modes

The LOAD_MODEL config is for `Transformer.py`. The notebook always loads from the checkpoint regardless of these flags.

| Flag | Default | Effect |
|---|---|---|
| `LOAD_MODEL` | `False` | Load a previous checkpoint from `PTH_LOCATION` before doing anything else. The loaded checkpoint restores the model weights, optimizer state, scheduler state, and all loss/accuracy history. Training then restarts from epoch 0 and runs for the full `NUM_EPOCHS` again. |

| `True` | `False` | Train from scratch, save at end. |
| `True` | `True` | Resume from checkpoint, continue training, save at end. |

---

## Configuration Reference

All configuration lives in `config.py`. Edit that file before running — both `Transformer.py` and `Transformer.ipynb` import from it automatically.

### Data Config

| Variable | Default | Description |
|---|---|---|
| `DATA_CSV` | `"data.csv"` | Filename of the input dataset. Must be in the same directory as the script. |
| `TRAINING_SPLIT` | `0.4` | Fraction of the dataset used for **training**. The remainder becomes the test set. (e.g. `0.4` → 40% train, 60% test.) |
| `DATA_SEED` | `598` | Random seed for dataset shuffling. Ensures the train/test split is reproducible. |

### Training Loop Config

| Variable | Default | Description |
|---|---|---|
| `NUM_EPOCHS` | `25000` | Total number of full-dataset passes to train for. |
| `CHECKPOINT_STEP` | `100` | Save a model weight snapshot every this many epochs. All snapshots are stored in memory and written to the `.pth` file at the end. |

### Transformer Config

These map directly to `HookedTransformerConfig` parameters.

| Variable | Default | Description |
|---|---|---|
| `SEQUENCE_LENGTH` | `22` | Fixed length of every input sequence. Must match the number of columns in `data.csv`. |
| `LAYERS` | `1` | Number of transformer blocks. More layers increase capacity but also interpretability complexity. |
| `HEADS` | `4` | Number of attention heads per layer. |
| `DIM_HEADS` | `64` | Dimension of each attention head. `DIM_MODEL` should equal `HEADS × DIM_HEADS`. |
| `DIM_MODEL` | `256` | Residual stream dimension (embedding size). |
| `DIM_MLP` | `256` | Hidden dimension of the MLP block inside each transformer layer. Typically `4 × DIM_MODEL`; set lower here to reduce parameters. |
| `TOKEN_TYPES` | `5` | Vocabulary size — total number of distinct token IDs the model can see (generators + padding + any special tokens). |
| `DIM_OUTPUT` | `3` | Number of output classes the model predicts at each position. |
| `TYPE` | `"relu"` | MLP activation function. `"relu"` is used here because it breaks linearization of the model's computation, which is desirable for mechanistic interpretability. |
| `ATTENTION_DIRECTION` | `"bidirectional"` | `"bidirectional"` lets every token attend to every other token. `"causal"` restricts each token to only attend to previous positions. |
| `NORMALIZATION` | `None` | Layer normalization type. `None` disables it entirely. Options: `None`, `"LN"`, `"LNPre"`, `"RMS"`, `"RMSPre"`. Disabled here to simplify the computation graph for interpretability. |
| `LENS_SEED` | `999` | Random seed for TransformerLens weight initialization (separate from `DATA_SEED`). |

### Optimizer Config

The optimizer is **AdamW**. The learning rate scheduler is `ReduceLROnPlateau`, which halves the learning rate when test loss stops improving.

| Variable | Default | Description |
|---|---|---|
| `LEARNING_RATE` | `1e-5` | Initial learning rate. Kept small because full-batch training produces very accurate gradients that do not need large steps. |
| `WEIGHT_DECAY` | `7` | L2 regularization strength. This is unusually large (typical values are 0.01–0.1); monitor for instability. |
| `BETAS` | `(0.9, 0.98)` | AdamW momentum parameters `(β₁, β₂)`. |
| `PATIENCE` | `20` | Number of epochs with no test-loss improvement before the scheduler reduces the learning rate. |

---

## Analysis Notebook (`Transformer.ipynb`)

The notebook loads the saved checkpoint and runs interpretability analysis on the trained model. It uses the same `DATA_CSV`, `DATA_SEED`, and `TRAINING_SPLIT` from `config.py` to reproduce the identical train/test split.

**What it contains:**
- Loss and accuracy curves (train vs. test over all checkpoints)
- Average attention patterns per head, averaged across the full training set
- Per-word attention pattern visualizations
- **Misclassification analysis** — in the autoregressive setting, a sequence is considered "imperfect" if the model fails to predict any token in that sequence correctly. Each sequence is assigned a *sequence accuracy* (fraction of non-padding positions correctly predicted). Sequences with accuracy < 1.0 are flagged, and their length distribution is plotted.

**Running the notebook:** The notebook's working directory should be `Summer_2026/` (one level above this directory). The first cell adds `Catagorical Transformer/` to `sys.path` so that `config.py` can be found automatically.

---

## Output

After training, the script saves a single `.pth` file to `workspace/_scratch/model.pth` containing:

| Key | Contents |
|---|---|
| `config` | The `HookedTransformerConfig` used to build the model. |
| `model` | Final model weights (`state_dict`). |
| `optimizer` | Optimizer state at end of training. |
| `scheduler` | Scheduler state at end of training. |
| `checkpoints` | List of model `state_dict` snapshots taken every `CHECKPOINT_STEP` epochs. |
| `checkpoint_epochs` | List of epoch indices corresponding to each checkpoint. |
| `train_losses` | Training loss recorded every epoch. |
| `test_losses` | Test loss recorded every epoch. |
| `train_accuracies` | Training accuracy recorded every epoch. |
| `test_accuracies` | Test accuracy recorded every epoch. |
