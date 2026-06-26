# Categorical Transformer

Trains a [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) `HookedTransformer` on Coxeter group word data using a **per-prefix right-descent-set prediction** objective. With **causal** attention, at each prefix `s₁…sᵢ` of a word the model predicts the *right descent set* of that prefix — i.e. for every generator independently, whether it is a right descent. This is a **multi-label** task (a prefix can have several descents at once), trained with masked binary cross-entropy.

Make sure to run the Torch Setup notebook to download all of the libraries.

## Files

| File | Purpose |
|---|---|
| `config.py` | **Single source of truth** for all configuration. Edit this file to change any setting. |
| `Transformer.py` | Training script **and** shared-helper module. Run as a script to train; import from it to reuse its functions. The training code is guarded by `if __name__ == "__main__"`, so importing it does **not** trigger training. |
| `Analysis.ipynb` | Analysis notebook. Imports config and helpers from `Transformer.py`, loads the checkpoint, and runs interpretability analysis. |

The dataset is generated separately by `../Set_Generation/build_descent_dataset.py` (which uses `descents.py` to compute right descent sets from the Coxeter matrix).

### Importable helpers in `Transformer.py`

Because training is behind the `__main__` guard, every function defined at module level can be imported without side effects. `Analysis.ipynb` imports these rather than redefining them, so the two stay in sync:

| Helper | Purpose |
|---|---|
| `setup_device()` | Configure CUDA env/memory; returns `(device, device1)`. |
| `build_cfg(device)` | Build the `HookedTransformerConfig` from `config.py` constants. |
| `build_model(cfg)` | Construct model, optimizer (AdamW), scheduler (ReduceLROnPlateau), and disable biases. Returns `(model, optimizer, scheduler)`. |
| `load_and_split_data(device1)` | Load `data.csv`, shuffle + split with `DATA_SEED`, move to device. Returns a dict of train/test tokens, targets, masks, and attention masks. |
| `load_checkpoint_into(model, optimizer, scheduler)` | Load weights/optimizer/scheduler state from a checkpoint in place; returns `(cached, history)`. |
| `load_descent_dataset(csv_path, n_generators)` | Parse the two-column CSV into `(tokens, targets, mask)` tensors. |
| `descent_loss_fn` / `descent_accuracy_fn` / `descent_sequence_accuracy` | Masked multi-label BCE loss; per-prefix exact-set-match accuracy; per-sequence descent accuracy. |
| `create_attention_mask` / `pad_mask_hook` / `register_pad_mask_hook` | Build the padding mask and register the attention hook that prevents attending to padding. |

## Quickstart

1. Place your dataset as `data.csv` in this directory (same folder as `config.py`).
2. Edit settings in `config.py` as needed.
3. Run: `python Transformer.py`

The trained model is saved to `workspace/_scratch/model.pth` when training finishes.

---

## Data Format

`data.csv` is a **two-column CSV with header `word,descents`**, one row per training example, all for a **single** group (there is no group-ID token):

- **`word`** — a Python list-string of generator IDs padded with `0`, e.g. `"['2','1','3','0', ...]"`. Token `0` is the **padding token** and is masked out of both attention and loss. The list length is `SEQUENCE_LENGTH`.
- **`descents`** — the per-position right-descent set encoded as a **bitmask int** (bit `j` ⇔ generator `j+1`), padded with `-1`. An empty descent set is `0`, which is distinct from the `-1` padding.

`load_descent_dataset` decodes this into `tokens [N,L]`, a multi-hot `targets [N,L,n_generators]`, and a padding `mask [N,L]`. Because attention is causal, the logits at position `i` predict the descent set of the prefix ending at letter `i` — no label shifting is needed.

`load_and_split_data` shuffles the full dataset (re-seeding with `DATA_SEED` immediately before the shuffle) and splits into train/test by `TRAINING_SPLIT`. Both the training script and the notebook call this same function, so they always reproduce the identical split.

---

## Boolean Modes

`LOAD_MODEL` affects `Transformer.py` only. The notebook always loads from the checkpoint. When `True`, load a previous checkpoint from `PTH_LOCATION` first — restoring model weights, optimizer state, scheduler state, and all loss/accuracy history — then restart from epoch 0 and run the full `NUM_EPOCHS` again. When `False`, train from scratch. Either way the model is saved at the end. |

---

## Configuration Reference

All configuration lives in `config.py`. Edit that file before running — both `Transformer.py` and `Analysis.ipynb` import from it automatically.

### Data Config

| Variable | Description |
|---|---|
| `DATA_CSV` | `"data.csv"` Filename of the input dataset. Must be in the same directory as `config.py`. |
| `TRAINING_SPLIT` | Fraction of the dataset used for **training**. The remainder becomes the test set. (e.g. `0.4` → 40% train, 60% test.) |
| `DATA_SEED` | Random seed for dataset shuffling and the train/test split. Ensures reproducibility. |

### Training Loop Config

| Variable | Description |
|---|---|
| `NUM_EPOCHS` | Total number of full-dataset passes to train for (training is full-batch). |
| `CHECKPOINT_STEP` | Save a model weight snapshot every this many epochs. All snapshots are stored in memory and written to the `.pth` file at the end. |

### Transformer Config

These map directly to `HookedTransformerConfig` parameters.

| Variable | Description |
|---|---|
| `SEQUENCE_LENGTH` | Max length of every input sequence. Must match the word length in `data.csv`. |
| `LAYERS` | Number of transformer blocks. More layers increase capacity but also interpretability complexity. |
| `HEADS` | Number of attention heads per layer. |
| `DIM_HEADS` | Dimension of each attention head. `DIM_MODEL` should equal `HEADS × DIM_HEADS`. |
| `DIM_MODEL` | Residual stream dimension (embedding size). |
| `DIM_MLP` | Hidden dimension of the MLP block inside each transformer layer. |
| `TOKEN_TYPES` | Input vocabulary size — number of generators + 1 padding token (A₂̃: 3 generators + pad = 4). |
| `DIM_OUTPUT` | Size of the multi-label head — one independent sigmoid unit per generator (A₂̃: 3). |
| `TYPE` | MLP activation function. `"relu"` breaks linearization of the model's computation, which is desirable for mechanistic interpretability. |
| `INIT_WEIGHTS` | Whether TransformerLens initializes the weights. |
| `NUM_DEVICES` | Number of devices TransformerLens shards the model across. |
| `LENS_SEED` | Random seed for TransformerLens weight initialization (separate from `DATA_SEED`). |
| `ATTENTION_DIRECTION` | `"causal"` restricts the prediction at position `i` to the prefix `s₁…sᵢ`. (`"bidirectional"` would let every token attend to every other token.) |
| `NORMALIZATION` |Layer normalization type. `None` disables it to simplify the computation graph for interpretability. Options: `None`, `"LN"`, `"LNPre"`, `"RMS"`, `"RMSPre"`. |
| `POSITIONAL_EMBEDDING_TYPE` | Learned absolute positional embeddings (the TransformerLens default, made explicit). Options: `"standard"`, `"rotary"`, `"shortformer"`, `"alibi"`. |

### Optimizer Config

The optimizer is **AdamW**. The learning-rate scheduler is `ReduceLROnPlateau`, which halves the learning rate when test loss stops improving.

| Variable | Description |
|---|---|
| `LEARNING_RATE` | Initial learning rate. Kept small because full-batch training produces very accurate gradients that do not need large steps. |
| `WEIGHT_DECAY` | L2 regularization strength. Unusually large (typical values are 0.01–0.1); monitor for instability. |
| `BETAS` | AdamW momentum parameters `(β₁, β₂)`. |
| `PATIENCE` | Number of epochs with no test-loss improvement before the scheduler reduces the learning rate. |

---

## Analysis Notebook (`Analysis.ipynb`)

The notebook loads the saved checkpoint and runs interpretability analysis on the trained model. It imports the shared helpers from `Transformer.py` and calls `load_and_split_data` to reproduce the identical train/test split used during training.

**What it contains:**
- Loss and accuracy curves (train vs. test over all epochs)
- Average attention patterns per head, averaged across the full training set
- Per-word attention pattern visualizations
- **Misclassification analysis** — each sequence is given a *sequence accuracy* (`descent_sequence_accuracy`: the fraction of its prefixes whose right-descent set is predicted exactly correctly). A sequence is "imperfect" if that accuracy is `< 1.0`. Imperfect sequences are flagged and their length distribution is plotted for both train and test sets.

**Running the notebook:** The first cell adds the Categorical Transformer directory to `sys.path` so `config.py` and `Transformer.py` can be imported. Run the cells top-to-bottom.

> **Environment note:** the checkpoint's pickled `cfg` was saved with a newer `transformer_lens` (the notebook runs in a Python 3.12 env). Loading it requires that same environment — the `transformer_lens` in `/projects/expmmllab/CoxeterEnv` (Python 3.11) cannot unpickle it. The notebook includes a `sys.modules` shim cell for its own runtime.

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
