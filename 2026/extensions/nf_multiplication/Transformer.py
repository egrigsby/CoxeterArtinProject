from config import *

# ---------------------------------------------------------------------------
# General Libraries
# ---------------------------------------------------------------------------

import torch
import os
import shutil
import subprocess
import tqdm.auto as tqdm
import copy
from pathlib import Path

# CSV Use Libraries
import pandas as pd
import ast

# Function Imports
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformer_lens import HookedTransformer, HookedTransformerConfig

# ---------------------------------------------------------------------------
# Loss and Accuracy
# ---------------------------------------------------------------------------
# CLASSIFICATION VARIANT of the frozen descent model: the only differences from
# the shared Transformer.py are in this section, load_class_dataset, and the
# metric names ("seq" accuracy instead of "bit" accuracy). The task is a single
# softmax over DIM_OUTPUT classes per position instead of independent sigmoids.

_ce = torch.nn.CrossEntropyLoss(reduction='none')

def class_loss_fn(logits, targets, mask):
    """
    Masked softmax cross-entropy. At each prefix position the model classifies
    among DIM_OUTPUT classes; `targets` holds one integer class id per position
    (-1 at padding, which `mask` excludes).

    logits : [batch, seq_len, n_classes]
    targets: [batch, seq_len]  (long)
    mask   : [batch, seq_len]  (1 for real letters, 0 for padding)

    With causal attention the logits at position i already depend only on the
    prefix s_1..s_i, so labels align position-for-position (no shift).
    """
    per_pos = _ce(logits.transpose(1, 2), targets.clamp(min=0))  # [batch, seq_len]
    m = mask.float()
    return (per_pos * m).sum() / m.sum()

def class_accuracy_fn(logits, targets, mask):
    """Per-position argmax accuracy, averaged over non-pad positions."""
    correct = (logits.argmax(dim=-1) == targets).float()        # [batch, seq_len]
    m = mask.float()
    return (correct * m).sum() / m.sum()

def class_sequence_accuracy_fn(logits, targets, mask):
    """
    Whole-sequence accuracy: the fraction of sequences whose every non-pad
    position is classified correctly. Coarser than class_accuracy_fn — the
    per-position accuracy keeps moving while the model is only partially
    correct, this counts only fully-solved words.
    """
    correct = (logits.argmax(dim=-1) == targets).float()
    m = mask.float()
    per_seq = ((correct * m).sum(dim=-1) == m.sum(dim=-1)).float()  # [batch]
    return per_seq.mean()

# ---------------------------------------------------------------------------
# Attention Masking
# ---------------------------------------------------------------------------

# Create attention mask: 1 for real tokens, 0 for padding
def create_attention_mask(data_tensor):
    # assuming padding is exactly 0
    return (data_tensor != 0).int()

def pad_mask_hook(attn_scores, hook, mask):
    # attn_scores: [batch, head, q_pos, k_pos]
    # mask: [batch, seq_len]
    # Mask padding tokens from being attended to
    # Set attention scores to -inf where key is padding
    pad_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
    attn_scores = attn_scores.masked_fill(~pad_mask.bool(), float('-inf'))
    return attn_scores

def register_pad_mask_hook(model, attention_mask):
    def mask_hook(attn_scores, hook):
        return pad_mask_hook(attn_scores, hook, attention_mask)

    for layer in range(model.cfg.n_layers):
        model.blocks[layer].attn.hook_attn_scores.add_hook(mask_hook)

# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_class_dataset(csv_path):
    """
    Loads the two-column classification CSV (`word`, `labels`) and returns three
    aligned tensors:

      tokens  : [N, seq_len]  padded token IDs (0 = padding)
      targets : [N, seq_len]  integer class id per position (-1 where unsupervised)
      mask    : [N, seq_len]  1 for supervised positions (the product
                              continuation), 0 on the unsupervised w prefix
                              and on padding

    Column `labels` stores one class id per position (task-specific meaning —
    see ../build_multiplication_datasets.py), with -1 at unsupervised positions.
    """
    df = pd.read_csv(csv_path)
    words = [[int(x) for x in ast.literal_eval(w)] for w in df["word"]]
    labels = [[int(x) for x in ast.literal_eval(l)] for l in df["labels"]]

    tokens = torch.tensor(words, dtype=torch.long)        # [N, seq_len]
    targets = torch.tensor(labels, dtype=torch.long)      # [N, seq_len], -1 unsupervised
    mask = (targets != -1).long()                         # [N, seq_len]

    print(f"Loaded dataset: tokens {tuple(tokens.shape)} | targets {tuple(targets.shape)}")
    return tokens, targets, mask


# ---------------------------------------------------------------------------
# Setup Factories (shared by Transformer.py training and Analysis.ipynb)
# ---------------------------------------------------------------------------

def setup_device():
    """Configure CUDA env/memory and return (device, device1).

    device  : "cuda" or "cpu" (string, for HookedTransformerConfig)
    device1 : torch.device("cuda:0") or None (for .to(...) calls)
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9, device=0)
        torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device1 = torch.device("cuda:0") if torch.cuda.is_available() else None

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    print(f"Device Name: {device}")

    return device, device1


def build_cfg(device):
    """Build the HookedTransformerConfig from config.py constants."""
    return HookedTransformerConfig(
        n_ctx=SEQUENCE_LENGTH,
        n_layers=LAYERS,
        n_heads=HEADS,
        d_head=DIM_HEADS,
        d_model=DIM_MODEL,
        d_mlp=DIM_MLP,
        d_vocab=TOKEN_TYPES,
        d_vocab_out=DIM_OUTPUT,
        act_fn=TYPE,
        init_weights=INIT_WEIGHTS,
        device=device,
        n_devices=NUM_DEVICES,
        seed=LENS_SEED,
        attention_dir=ATTENTION_DIRECTION,
        normalization_type=NORMALIZATION,
        positional_embedding_type=POSITIONAL_EMBEDDING_TYPE,
    )


def build_model(cfg):
    """Construct model, optimizer, scheduler and disable biases. Returns the three."""
    model = HookedTransformer(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=BETAS
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=PATIENCE, factor=0.5)

    # Disable biases for interpretability
    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    return model, optimizer, scheduler


def load_train_test_data(device1):
    """Load the pre-split train/test CSVs (TRAIN_CSV / TEST_CSV) and move to device1.

    The train/test partition is fixed by the two files — no shuffling or splitting
    happens here, so the training script and the notebook always see the exact
    same sets.

    Returns a dict of train/test tokens, targets, masks, and attention masks.
    """
    train_tokens, train_targets, train_mask = load_class_dataset(DATA_PATH / TRAIN_CSV)
    test_tokens,  test_targets,  test_mask  = load_class_dataset(DATA_PATH / TEST_CSV)

    print(f"Train size: {train_tokens.shape[0]} | Test size: {test_tokens.shape[0]} | Seq Len: {train_tokens.shape[1]}")

    train_tokens,  test_tokens  = train_tokens.to(device1),  test_tokens.to(device1)
    train_targets, test_targets = train_targets.to(device1), test_targets.to(device1)
    train_mask,    test_mask    = train_mask.to(device1),    test_mask.to(device1)

    return {
        "train_tokens": train_tokens,   "test_tokens": test_tokens,
        "train_targets": train_targets, "test_targets": test_targets,
        "train_mask": train_mask,       "test_mask": test_mask,
        # The attention-hook padding mask covers ALL real tokens (the w prefix,
        # the MUL marker, and the product), unlike the loss mask above, which
        # covers only the supervised product-continuation positions.
        "train_attention_mask": create_attention_mask(train_tokens),
        "test_attention_mask":  create_attention_mask(test_tokens),
    }


def load_checkpoint_into(model, optimizer, scheduler, path=PTH_LOCATION):
    """Load weights/optimizer/scheduler state from a checkpoint in-place.

    Returns (cached, history) where `cached` is the raw checkpoint dict and
    `history` holds the training-curve lists and weight snapshots.
    """
    cached = torch.load(path, weights_only=False)
    model.load_state_dict(cached["model"])
    optimizer.load_state_dict(cached["optimizer"])
    scheduler.load_state_dict(cached["scheduler"])
    history = {
        k: cached.get(k, [])
        for k in (
            "checkpoints", "checkpoint_epochs",
            "train_losses", "test_losses",
            "train_accuracies", "test_accuracies",
            "train_seq_accuracies", "test_seq_accuracies",
        )
    }
    return cached, history


# ---------------------------------------------------------------------------
# Training entry point — only runs when executed as a script, not on import.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(Path(PTH_LOCATION).parent, exist_ok=True)

    # -----------------------------------------------------------------------
    # Device + Seed
    # -----------------------------------------------------------------------

    device, device1 = setup_device()
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi"])

    torch.manual_seed(seed=DATA_SEED)
    torch.cuda.manual_seed_all(DATA_SEED)

    num_epochs = NUM_EPOCHS
    checkpoint_every = CHECKPOINT_STEP

    # -----------------------------------------------------------------------
    # Model + Optimizer (fresh, or resumed from checkpoint when LOAD_MODEL)
    # -----------------------------------------------------------------------

    cfg = build_cfg(device)
    if LOAD_MODEL:
        cfg = torch.load(PTH_LOCATION, weights_only=False)["config"]

    model, optimizer, scheduler = build_model(cfg)

    if LOAD_MODEL:
        _, history = load_checkpoint_into(model, optimizer, scheduler)
        model_checkpoints = history["checkpoints"]
        checkpoint_epochs = history["checkpoint_epochs"]
        train_losses = history["train_losses"]
        test_losses = history["test_losses"]
        train_accuracies = history["train_accuracies"]
        test_accuracies = history["test_accuracies"]
        train_seq_accuracies = history["train_seq_accuracies"]
        test_seq_accuracies = history["test_seq_accuracies"]
    else:
        model_checkpoints = []
        checkpoint_epochs = []
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        train_seq_accuracies = []
        test_seq_accuracies = []

    # -----------------------------------------------------------------------
    # Dataset (load + shuffle + split + move to device)
    # -----------------------------------------------------------------------

    data = load_train_test_data(device1)
    train_tokens,  test_tokens  = data["train_tokens"],  data["test_tokens"]
    train_targets, test_targets = data["train_targets"], data["test_targets"]
    train_mask,    test_mask    = data["train_mask"],    data["test_mask"]
    train_attention_mask = data["train_attention_mask"]
    test_attention_mask  = data["test_attention_mask"]

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    for epoch in tqdm.tqdm(range(num_epochs)):
        # ---- Register hook for train ----
        model.reset_hooks()
        register_pad_mask_hook(model, train_attention_mask)

        # ---- Forward pass ----
        train_logits = model(train_tokens)      # [batch, seq_len, n_classes]
        train_loss = class_loss_fn(train_logits, train_targets, train_mask)
        train_loss.backward()

        # --- Gradient Clipping ---
        #clip_grad_norm_(model.parameters(), max_norm=3.0)

        # ---- Accuracy (train) ----
        train_accuracy = class_accuracy_fn(train_logits, train_targets, train_mask)
        train_seq_accuracy = class_sequence_accuracy_fn(train_logits, train_targets, train_mask)

        # ---- Optimizer + Scheduler step ----
        optimizer.step()
        #scheduler.step()           # do with mini batches of data
        optimizer.zero_grad()

        # ---- Evaluation ----
        with torch.inference_mode():
            # Add Attention Masking Hook:
            model.reset_hooks()
            register_pad_mask_hook(model, test_attention_mask)

            # ---- Forward pass (test) ----
            test_logits = model(test_tokens)
            test_loss = class_loss_fn(test_logits, test_targets, test_mask)

            # ---- Accuracy (test) ----
            test_accuracy = class_accuracy_fn(test_logits, test_targets, test_mask)
            test_seq_accuracy = class_sequence_accuracy_fn(test_logits, test_targets, test_mask)

        # ---- Checkpoint ----
        if ((epoch) % checkpoint_every) == 0:
            checkpoint_epochs.append(epoch)
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())
            test_accuracies.append(test_accuracy)
            train_seq_accuracies.append(train_seq_accuracy)
            test_seq_accuracies.append(test_seq_accuracy)
            # Save model’s weights (not model)
            model_checkpoints.append(copy.deepcopy(model.state_dict()))
            print(
                f"Epoch {epoch} | "
                f"Train Loss: {train_loss.item():.4f} | "
                f"Test Loss: {test_loss.item():.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Test Acc: {test_accuracy:.4f} | "
                f"Train Seq Acc: {train_seq_accuracy:.4f} | "
                f"Test Seq Acc: {test_seq_accuracy:.4f}"
            )

    # -----------------------------------------------------------------------
    # Save / Load Model
    # -----------------------------------------------------------------------

    torch.save(
        {
            "config": model.cfg,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "checkpoints": model_checkpoints,
            "checkpoint_epochs": checkpoint_epochs,
            "test_losses": test_losses,
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies,
            "train_seq_accuracies": train_seq_accuracies,
            "test_seq_accuracies": test_seq_accuracies,
        },
        PTH_LOCATION,
    )
