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

_bce = torch.nn.BCEWithLogitsLoss(reduction='none')

def descent_loss_fn(logits, targets, mask):
    """
    Masked multi-label BCE. At each prefix position the model predicts, for every
    generator independently, whether it is a right descent of that prefix.

    logits, targets: [batch, seq_len, n_generators]
    mask:            [batch, seq_len]  (1 for real letters, 0 for padding)

    With causal attention the logits at position i already depend only on the
    prefix s_1..s_i, so labels align position-for-position (no shift).
    """
    per_unit = _bce(logits, targets)            # [batch, seq_len, n_generators]
    m = mask.unsqueeze(-1).float()              # [batch, seq_len, 1]
    return (per_unit * m).sum() / (m.sum() * logits.size(-1))

def descent_accuracy_fn(logits, targets, mask):
    """
    Per-prefix exact-set-match accuracy: a position counts as correct only if all
    n_generators descent bits are predicted correctly. Averaged over non-pad positions.
    """
    preds = (logits > 0).float()                # sigmoid(logit) > 0.5  <=>  logit > 0
    correct_bits = (preds == targets).float().sum(dim=-1)   # [batch, seq_len]
    exact = (correct_bits == logits.size(-1)).float()       # all generators right
    m = mask.float()
    return (exact * m).sum() / m.sum()

def descent_bit_accuracy_fn(logits, targets, mask):
    """
    Per-bit (per-generator) accuracy: the fraction of individual descent bits
    predicted correctly, averaged over all non-pad positions AND all generators.

    Unlike descent_accuracy_fn (which credits a position only when all
    n_generators bits are right), this gives partial credit, so it keeps moving
    while the model is only partially correct — a far better training signal for
    diagnosing whether learning is happening at all. Normalization matches
    descent_loss_fn (divide by non-pad positions * n_generators).
    """
    preds = (logits > 0).float()                # sigmoid(logit) > 0.5  <=>  logit > 0
    correct = (preds == targets).float()        # [batch, seq_len, n_generators]
    m = mask.unsqueeze(-1).float()              # [batch, seq_len, 1]
    return (correct * m).sum() / (m.sum() * logits.size(-1))

def descent_sequence_accuracy(logits, targets, mask):
    """Per-sequence fraction of prefixes with an exactly-correct descent set.
    Returns shape (batch_size,)."""
    preds = (logits > 0).float()
    correct_bits = (preds == targets).float().sum(dim=-1)
    exact = (correct_bits == logits.size(-1)).float()
    m = mask.float()
    return (exact * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1)

# ---------------------------------------------------------------------------
# Attention Masking (Truncate after 5th letter + Padding)
# ---------------------------------------------------------------------------

# Replace 'mask_first_5_tokens_mask' with this:
def mask_even_indices_mask(data_tensor):
    """
    0 = Mask out (even indices 0, 2, 4, ... or padding)
    1 = Keep (odd indices 1, 3, 5, ... provided they are not padding)
    """
    batch_size, seq_len = data_tensor.shape
    
    # 1. Mask non-zero padding tokens
    real_tokens_mask = (data_tensor != 0)
    
    # 2. Keep only odd positions (1, 3, 5, ...) -> (pos % 2 != 0)
    pos_indices = torch.arange(seq_len, device=data_tensor.device).unsqueeze(0)
    odd_indices_mask = (pos_indices % 2 != 0)
    
    # Combine: Must be non-padding AND at an odd index
    combined_mask = real_tokens_mask & odd_indices_mask
    return combined_mask.long()
    
def pad_mask_hook(attn_scores, hook, mask):
    # attn_scores: [batch, head, q_pos, k_pos]
    # mask: [batch, seq_len] (1 = keep, 0 = mask out)
    
    # 1. Mask key positions (set invalid keys to -inf)
    key_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, k_pos]
    attn_scores = attn_scores.masked_fill(~key_mask.bool(), float('-inf'))
    
    # 2. To avoid NaN in softmax for disabled query positions (0..4),
    # allow disabled query tokens to attend ONLY to themselves.
    # Create diagonal mask [1, 1, seq_len, seq_len]
    seq_len = attn_scores.size(-1)
    eye = torch.eye(seq_len, device=attn_scores.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
    
    # Query positions that are disabled (mask == 0)
    q_disabled = (~mask.bool()).unsqueeze(1).unsqueeze(-1)  # [batch, 1, q_pos, 1]
    
    # Force diagonal to 0.0 (valid attention score) ONLY where query is disabled AND key == query position
    force_self_attn = q_disabled & eye
    attn_scores = torch.where(force_self_attn, torch.tensor(0.0, device=attn_scores.device), attn_scores)
    
    return attn_scores
    
def register_pad_mask_hook(model, attention_mask):
    def mask_hook(attn_scores, hook):
        return pad_mask_hook(attn_scores, hook, attention_mask)

    for layer in range(model.cfg.n_layers):
        model.blocks[layer].attn.hook_attn_scores.add_hook(mask_hook)
# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_descent_dataset(csv_path, n_generators):
    """
    Loads the two-column descent CSV (`word`, `descents`) produced by
    build_descent_dataset.py and returns three aligned tensors:

      tokens  : [N, seq_len]              padded generator IDs (0 = padding)
      targets : [N, seq_len, n_generators] multi-hot right-descent set per prefix
      mask    : [N, seq_len]              1 for real letters, 0 for padding

    Column `descents` stores one bitmask int per position (bit j <=> generator j+1),
    with -1 at padding positions; we decode it into the multi-hot target.
    """
    df = pd.read_csv(csv_path)
    words = [[int(x) for x in ast.literal_eval(w)] for w in df["word"]]
    descs = [[int(x) for x in ast.literal_eval(d)] for d in df["descents"]]

    tokens = torch.tensor(words, dtype=torch.long)        # [N, seq_len]
    mask = (tokens != 0).long()                           # [N, seq_len]

    bitmasks = torch.tensor(descs, dtype=torch.long)      # [N, seq_len], -1 on padding
    # Decode bitmask -> multi-hot: bit j of each entry gives generator j+1.
    bits = torch.arange(n_generators)
    targets = ((bitmasks.clamp(min=0).unsqueeze(-1) >> bits) & 1).float()  # [N, seq_len, n]
    targets = targets * mask.unsqueeze(-1).float()        # zero out padding rows

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
    train_tokens, train_targets, train_mask = load_descent_dataset(DATA_PATH / TRAIN_CSV, DIM_OUTPUT)
    test_tokens,  test_targets,  test_mask  = load_descent_dataset(DATA_PATH / TEST_CSV,  DIM_OUTPUT)

    train_tokens,  test_tokens  = train_tokens.to(device1),  test_tokens.to(device1)
    train_targets, test_targets = train_targets.to(device1), test_targets.to(device1)
    train_mask,    test_mask    = train_mask.to(device1),    test_mask.to(device1)

    # Generate the masked attention/loss masks
    train_att_mask = mask_first_5_tokens_mask(train_tokens, num_masked_prefix=5)
    test_att_mask  = mask_first_5_tokens_mask(test_tokens,  num_masked_prefix=5)

    return {
        "train_tokens": train_tokens,   "test_tokens": test_tokens,
        "train_targets": train_targets, "test_targets": test_targets,
        "train_mask": train_att_mask,   "test_mask": test_att_mask, # Pass att_mask as the loss mask too!
        "train_attention_mask": train_att_mask, 
        "test_attention_mask": test_att_mask,
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
            # .get: checkpoints saved before bit accuracy was logged lack these keys
            "train_bit_accuracies", "test_bit_accuracies",
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
        train_bit_accuracies = history["train_bit_accuracies"]
        test_bit_accuracies = history["test_bit_accuracies"]
    else:
        model_checkpoints = []
        checkpoint_epochs = []
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        train_bit_accuracies = []
        test_bit_accuracies = []

    # -----------------------------------------------------------------------
    # Dataset (load + shuffle + split + move to device)
    # -----------------------------------------------------------------------

def load_train_test_data(device1):
    train_tokens, train_targets, _ = load_descent_dataset(DATA_PATH / TRAIN_CSV, DIM_OUTPUT)
    test_tokens,  test_targets,  _ = load_descent_dataset(DATA_PATH / TEST_CSV,  DIM_OUTPUT)

    train_tokens,  test_tokens  = train_tokens.to(device1),  test_tokens.to(device1)
    train_targets, test_targets = train_targets.to(device1), test_targets.to(device1)

    # Generate the even-indices attention/loss masks
    train_att_mask = mask_even_indices_mask(train_tokens)
    test_att_mask  = mask_even_indices_mask(test_tokens)

    return {
        "train_tokens": train_tokens,   "test_tokens": test_tokens,
        "train_targets": train_targets, "test_targets": test_targets,
        "train_mask": train_att_mask,   "test_mask": test_att_mask, # Pass att_mask as the loss mask
        "train_attention_mask": train_att_mask, 
        "test_attention_mask": test_att_mask,
    }
    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    for epoch in tqdm.tqdm(range(num_epochs)):
        # ---- Register hook for train ----
        model.reset_hooks()
        register_pad_mask_hook(model, train_attention_mask)

        # ---- Forward pass ----
        train_logits = model(train_tokens)      # [batch, seq_len, n_generators]
        train_loss = descent_loss_fn(train_logits, train_targets, train_mask)
        train_loss.backward()

        # --- Gradient Clipping ---
        #clip_grad_norm_(model.parameters(), max_norm=3.0)

        # ---- Accuracy (train) ----
        train_accuracy = descent_accuracy_fn(train_logits, train_targets, train_mask)
        train_bit_accuracy = descent_bit_accuracy_fn(train_logits, train_targets, train_mask)

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
            test_loss = descent_loss_fn(test_logits, test_targets, test_mask)

            # ---- Accuracy (test) ----
            test_accuracy = descent_accuracy_fn(test_logits, test_targets, test_mask)
            test_bit_accuracy = descent_bit_accuracy_fn(test_logits, test_targets, test_mask)

        # ---- Checkpoint ----
        if ((epoch) % checkpoint_every) == 0:
            checkpoint_epochs.append(epoch)
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())
            test_accuracies.append(test_accuracy)
            train_bit_accuracies.append(train_bit_accuracy)
            test_bit_accuracies.append(test_bit_accuracy)
            # Save model’s weights (not model)
            model_checkpoints.append(copy.deepcopy(model.state_dict()))
            print(
                f"Epoch {epoch} | "
                f"Train Loss: {train_loss.item():.4f} | "
                f"Test Loss: {test_loss.item():.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Test Acc: {test_accuracy:.4f} | "
                f"Train Bit Acc: {train_bit_accuracy:.4f} | "
                f"Test Bit Acc: {test_bit_accuracy:.4f}"
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
            "train_bit_accuracies": train_bit_accuracies,
            "test_bit_accuracies": test_bit_accuracies,
        },
        PTH_LOCATION,
    )
