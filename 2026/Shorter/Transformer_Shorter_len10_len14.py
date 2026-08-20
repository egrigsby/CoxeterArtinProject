from config_shorter_len10_len14 import *

# ---------------------------------------------------------------------------
# General Libraries
# ---------------------------------------------------------------------------

import torch
import os
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
    per_unit = _bce(logits, targets)
    m = mask.unsqueeze(-1).float()
    return (per_unit * m).sum() / (m.sum() * logits.size(-1))

def descent_accuracy_fn(logits, targets, mask):
    preds = (logits > 0).float()
    correct_bits = (preds == targets).float().sum(dim=-1)
    exact = (correct_bits == logits.size(-1)).float()
    m = mask.float()
    return (exact * m).sum() / m.sum()

def descent_sequence_accuracy(logits, targets, mask):
    preds = (logits > 0).float()
    correct_bits = (preds == targets).float().sum(dim=-1)
    exact = (correct_bits == logits.size(-1)).float()
    m = mask.float()
    return (exact * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1)

# ---------------------------------------------------------------------------
# Attention Masking
# ---------------------------------------------------------------------------

def create_attention_mask(data_tensor):
    return (data_tensor != 0).int()

def pad_mask_hook(attn_scores, hook, mask):
    pad_mask = mask.unsqueeze(1).unsqueeze(2)
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

def load_descent_dataset(csv_path, n_generators):
    """
    Loads CSV with columns:
      word     : list-string of token IDs, padded by 0
      descents : list-string of right-descent bitmasks, padded by -1
    Extra columns like true_length are ignored.
    """
    df = pd.read_csv(csv_path)
    words = [[int(x) for x in ast.literal_eval(w)] for w in df["word"]]
    descs = [[int(x) for x in ast.literal_eval(d)] for d in df["descents"]]

    tokens = torch.tensor(words, dtype=torch.long)
    mask = (tokens != 0).long()

    bitmasks = torch.tensor(descs, dtype=torch.long)
    bits = torch.arange(n_generators)
    targets = ((bitmasks.clamp(min=0).unsqueeze(-1) >> bits) & 1).float()
    targets = targets * mask.unsqueeze(-1).float()

    print(f"Loaded {csv_path}: tokens {tuple(tokens.shape)} | targets {tuple(targets.shape)}")
    return tokens, targets, mask

# ---------------------------------------------------------------------------
# Setup Factories
# ---------------------------------------------------------------------------

def setup_device():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9, device=0)
        torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device1 = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    print(f"Device Name: {device}")

    return device, device1

def build_cfg(device):
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
    model = HookedTransformer(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=BETAS
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=PATIENCE, factor=0.5)

    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    return model, optimizer, scheduler

def move_data_to_device(tokens, targets, mask, device1):
    tokens = tokens.to(device1)
    targets = targets.to(device1)
    mask = mask.to(device1)
    return tokens, targets, mask, mask

def load_and_split_data(device1):
    """Original single-file mode, kept for non-curriculum runs."""
    all_tokens, all_targets, all_mask = load_descent_dataset(DATA_PATH / DATA_CSV, DIM_OUTPUT)

    torch.manual_seed(DATA_SEED)
    perm = torch.randperm(all_tokens.size(0))
    all_tokens = all_tokens[perm]
    all_targets = all_targets[perm]
    all_mask = all_mask[perm]

    n_train = int(TRAINING_SPLIT * all_tokens.size(0))
    train_tokens, test_tokens = all_tokens[:n_train], all_tokens[n_train:]
    train_targets, test_targets = all_targets[:n_train], all_targets[n_train:]
    train_mask, test_mask = all_mask[:n_train], all_mask[n_train:]

    train_tokens, train_targets, train_mask, train_attention_mask = move_data_to_device(train_tokens, train_targets, train_mask, device1)
    test_tokens, test_targets, test_mask, test_attention_mask = move_data_to_device(test_tokens, test_targets, test_mask, device1)

    return {
        "train_tokens": train_tokens, "test_tokens": test_tokens,
        "train_targets": train_targets, "test_targets": test_targets,
        "train_mask": train_mask, "test_mask": test_mask,
        "train_attention_mask": train_attention_mask, "test_attention_mask": test_attention_mask,
    }

def split_tensors(tokens, targets, mask, seed):
    """Shuffle and split one loaded dataset into train/test tensors.

    This mirrors the original Transformer.py design: generation writes one CSV,
    and the transformer file controls the reproducible train/test split using
    TRAINING_SPLIT.
    """
    torch.manual_seed(seed)
    perm = torch.randperm(tokens.size(0))
    tokens = tokens[perm]
    targets = targets[perm]
    mask = mask[perm]

    n_train = int(TRAINING_SPLIT * tokens.size(0))
    train_tokens, test_tokens = tokens[:n_train], tokens[n_train:]
    train_targets, test_targets = targets[:n_train], targets[n_train:]
    train_mask, test_mask = mask[:n_train], mask[n_train:]

    print(
        f"Split with seed {seed}: "
        f"Train {train_tokens.shape[0]:,} | Test {test_tokens.shape[0]:,} | "
        f"Train fraction {TRAINING_SPLIT}"
    )

    return train_tokens, test_tokens, train_targets, test_targets, train_mask, test_mask

def load_curriculum_stage_data(stage, device1):
    """Load one exact-length stage CSV, then split train/test inside this file."""
    stage_path = CURRICULUM_DIR / f"exact_len_{stage}.csv"

    all_tokens, all_targets, all_mask = load_descent_dataset(stage_path, DIM_OUTPUT)
    (
        train_tokens, test_tokens,
        train_targets, test_targets,
        train_mask, test_mask,
    ) = split_tensors(all_tokens, all_targets, all_mask, seed=DATA_SEED + stage)

    train_tokens, train_targets, train_mask, train_attention_mask = move_data_to_device(
        train_tokens, train_targets, train_mask, device1
    )
    test_tokens, test_targets, test_mask, test_attention_mask = move_data_to_device(
        test_tokens, test_targets, test_mask, device1
    )

    return {
        "train_tokens": train_tokens, "test_tokens": test_tokens,
        "train_targets": train_targets, "test_targets": test_targets,
        "train_mask": train_mask, "test_mask": test_mask,
        "train_attention_mask": train_attention_mask, "test_attention_mask": test_attention_mask,
    }

def load_checkpoint_into(model, optimizer, scheduler, path=PTH_LOCATION):
    cached = torch.load(path, weights_only=False)
    model.load_state_dict(cached["model"])
    optimizer.load_state_dict(cached["optimizer"])
    scheduler.load_state_dict(cached["scheduler"])
    history = {
        k: cached[k]
        for k in (
            "checkpoints", "checkpoint_epochs",
            "train_losses", "test_losses",
            "train_accuracies", "test_accuracies",
        )
    }
    return cached, history


# ---------------------------------------------------------------------------
# Save Helper
# ---------------------------------------------------------------------------

def save_training_state(model, optimizer, scheduler, history, path):
    torch.save(
        {
            "config": model.cfg,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "checkpoints": history["checkpoints"],
            "checkpoint_epochs": history["checkpoint_epochs"],
            "stages": history.get("stages", []),
            "test_losses": history["test_losses"],
            "train_losses": history["train_losses"],
            "train_accuracies": history["train_accuracies"],
            "test_accuracies": history["test_accuracies"],
        },
        path,
    )
    print(f"Saved model to {path}")

# ---------------------------------------------------------------------------
# Training entry point — curriculum version, written close to original style
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    save_path = CURRICULUM_PTH_LOCATION if CURRICULUM_MODE else PTH_LOCATION
    os.makedirs(Path(save_path).parent, exist_ok=True)

    # -----------------------------------------------------------------------
    # Device + Seed
    # -----------------------------------------------------------------------

    device, device1 = setup_device()
    subprocess.run(["nvidia-smi"])

    torch.manual_seed(seed=DATA_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(DATA_SEED)

    # -----------------------------------------------------------------------
    # Model + Optimizer
    # -----------------------------------------------------------------------

    cfg = build_cfg(device)
    if LOAD_MODEL:
        cfg = torch.load(save_path, weights_only=False)["config"]

    model, optimizer, scheduler = build_model(cfg)

    if LOAD_MODEL:
        _, history = load_checkpoint_into(model, optimizer, scheduler, path=save_path)
        history.setdefault("stages", [])
    else:
        history = {
            "checkpoints": [],
            "checkpoint_epochs": [],
            "stages": [],
            "train_losses": [],
            "train_accuracies": [],
            "test_losses": [],
            "test_accuracies": [],
        }

    # -----------------------------------------------------------------------
    # Choose datasets: exact-length curriculum stages or original single DATA_CSV
    # -----------------------------------------------------------------------

    if CURRICULUM_MODE:
        stages_to_run = CURRICULUM_STAGES
    else:
        stages_to_run = [None]

    # -----------------------------------------------------------------------
    # Exact-Length Curriculum Training
    # -----------------------------------------------------------------------

    for stage in stages_to_run:
        if stage is None:
            stage_name = "single"
            print("\n" + "=" * 80)
            print(f"Single-dataset training using {DATA_CSV}")
            print("=" * 80)

            epochs_per_dataset = NUM_EPOCHS

            # Same as original Transformer.py:
            # load one CSV, shuffle, split, move to device.
            data = load_and_split_data(device1)
        else:
            stage_name = f"len={stage}"
            epochs_per_dataset = EPOCHS_BY_STAGE.get(stage, EPOCHS_PER_STAGE)
            print("\n" + "=" * 80)
            print(f"Curriculum stage: true word length exactly {stage}")
            print(f"Epochs for this stage: {epochs_per_dataset}")
            print("=" * 80)

            # Curriculum replacement for original load_and_split_data:
            # load exact_len_{stage}.csv, then shuffle/split inside transformer.
            data = load_curriculum_stage_data(stage, device1)

        # -------------------------------------------------------------------
        # Dataset (already loaded + shuffled + split + moved to device)
        # -------------------------------------------------------------------

        train_tokens,  test_tokens  = data["train_tokens"],  data["test_tokens"]
        train_targets, test_targets = data["train_targets"], data["test_targets"]
        train_mask,    test_mask    = data["train_mask"],    data["test_mask"]
        train_attention_mask = data["train_attention_mask"]
        test_attention_mask  = data["test_attention_mask"]

        # -------------------------------------------------------------------
        # Training — same logic as original Transformer.py
        # -------------------------------------------------------------------

        for epoch in tqdm.tqdm(range(epochs_per_dataset), desc=f"Stage {stage_name}"):
            # ---- Register hook for train ----
            model.reset_hooks()
            register_pad_mask_hook(model, train_attention_mask)

            # ---- Forward pass ----
            train_logits = model(train_tokens)      # [batch, seq_len, n_generators]
            train_loss = descent_loss_fn(train_logits, train_targets, train_mask)
            train_loss.backward()

            # --- Gradient Clipping ---
            # clip_grad_norm_(model.parameters(), max_norm=3.0)

            # ---- Accuracy (train) ----
            train_accuracy = descent_accuracy_fn(train_logits, train_targets, train_mask)

            # ---- Optimizer step ----
            optimizer.step()
            optimizer.zero_grad()

            # ---- Evaluation ----
            with torch.inference_mode():
                model.reset_hooks()
                register_pad_mask_hook(model, test_attention_mask)

                test_logits = model(test_tokens)
                test_loss = descent_loss_fn(test_logits, test_targets, test_mask)
                test_accuracy = descent_accuracy_fn(test_logits, test_targets, test_mask)

            # ---- Scheduler step ----
            # ReduceLROnPlateau expects a validation metric.
            scheduler.step(test_loss.item())

            # ---- Checkpoint ----
            if epoch % CHECKPOINT_STEP == 0:
                global_epoch = len(history["train_losses"])
                history["checkpoint_epochs"].append(global_epoch)
                history["stages"].append(stage_name)
                history["train_accuracies"].append(float(train_accuracy))
                history["train_losses"].append(train_loss.item())
                history["test_losses"].append(test_loss.item())
                history["test_accuracies"].append(float(test_accuracy))
                history["checkpoints"].append(copy.deepcopy(model.state_dict()))

                print(
                    f"Stage {stage_name} | Epoch {epoch} | "
                    f"Train Loss: {train_loss.item():.4f} | "
                    f"Test Loss: {test_loss.item():.4f} | "
                    f"Train Acc: {float(train_accuracy):.4f} | "
                    f"Test Acc: {float(test_accuracy):.4f}"
                )

        # -------------------------------------------------------------------
        # Save after every dataset/stage
        # -------------------------------------------------------------------

        if stage is None:
            stage_path = Path(save_path)
        else:
            stage_path = Path(save_path).with_name(f"curriculum_model_exact_len_{stage}.pth")

        save_training_state(model, optimizer, scheduler, history, stage_path)

    # -----------------------------------------------------------------------
    # Final Save
    # -----------------------------------------------------------------------

    save_training_state(model, optimizer, scheduler, history, save_path)
