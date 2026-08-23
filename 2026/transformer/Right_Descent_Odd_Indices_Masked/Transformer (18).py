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

def descent_loss_fn(logits, targets, data_tensor, start_idx=1):
    """
    logits:      [batch, seq_len_logits, n_generators]
    targets:     [batch, seq_len_targets, n_generators]
    data_tensor: [batch, seq_len_logits] (token IDs, 0 = padding)
    """
    # Slice sequence dimension to match target length (e.g., 3 vs 36)
    if logits.shape[1] != targets.shape[1]:
        logits = logits[:, :targets.shape[1]]
        data_tensor = data_tensor[:, :targets.shape[1]]

    per_unit_loss = _bce(logits, targets)
    real_tokens_mask = (data_tensor != 0)
    
    batch_size, seq_len = data_tensor.shape
    pos_indices = torch.arange(seq_len, device=data_tensor.device).unsqueeze(0)
    valid_positions = pos_indices >= start_idx
    
    loss_mask = real_tokens_mask & valid_positions
    m = loss_mask.unsqueeze(-1).float()
    
    total_loss = (per_unit_loss * m).sum()
    total_tokens = m.sum() * logits.size(-1)
    
    return total_loss / torch.clamp(total_tokens, min=1.0)
    
def descent_accuracy_fn(logits, targets, data_tensor, start_idx=1):
    preds = (logits > 0).float()
    
    # Slice sequence dimension to match target length
    if preds.shape[1] != targets.shape[1]:
        preds = preds[:, :targets.shape[1]]
        data_tensor = data_tensor[:, :targets.shape[1]]

    if targets.ndim == 3:
        correct_bits = (preds == targets).float().sum(dim=-1)
        exact = (correct_bits == logits.size(-1)).float()
    else:
        exact = (preds == targets).float()

    real_tokens_mask = (data_tensor != 0)
    batch_size, seq_len = data_tensor.shape
    pos_indices = torch.arange(seq_len, device=data_tensor.device).unsqueeze(0)
    valid_positions = pos_indices >= start_idx
    
    m = (real_tokens_mask & valid_positions).float()
    return (exact * m).sum() / torch.clamp(m.sum(), min=1.0)

def descent_bit_accuracy_fn(logits, targets, data_tensor):
    preds = (logits > 0).float()
    
    # Slice sequence dimension to match target length
    if preds.shape[1] != targets.shape[1]:
        preds = preds[:, :targets.shape[1]]
        data_tensor = data_tensor[:, :targets.shape[1]]

    correct = (preds == targets).float()        
    token_mask = (data_tensor != 0).unsqueeze(-1).float()              
    return (correct * token_mask).sum() / torch.clamp(token_mask.sum() * logits.size(-1), min=1.0)

def descent_sequence_accuracy(logits, targets, data_tensor):
    preds = (logits > 0).float()
    
    # Slice sequence dimension to match target length
    if preds.shape[1] != targets.shape[1]:
        preds = preds[:, :targets.shape[1]]
        data_tensor = data_tensor[:, :targets.shape[1]]

    correct_bits = (preds == targets).float().sum(dim=-1)
    exact = (correct_bits == logits.size(-1)).float()
    m = (data_tensor != 0).float()
    return (exact * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1)

# ---------------------------------------------------------------------------
# Attention Masking (Mask Out ALL Odd Key Indices: 1, 3, 5, ...)
# ---------------------------------------------------------------------------

def create_odd_indices_attention_mask(data_tensor):
    """
    Creates a (batch_size, seq_len, seq_len) causal attention mask:
    - Standard causal mask: Query i can attend to Key j <= i.
    - Key rule constraint: Key j must be an EVEN index (j % 2 == 0).
      Odd key positions (1, 3, 5, ...) are strictly masked out.
    """
    batch_size, seq_len = data_tensor.shape
    
    # 1. Standard Causal Mask [seq_len, seq_len]
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=data_tensor.device))
    
    # 2. Key position constraint (Even key positions only)
    key_indices = torch.arange(seq_len, device=data_tensor.device).unsqueeze(0)  # [1, seq_len]
    even_keys_mask = (key_indices % 2 == 0)                                      # True for 0, 2, 4, 6...
    
    # 3. Combine Causal AND Even-Key Rule
    combined_mask = causal_mask & even_keys_mask
    
    # 4. Expand to batch shape: [batch_size, seq_len, seq_len]
    return combined_mask.unsqueeze(0).repeat(batch_size, 1, 1)

def pad_mask_hook(attn_scores, hook, mask_2d):
    """
    Applies the attention mask matrix to raw attention scores BEFORE softmax.
    attn_scores shape: [batch, num_heads, q_pos, k_pos]
    mask_2d shape:     [batch, q_pos, k_pos]
    """
    mask_expanded = mask_2d.unsqueeze(1)
    return attn_scores.masked_fill(~mask_expanded, float('-inf'))

def register_pad_mask_hook(model, attention_mask_2d):
    def mask_hook(attn_scores, hook):
        return pad_mask_hook(attn_scores, hook, attention_mask_2d)

    model.reset_hooks()
    for layer in range(model.cfg.n_layers):
        model.blocks[layer].attn.hook_attn_scores.add_hook(mask_hook)

# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_descent_dataset(csv_path, n_generators):
    df = pd.read_csv(csv_path)
    words = [[int(x) for x in ast.literal_eval(w)] for w in df["word"]]
    descs = [[int(x) for x in ast.literal_eval(d)] for d in df["descents"]]

    tokens = torch.tensor(words, dtype=torch.long)
    mask = (tokens != 0).long()

    bitmasks = torch.tensor(descs, dtype=torch.long)
    bits = torch.arange(n_generators)
    targets = ((bitmasks.clamp(min=0).unsqueeze(-1) >> bits) & 1).float()
    targets = targets * mask.unsqueeze(-1).float()

    print(f"Loaded dataset: tokens {tuple(tokens.shape)} | targets {tuple(targets.shape)}")
    return tokens, targets, mask

def load_train_test_data(device1):
    train_tokens, train_targets, _ = load_descent_dataset(DATA_PATH / TRAIN_CSV, DIM_OUTPUT)
    test_tokens,  test_targets,  _ = load_descent_dataset(DATA_PATH / TEST_CSV,  DIM_OUTPUT)

    print(f"Train size: {train_tokens.shape[0]} | Test size: {test_tokens.shape[0]} | Seq Len: {train_tokens.shape[1]}")

    if device1 is not None:
        train_tokens,  test_tokens  = train_tokens.to(device1),  test_tokens.to(device1)
        train_targets, test_targets = train_targets.to(device1), test_targets.to(device1)

    train_att_mask = create_odd_indices_attention_mask(train_tokens)
    test_att_mask  = create_odd_indices_attention_mask(test_tokens)

    return {
        "train_tokens": train_tokens,   "test_tokens": test_tokens,
        "train_targets": train_targets, "test_targets": test_targets,
        "train_attention_mask": train_att_mask, 
        "test_attention_mask": test_att_mask,
    }

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

def load_checkpoint_into(model, optimizer, scheduler, path=PTH_LOCATION):
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
            "train_bit_accuracies", "test_bit_accuracies",
        )
    }
    return cached, history

# ---------------------------------------------------------------------------
# Training Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(Path(PTH_LOCATION).parent, exist_ok=True)

    device, device1 = setup_device()
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi"])

    torch.manual_seed(seed=DATA_SEED)
    torch.cuda.manual_seed_all(DATA_SEED)

    num_epochs = NUM_EPOCHS
    checkpoint_every = CHECKPOINT_STEP

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
        model_checkpoints, checkpoint_epochs = [], []
        train_losses, test_losses = [], []
        train_accuracies, test_accuracies = [], []
        train_bit_accuracies, test_bit_accuracies = [], []

    data = load_train_test_data(device1)
    train_tokens,  test_tokens  = data["train_tokens"],  data["test_tokens"]
    train_targets, test_targets = data["train_targets"], data["test_targets"]
    train_attention_mask = data["train_attention_mask"]
    test_attention_mask  = data["test_attention_mask"]

    for epoch in tqdm.tqdm(range(num_epochs)):
        # ---- Forward pass (train) ----
        model.reset_hooks()
        register_pad_mask_hook(model, train_attention_mask)

        train_logits = model(train_tokens)
        train_loss = descent_loss_fn(train_logits, train_targets, train_tokens)
        train_loss.backward()

        train_accuracy = descent_accuracy_fn(train_logits, train_targets, train_tokens)
        train_bit_accuracy = descent_bit_accuracy_fn(train_logits, train_targets, train_tokens)

        optimizer.step()
        optimizer.zero_grad()

        # ---- Forward pass (test) ----
        with torch.inference_mode():
            model.reset_hooks()
            register_pad_mask_hook(model, test_attention_mask)

            test_logits = model(test_tokens)
            test_loss = descent_loss_fn(test_logits, test_targets, test_tokens)

            test_accuracy = descent_accuracy_fn(test_logits, test_targets, test_tokens)
            test_bit_accuracy = descent_bit_accuracy_fn(test_logits, test_targets, test_tokens)

        # ---- Logging / Checkpointing ----
        if (epoch % checkpoint_every) == 0:
            checkpoint_epochs.append(epoch)
            train_accuracies.append(train_accuracy.item() if isinstance(train_accuracy, torch.Tensor) else train_accuracy)
            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())
            test_accuracies.append(test_accuracy.item() if isinstance(test_accuracy, torch.Tensor) else test_accuracy)
            train_bit_accuracies.append(train_bit_accuracy.item() if isinstance(train_bit_accuracy, torch.Tensor) else train_bit_accuracy)
            test_bit_accuracies.append(test_bit_accuracy.item() if isinstance(test_bit_accuracy, torch.Tensor) else test_bit_accuracy)
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