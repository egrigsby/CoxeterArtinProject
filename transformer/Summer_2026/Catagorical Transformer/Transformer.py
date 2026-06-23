from config import *

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

# Function Imports
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformer_lens import HookedTransformer, HookedTransformerConfig

os.makedirs(Path(PTH_LOCATION).parent, exist_ok=True)

# ---------------------------------------------------------------------------
# GPU Setup
# ---------------------------------------------------------------------------

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.cuda.set_per_process_memory_fraction(0.9, device=0)
torch.cuda.empty_cache()

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device Name: {device}")

device1 = torch.device("cuda:0") if torch.cuda.is_available() else None

subprocess.run(["nvidia-smi"])

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Set Randomization
torch.manual_seed(seed=DATA_SEED)
torch.cuda.manual_seed_all(DATA_SEED)

# --- Training Loop Config ---
num_epochs = NUM_EPOCHS
checkpoint_every = CHECKPOINT_STEP

frac_train = TRAINING_SPLIT
data_csv = DATA_CSV

# --- Transformer Config ---
cfg = HookedTransformerConfig(
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
)

cached_data = None
if LOAD_MODEL:
    cached_data = torch.load(PTH_LOCATION, weights_only=False)
    cfg = cached_data["config"]

# --- Optimizer Config --- 
# NOTE: a small learning rate (lr) is recommended for more complex models to avoid massive jumps, a schdeuler is likely necessary for >1 transformer layer
lr = LEARNING_RATE
wd = WEIGHT_DECAY
betas = BETAS
patience = PATIENCE

# --- Schedueler Config --- 
# warmup_steps = (int) (num_epochs * 0.1)

# ---------------------------------------------------------------------------
# Model and Optimizer
# ---------------------------------------------------------------------------

model = HookedTransformer(cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5)

# Disable biases for interpretability
for name, param in model.named_parameters():
    if "b_" in name:
        param.requires_grad = False

if LOAD_MODEL:
    model.load_state_dict(cached_data["model"])
    optimizer.load_state_dict(cached_data["optimizer"])
    scheduler.load_state_dict(cached_data["scheduler"])
    model_checkpoints = cached_data["checkpoints"]
    checkpoint_epochs = cached_data["checkpoint_epochs"]
    test_losses = cached_data["test_losses"]
    train_losses = cached_data["train_losses"]
    train_accuracies = cached_data["train_accuracies"]
    test_accuracies = cached_data["test_accuracies"]
else:
    model_checkpoints = []
    checkpoint_epochs = []
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

# ---------------------------------------------------------------------------
# Loss and Accuracy
# ---------------------------------------------------------------------------

#clsDex = 0  # index of CLS token, the token to be decoded (first token)

def autoregressive_loss_fn(logits, tokens, padding_token=0):
    """
    Grades the model at every single step (predicting descents and next characters)
    while ignoring empty padding at the end of the sequence.
    """
    # 1. Standard autoregressive shift alignment
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()
    
    # 2. Compute cross-entropy, ignoring padding positions
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=padding_token)
    
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    
    return loss_fn(flat_logits, flat_labels)

def autoregressive_accuracy_fn(logits, tokens, padding_token=0):
    """
    Calculates the exact matching percentage of the model's predictions,
    ignoring padding tokens.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous()
    
    # Get the token ID with the highest probability prediction
    predictions = torch.argmax(shift_logits, dim=-1)
    
    # Track correct matches where the target isn't padding
    correct = (predictions == shift_labels).float()
    valid_mask = (shift_labels != padding_token).float()
    
    return (correct * valid_mask).sum() / valid_mask.sum()

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

    for layer in range(cfg.n_layers):
        model.blocks[layer].attn.hook_attn_scores.add_hook(mask_hook)

# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_interleaved_dataset(csv_path):
    """
    Loads a CSV file where words and descents are already interleaved,
    and converts them into a 2D PyTorch tensor.
    """
    # Read the CSV file (assumes no header row, just rows of numbers)
    df = pd.read_csv(csv_path, header=None)

    # Convert the dataframe values directly into a long integer tensor
    data_tensor = torch.tensor(df.values, dtype=torch.long)

    print(f"Loaded dataset shape: {data_tensor.shape}") # Expecting [Batch_Size, Sequence_Length]
    return data_tensor

# Why is this here
def write_dataset_to_csv(file_path, data_tensor):
    """
    Saves a 2D PyTorch tensor of tokens back into a raw grid CSV file
    that load_interleaved_dataset can seamlessly read later.
    """
    # Move the tensor to CPU and convert it back to a numpy array/dataframe
    if isinstance(data_tensor, torch.Tensor):
        df = pd.DataFrame(data_tensor.detach().cpu().numpy())
    else:
        df = pd.DataFrame(data_tensor)

    # Save without a header row and without the row index numbers
    df.to_csv(file_path, index=False, header=False)

# Load the single dataset and shuffle
all_data = load_interleaved_dataset(DATA_PATH / data_csv)

perm = torch.randperm(all_data.size(0))
all_data = all_data[perm]

# Split into train and test using frac_train
n_train = int(frac_train * all_data.size(0))
train_data = all_data[:n_train]
test_data  = all_data[n_train:]

print(f"Train size: {train_data.shape[0]} | Test size: {test_data.shape[0]} | Seq Len: {train_data.shape[1]}")

# ---------------------------------------------------------------------------
# Move to Device and Create Masks
# ---------------------------------------------------------------------------

train_data = train_data.to(device1)
test_data  = test_data.to(device1)

train_attention_mask = create_attention_mask(train_data).to(device1)
test_attention_mask  = create_attention_mask(test_data).to(device1)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

for epoch in tqdm.tqdm(range(num_epochs)):
    # ---- Register hook for train ----
    model.reset_hooks()
    register_pad_mask_hook(model, train_attention_mask)

    # ---- Forward pass ----
    train_logits = model(train_data)
    train_loss = autoregressive_accuracy_fn(train_logits, train_data)
    train_loss.backward()
    train_losses.append(train_loss.item())

    # --- Gradient Clipping ---
    #clip_grad_norm_(model.parameters(), max_norm=3.0)

    # ---- Accuracy (train) ----
    train_accuracy = autoregressive_accuracy_fn(train_logits, train_data)
    train_accuracies.append(train_accuracy)

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
        test_logits = model(test_data)
        test_loss = autoregressive_loss_fn(test_logits, test_data)
        test_losses.append(test_loss.item())

        # ---- Accuracy (test) ----
        test_accuracy = autoregressive_accuracy_fn(test_logits,test_data)
        test_accuracies.append(test_accuracy)

    # ---- Checkpoint ----
    if ((epoch + 1) % checkpoint_every) == 0:
        checkpoint_epochs.append(epoch)
        # Save model’s weights (not model)
        model_checkpoints.append(copy.deepcopy(model.state_dict()))
        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss.item():.4f} | "
            f"Test Loss: {test_loss.item():.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Test Acc: {test_accuracy:.4f}"
        )


# ---------------------------------------------------------------------------
# Save / Load Model
# ---------------------------------------------------------------------------

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
    },
    PTH_LOCATION,
)
