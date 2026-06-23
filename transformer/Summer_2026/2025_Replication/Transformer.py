# ---------------------------------------------------------------------------
# Boolean Modes
# ---------------------------------------------------------------------------

TRAIN_MODEL = True   # if False, skip training
LOAD_MODEL  = True  

# if LOAD_MODEL = True, loads from PTH_LOCATION to continue training, 
# and the loop always restarts from epoch = 0 and runs the full num_epochs again.

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

SEQUENCE_LENGTH=22,                     # fixed sequence length (size of an input)
LAYERS=1,
HEADS=4,
DIM_HEADS=64,
DIM_MODEL=256,                          # nHeads * dHeads
DIM_MLP=256,                            # d_model * 4 (recommended)
TOKEN_TYPES=5,                          # token types = numOfGens + 1Padding + 1 mask?
DIM_OUTPUT=2,                           # Output types
TYPE="relu",                            # breaks linearization
INIT_WEIGHTS=True,
#DEVICE=device,
NUM_DEVICES=1,
#seed=LENS_SEED,
ATTENTION_DIRECTION="bidirectional",    # bidirectional/causal
NORMALIZATION=None,                     # None, LN, LNPre, RMS, RMSPre

NUM_EPOCHS = 25000
CHECKPOINT_STEP = 100

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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

directory = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = Path(directory)

PTH_LOCATION = os.path.join(directory, "workspace/_scratch/model.pth")
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

DATA_SEED = 598
LENS_SEED = 999
torch.manual_seed(seed=DATA_SEED)
torch.cuda.manual_seed_all(DATA_SEED)

num_epochs = NUM_EPOCHS
checkpoint_every = CHECKPOINT_STEP
frac_train = 0.4

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

lr = 1e-5
wd = 7
betas = (0.9, 0.98)
patience = 20

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

clsDex = 0  # index of CLS token, the token to be decoded (first token)

# returns scalar representing performance in the whole batch
def loss_fn(logits, labels):
    # get (batches, logits) for all last tokens in all batches
    if len(logits.shape)==3:
        logits = logits[:, clsDex]
    logits = logits.to(torch.float64)
    labels = labels.to(logits.device)   # Move labels to the same device as logits
    labels = labels.unsqueeze(1)        # (batch size, 1) turned into appropriate shape
    # logits shape: (batch, # of classes)       # focus on last dimension of logits, which is the list of classes (0 or 1), get index based probability
    log_probs = logits.log_softmax(dim=-1)      # turn logits/last dimension (per sequence) into a prob distrib (then logs it)
    correct_log_probs = log_probs.gather(dim=-1, index=labels).squeeze(1)   # squeeze removes single dimension, so now list of size batches
    return -correct_log_probs.mean()

# NOTE item() used for final output of accuracy and not scalar because, loss needs to run backward while accuracy can just be stored

#returns scalar representing combined percent scores (performance) of model
def accuracy_fn(logits, labels):
    if len(logits.shape) == 3:
        logits = logits[:, clsDex]
    labels = labels.to(logits.device) # Move labels to the same device as logits
    # get index of largest num in 2d logit, for all sequences, then compare with label
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()

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

def load_dataset_from_csv(file_path):
    df = pd.read_csv(file_path, names=['tokens', 'label'], skiprows=1)
    df['tokens'] = df['tokens'].apply(lambda x: [int(i.strip(" '")) for i in x.strip("[]").split(",")])
    words = [torch.tensor(seq) for seq in df['tokens']]
    labels = torch.tensor(df['label'].values)
    return torch.stack(words), labels

def write_dataset_to_csv(file_path, data, labels):
    data_list = data.tolist()
    label_list = labels.tolist()
    formatted_tokens = [str([str(token) for token in seq]) for seq in data_list]
    df = pd.DataFrame({'tokens': formatted_tokens, 'label': label_list})
    df.to_csv(file_path, index=False, header=True)

reservedFiles = {
    "test": "test.csv",
    "train": "train.csv",
    "relators": "relators.csv"
}
reservedFilesL = list(reservedFiles.values())

test_data, test_labels = load_dataset_from_csv(DATA_PATH / reservedFiles["test"])

USE_TRAIN_CSV = True

if USE_TRAIN_CSV:
    train_data, train_labels = load_dataset_from_csv(DATA_PATH / reservedFiles["train"])
else:
    all_datasets = {}
    relators_data, relators_labels = load_dataset_from_csv(DATA_PATH / reservedFiles["relators"])

    fileNames = sorted(
        [f for f in os.listdir(DATA_PATH) if f.endswith(".csv") and f not in reservedFilesL],
        key=lambda name: int(name.split('-')[0])
    )

    all_datasets["relators"] = (relators_data, relators_labels)
    for fileName in fileNames:
        all_datasets[fileName.replace(".csv", "")] = load_dataset_from_csv(DATA_PATH / fileName)

    print("Loaded datasets:")
    for name, (data, labels) in all_datasets.items():
        print(f"  {name}: {data.shape} words, {labels.shape} labels")

    train_data = torch.cat([d for d, _ in all_datasets.values()], dim=0)
    train_labels = torch.cat([l for _, l in all_datasets.values()], dim=0)
    write_dataset_to_csv(DATA_PATH / reservedFiles["train"], train_data, train_labels)

tPerm = torch.randperm(train_data.size(0))

train_data = train_data[tPerm]
train_labels = train_labels[tPerm].to(device1)

# Nanda uses one dataset. frac train for training and the rest for testing
# n_train = int(frac_train * train_data.size(0)) 
# train_data = train_data[:n_train]
# train_labels = train_labels[:n_train]

print(f"Batch Size: {train_data.shape[0]} | Seq Len: {train_data.shape[1]}")

# ---------------------------------------------------------------------------
# Move to Device and Create Masks
# ---------------------------------------------------------------------------

test_data = test_data.to(device1)
test_labels = test_labels.to(device1)

test_attention_mask = create_attention_mask(test_data).to(device1)
train_attention_mask = create_attention_mask(train_data).to(device1)

ALTERNATE_TRAINING = False
if not USE_TRAIN_CSV and ALTERNATE_TRAINING:
    for name in all_datasets:
        data, labels = all_datasets[name]
        all_datasets[name] = (data.to(device1), labels.to(device1))

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

#num_epochs = 100   # num_epochs    (debug, keep commented out)

if TRAIN_MODEL:
    for epoch in tqdm.tqdm(range(num_epochs)):
        # ---- Register hook for train ----
        model.reset_hooks()
        register_pad_mask_hook(model, train_attention_mask)

        # ---- Forward pass ----
        train_logits = model(train_data)
        train_loss = loss_fn(train_logits, train_labels)
        train_loss.backward()
        train_losses.append(train_loss.item())

        # --- Gradient Clipping ---
        #clip_grad_norm_(model.parameters(), max_norm=3.0)

        # ---- Accuracy (train) ----
        train_accuracy = accuracy_fn(train_logits, train_labels)
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
            test_loss = loss_fn(test_logits, test_labels)
            test_losses.append(test_loss.item())

            # ---- Accuracy (test) ----
            test_accuracy = accuracy_fn(test_logits,test_labels)
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

if TRAIN_MODEL:
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
