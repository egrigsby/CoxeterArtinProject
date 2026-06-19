# General Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import os
import subprocess
import tqdm.auto as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from typing import List, Union, Optional
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses

# CSV Use Libraries
import pandas as pd
import ast

# Function Imports
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_cosine_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_

import transformer_lens
from transformer_lens import HookedRootModule
import transformer_lens.utilities as utils
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import transformer_lens.config.hooked_transformer_config as htc

import plotly.io as pio
import matplotlib.pyplot as plt
from collections import defaultdict
import gc

from neel_plotly.plot import line_or_scatter, line

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

oldDir = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = Path(oldDir)

PTH_LOCATION = os.path.join(oldDir, "workspace/_scratch/model.pth")
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
# Plotly Setup
# ---------------------------------------------------------------------------

pio.renderers.default = "browser"
pio.templates['plotly'].layout.xaxis.title.font.size = 20
pio.templates['plotly'].layout.yaxis.title.font.size = 20
pio.templates['plotly'].layout.title.font.size = 30

# ---------------------------------------------------------------------------
# Graphing Helpers
# ---------------------------------------------------------------------------

def imshow(tensor, renderer=None, xaxis="", yaxis="", xlabels=None, ylabels=None, aspect="auto", **kwargs):
    fig = px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": xaxis, "y": yaxis},
        aspect=aspect,
        **kwargs
    )
    if xlabels is not None:
        fig.update_xaxes(tickmode='array', tickvals=list(range(len(xlabels))), ticktext=xlabels)
    if ylabels is not None:
        fig.update_yaxes(tickmode='array', tickvals=list(range(len(ylabels))), ticktext=ylabels)
    fig.update_yaxes(scaleanchor=None)
    fig.show(renderer)
    return fig

def line_plot(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x": xaxis, "y": yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs).show(renderer)

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Boolean Modes
# ---------------------------------------------------------------------------

TRAIN_MODEL = True   # if False, load the pretrained model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_SEED = 598
LENS_SEED = 999
torch.manual_seed(seed=DATA_SEED)
torch.cuda.manual_seed_all(DATA_SEED)

frac_train = 0.4

num_epochs = 25000
checkpoint_every = 100

cfg = htc.HookedTransformerConfig(
    n_ctx=22,
    n_layers=1,
    n_heads=4,
    d_head=64,
    d_model=256,
    d_mlp=256,
    d_vocab=5,
    d_vocab_out=2,
    act_fn="relu",
    init_weights=True,
    device=device,
    n_devices=1,
    seed=LENS_SEED,
    attention_dir="bidirectional",
    normalization_type=None,
)

cached_data = None
if not TRAIN_MODEL:
    cached_data = torch.load(PTH_LOCATION, weights_only=False)
    cfg = cached_data["config"]

lr = 1e-5
wd = 7
betas = (0.9, 0.98)
patience = 20

warmup_steps = int(num_epochs * 0.1)

# ---------------------------------------------------------------------------
# Model and Optimizer
# ---------------------------------------------------------------------------

model = HookedTransformer(cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5)

if not TRAIN_MODEL:
    model.load_state_dict(cached_data["model"])
    optimizer.load_state_dict(cached_data["optimizer"])
    scheduler.load_state_dict(cached_data["scheduler"])

# Disable biases for interpretability
for name, param in model.named_parameters():
    if "b_" in name:
        param.requires_grad = False

# ---------------------------------------------------------------------------
# Loss and Accuracy (Need to change to Catagorical Loss)
# ---------------------------------------------------------------------------

clsDex = 0  # index of CLS token (first token)

def loss_fn(logits, labels):
    if len(logits.shape) == 3:
        logits = logits[:, clsDex]
    logits = logits.to(torch.float64)
    labels = labels.to(logits.device)
    labels = labels.unsqueeze(1)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels).squeeze(1)
    return -correct_log_probs.mean()

def accuracy_fn(logits, labels):
    if len(logits.shape) == 3:
        logits = logits[:, clsDex]
    labels = labels.to(logits.device)
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()

# ---------------------------------------------------------------------------
# Attention Masking
# ---------------------------------------------------------------------------

def create_attention_mask(data_tensor):
    return (data_tensor != 0).int()

def pad_mask_hook(attn_scores, hook, mask):
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

if not USE_TRAIN_CSV:
    all_datasets = {}
    relators_data, relators_labels = load_dataset_from_csv(DATA_PATH / reservedFiles["relators"])

    fileNames = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".csv") and file not in reservedFilesL:
            fileNames.append(file)
    fileNames = sorted(fileNames, key=lambda name: int(name.split('-')[0]))

    all_datasets["relators"] = load_dataset_from_csv(DATA_PATH / "relators.csv")
    for fileName in fileNames:
        dataset_name = fileName.replace(".csv", "")
        data, labels = load_dataset_from_csv(DATA_PATH / fileName)
        all_datasets[dataset_name] = (data, labels)

    print("Loaded datasets:")
    for name in all_datasets:
        print(f"  {name}: {all_datasets[name][0].shape} words, {all_datasets[name][1].shape} labels")

if not USE_TRAIN_CSV:
    all_data_list = []
    all_labels_list = []
    for name, (data, labels) in all_datasets.items():
        all_data_list.append(data)
        all_labels_list.append(labels)
    train_data = torch.cat(all_data_list, dim=0)
    train_labels = torch.cat(all_labels_list, dim=0)

if USE_TRAIN_CSV:
    train_data, train_labels = load_dataset_from_csv(DATA_PATH / "train.csv")

tPerm = torch.randperm(train_data.size(0))
originalTrainDex = torch.argsort(tPerm)

write_dataset_to_csv(DATA_PATH / reservedFiles["train"], train_data, train_labels)

train_data = train_data[tPerm]
train_labels = train_labels[tPerm].to(device1)

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
        data = data.to(device1)
        labels = labels.to(device1)
        all_datasets[name] = (data, labels)

model_checkpoints = []
checkpoint_epochs = []
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

if TRAIN_MODEL:
    for epoch in tqdm.tqdm(range(num_epochs)):
        model.reset_hooks()
        register_pad_mask_hook(model, train_attention_mask)

        train_logits = model(train_data)
        train_loss = loss_fn(train_logits, train_labels)
        train_loss.backward()
        train_losses.append(train_loss.item())

        train_accuracy = accuracy_fn(train_logits, train_labels)
        train_accuracies.append(train_accuracy)

        optimizer.step()
        optimizer.zero_grad()

        with torch.inference_mode():
            model.reset_hooks()
            register_pad_mask_hook(model, test_attention_mask)

            test_logits = model(test_data)
            test_loss = loss_fn(test_logits, test_labels)
            test_losses.append(test_loss.item())

            test_accuracy = accuracy_fn(test_logits, test_labels)
            test_accuracies.append(test_accuracy)

        if ((epoch + 1) % checkpoint_every) == 0:
            checkpoint_epochs.append(epoch)
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

if not TRAIN_MODEL:
    cached_data = torch.load(PTH_LOCATION, weights_only=False)
    model.load_state_dict(cached_data['model'])
    optimizer.load_state_dict(cached_data["optimizer"])
    scheduler.load_state_dict(cached_data["scheduler"])
    model_checkpoints = cached_data["checkpoints"]
    checkpoint_epochs = cached_data["checkpoint_epochs"]
    test_losses = cached_data['test_losses']
    train_losses = cached_data['train_losses']
    train_accuracies = cached_data["train_accuracies"]
    test_accuracies = cached_data["test_accuracies"]

"""
# ---------------------------------------------------------------------------
# Graph Results
# ---------------------------------------------------------------------------

skipBy = 100

def createGraph(yTrain, yTest, yName, title):
    line_fn = partial(line_or_scatter, plot_type="line", return_fig=True)
    fig = line_fn(
        [yTrain[::skipBy], yTest[::skipBy]],
        x=np.arange(0, len(yTrain), skipBy),
        xaxis="Epoch",
        yaxis=yName,
        log_y=True,
        title=title,
        line_labels=['train', 'test'],
        toggle_x=True,
        toggle_y=True,
    )
    return fig

fig1 = createGraph(train_losses, test_losses, "Loss", "Loss Curve for Word Problem")
fig2 = createGraph(train_accuracies, test_accuracies, "Accuracy", "Accuracy Curve for Word Problem")

fig1.show()
fig2.show()

fig1.write_html("loss_curve.html")
fig2.write_html("accuracy_curve.html")

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

torch.set_grad_enabled(False)
gc.collect()
torch.cuda.empty_cache()

# --- Helper Functions ---

def getLogits(model: HookedTransformer, data, getCache: bool = False):
    with torch.inference_mode():
        model.reset_hooks()
        mask = create_attention_mask(data).to(device1)
        register_pad_mask_hook(model, mask)
        if getCache:
            original_logits_og, cache = model.run_with_cache(data)
        else:
            original_logits_og = model(data)
    original_logits = original_logits_og.detach().clone()
    del original_logits_og, mask
    torch.cuda.empty_cache()
    if getCache:
        return original_logits, cache
    else:
        return original_logits

def getPredictions(model: HookedTransformer, data):
    logits = getLogits(model, data, getCache=False)
    if len(logits.shape) == 3:
        logits = logits[:, clsDex]
    preds = logits.argmax(dim=-1)
    del logits
    return preds

# --- Key Weight Matrices ---
W_E = model.embed.W_E[:-1]
print("W_E", W_E.shape)
W_neur = W_E @ model.blocks[0].attn.W_V @ model.blocks[0].attn.W_O @ model.blocks[0].mlp.W_in
print("W_neur", W_neur.shape)
W_logit = model.blocks[0].mlp.W_out @ model.unembed.W_U
print("W_logit", W_logit.shape)

original_logits, cache = getLogits(model, train_data, getCache=True)

original_loss = loss_fn(original_logits, train_labels).item()
print("Original Loss:", original_loss)

# --- Activation Cache Shapes ---
for param_name, param in cache.items():
    print(param_name, param.shape)

# --- Average Attention Heads ---
n_heads = model.cfg.n_heads
seq_len = model.cfg.n_ctx

head_sums = None
num_samples = 0

with torch.inference_mode():
    for layer in range(model.cfg.n_layers):
        for wordIndex in range(len(train_data)):
            attn_patterns = cache["pattern", 0][wordIndex]
            if head_sums is None:
                head_sums = torch.zeros_like(attn_patterns)
            head_sums += attn_patterns
            num_samples += 1

        avg_attn = head_sums / num_samples
        str_tokens = [str(i) for i in range(seq_len)]

        for head in range(n_heads):
            imshow(
                avg_attn[head].detach().cpu(),
                x=str_tokens,
                y=str_tokens,
                xaxis="Key (Attended To)",
                yaxis="Query (Paying Attention)",
                title=f"Layer {layer} Head {head} — Average Attention Across Dataset",
                aspect=None,
            )

# --- Average Attention Pattern over Attention Heads ---
for layer in range(model.cfg.n_layers):
    attention = cache["pattern", layer].mean(dim=0)[:, clsDex, :]
    imshow(
        attention,
        title=f"Average Attention Paid | for token {clsDex} | per head | layer {layer}",
        xaxis="Source",
        yaxis="Head",
        x=[str(i) for i in range(train_data.shape[1])],
        ylabels=[f"{i}" for i in range(attention.shape[0])]
    )

# --- Single Word Analysis ---
wordIndex = 5
wordIndex = originalTrainDex[wordIndex]

print(f"Word: {train_data[wordIndex].tolist()}")

wordTensor = train_data[wordIndex]
wordLabel = train_labels[wordIndex]
wordTensorFit = wordTensor.unsqueeze(0)
pred = getPredictions(model, wordTensorFit)

print(f"Word: {wordTensor.tolist()}")
print(f"Label: {wordLabel}")
print(f"Pred : {pred.item()}")

chosenWord = train_data[wordIndex].unsqueeze(0).to(device1)
print(chosenWord)

str_tokens = [f"{i}" for i, tok in enumerate(chosenWord[0])]

for layer in range(model.cfg.n_layers):
    for head in range(n_heads):
        attn_single_head = cache["pattern", layer][wordIndex, head].detach().cpu()
        imshow(
            attn_single_head,
            x=str_tokens,
            y=str_tokens,
            xaxis="Key (Attended To)",
            yaxis="Query (Paying Attention)",
            title=f"Layer {layer} Head {head} Attention Pattern (Positional Tokens)",
            aspect=None,
        )

token_strs = [str(x) for x in chosenWord.squeeze(0).tolist()]

for layer in range(model.cfg.n_layers):
    attention = cache["pattern", layer][wordIndex][:, clsDex, :]
    imshow(
        attention,
        title=f"Attention Pattern (for word {wordIndex}) over Attention Heads in layer {layer}",
        xaxis="Source",
        yaxis="Head",
        xlabels=token_strs,
        ylabels=[f"{i}" for i in range(attention.shape[0])],
    )

# ---------------------------------------------------------------------------
# Misclassification Graphs (Train)
# ---------------------------------------------------------------------------

words = []
wrongs = []
lengths = []
correct_labels_list = []
dataset_indices = []
total_by_length = defaultdict(int)
misclassified_by_length = defaultdict(int)

train_preds = getPredictions(model, train_data)
correct_mask = (train_preds == train_labels)

for i in range(len(train_preds)):
    input_seq = train_data[i].tolist()
    word_str = ' '.join([str(int(tok)) for tok in input_seq])
    word_len = sum(1 for ch in word_str.split(" ") if ch != "0" and ch.strip() != "")
    word_len = word_len - 1  # subtract special token

    words.append(word_str)
    lengths.append(word_len)
    correct_labels_list.append(int(train_labels[i]))
    dataset_indices.append(i)

    total_by_length[word_len] += 1
    if not correct_mask[i]:
        wrongs.append(1)
        misclassified_by_length[word_len] += 1
    else:
        wrongs.append(0)

print(misclassified_by_length)

misclassified = [
    (words[i], lengths[i], correct_labels_list[i], dataset_indices[i])
    for i in range(len(words)) if wrongs[i] > 0
]

print("\nRandom Sample of Misclassified Train Words:")
for word, length, label, idx in random.sample(misclassified, min(20, len(misclassified))):
    print(f"Word: {word} | Length: {length} | Label: {label} | Dataset Index: {idx}")

misclassified_lengths = [length for _, length, _, _ in misclassified]
plt.figure(figsize=(10, 6))
plt.hist(misclassified_lengths, bins=range(min(misclassified_lengths), max(misclassified_lengths) + 2), edgecolor='black')
plt.title("Length Distribution of Misclassified Train Words (Post Training)")
plt.xlabel("Word Length")
plt.ylabel("Number of Words")
plt.grid(True)
plt.tight_layout()
plt.show()

lengths_sorted = sorted(total_by_length.keys())
percent_misclassified = [100 * misclassified_by_length[l] / total_by_length[l] for l in lengths_sorted]
misclassified_counts = [misclassified_by_length[l] for l in lengths_sorted]

plt.figure(figsize=(10, 6))
bars = plt.bar(lengths_sorted, percent_misclassified, edgecolor='black')
for bar, count in zip(bars, misclassified_counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{count}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.title("Percentage of Misclassified Train Words by Length")
plt.xlabel("Word Length Number")
plt.ylabel("Misclassification Rate (%)")
plt.xticks(lengths_sorted)
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Misclassification Graphs (Test)
# ---------------------------------------------------------------------------

words = []
wrongs = []
lengths = []
correct_labels_list = []
dataset_indices = []
total_by_length = defaultdict(int)
misclassified_by_length = defaultdict(int)

test_preds = getPredictions(model, test_data)
correct_mask = (test_preds == test_labels)

for i in range(len(test_preds)):
    input_seq = test_data[i].tolist()
    word_str = ' '.join([str(int(tok)) for tok in input_seq])
    word_len = sum(1 for ch in word_str.split(" ") if ch != "0" and ch.strip() != "")
    word_len = word_len - 1

    words.append(word_str)
    lengths.append(word_len)
    correct_labels_list.append(int(test_labels[i]))
    dataset_indices.append(i)

    total_by_length[word_len] += 1
    if not correct_mask[i]:
        wrongs.append(1)
        misclassified_by_length[word_len] += 1
    else:
        wrongs.append(0)

misclassified = [
    (words[i], lengths[i], correct_labels_list[i], dataset_indices[i])
    for i in range(len(words)) if wrongs[i] > 0
]

print("\nRandom Sample of Misclassified Test Words:")
for word, length, label, idx in random.sample(misclassified, min(20, len(misclassified))):
    print(f"Word: {word} | Length: {length} | Label: {label} | Dataset Index: {idx}")

misclassified_lengths = [length for _, length, _, _ in misclassified]
plt.figure(figsize=(10, 6))
plt.hist(misclassified_lengths, bins=range(min(misclassified_lengths), max(misclassified_lengths) + 2), edgecolor='black')
plt.title("Length Distribution of Misclassified Test Words (Post Training)")
plt.xlabel("Word Length")
plt.ylabel("Number of Words")
plt.grid(True)
plt.tight_layout()
plt.show()

lengths_sorted = sorted(total_by_length.keys())
percent_misclassified = [100 * misclassified_by_length[l] / total_by_length[l] for l in lengths_sorted]
misclassified_counts = [misclassified_by_length[l] for l in lengths_sorted]

plt.figure(figsize=(10, 6))
bars = plt.bar(lengths_sorted, percent_misclassified, edgecolor='black')
for bar, count in zip(bars, misclassified_counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{count}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.title("Percentage of Misclassified Test Words by Length")
plt.xlabel("Word Length Number")
plt.ylabel("Misclassification Rate (%)")
plt.xticks(lengths_sorted)
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"total word count: {sum(total_by_length.values())}")
print(f"total misclassified count: {sum(misclassified_by_length.values())}")
"""