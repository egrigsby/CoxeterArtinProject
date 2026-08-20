import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import ast
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
set_seed(42)

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Converts lists of sequences and labels into a DataLoader
def make_dataloader(words, labels, batch_size=64):
    padded = pad_sequence(words, batch_first=True, padding_value=0)  # Pad sequences with 0
    return DataLoader(TensorDataset(padded, labels), batch_size=batch_size, shuffle=True)

# Define the RNN classifier model
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)  # Embed tokens
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True)  # RNN layer
        self.fc = nn.Linear(hidden_dim, 1)  # Final output layer (binary classification)

    def forward(self, x):
        lengths = (x != 0).sum(dim=1)  # Get lengths of sequences
        embedded = self.embedding(x)  # Convert input tokens to embeddings
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)  # Pack sequences
        _, hidden = self.rnn(packed)  # Run through RNN; get hidden states
        out = hidden[-1]  # Take output from last RNN layer
        return self.fc(out)  # Output logits
    
    # Return all hidden states per timestep
    def extract_hidden_states(self, x):
        self.eval()
        with torch.no_grad():
            lengths = (x != 0).sum(dim = 1)
            embedded = self.embedding(x)
            packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            output, _= self.rnn(packed)
            unpacked, _= nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return unpacked # (batch_size, seq_len, hidden_dim)
        
# Get hidden states for all sequences in the dataset
def get_all_hidden_states(model, words, labels, batch_size=512, max_samples=500):
    dataloader = make_dataloader(words, labels, batch_size)
    all_hidden_states = []
    all_labels = []
    for x,y in dataloader:
        x = x.to(device)
        hidden_states = model.extract_hidden_states(x) # (batch_size, seq_len, hidden_dim)
        for hs, label in zip(hidden_states.cpu(), y.cpu()):
            all_hidden_states.append(hs) # list of (seq_len, hidden_dim) tensors
            all_labels.append(label)
            if len(all_hidden_states) >= max_samples:
                return all_hidden_states, all_labels
    return all_hidden_states, all_labels

# Binary Cross-Entropy loss with label smoothing
class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()  # Combines sigmoid + BCELoss

    def forward(self, preds, targets):
        # Smooth labels: reduce confidence of true labels slightly
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(preds, targets)

# Computes moving average of a list (for smoother loss curves)
def moving_average(data, window=5):
    return [sum(data[max(0, i-window):i+1]) / (i - max(0, i-window) + 1) for i in range(len(data))]

def plot_confusion_matrix(model, dataloader, epoch=None, out_dir="."):
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze()  # Shape: (batch,)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Inf Tri Confusion Matrix (Epoch {epoch})" if epoch is not None else "Confusion Matrix")
    plt.tight_layout()
    filename = f"{out_dir}/inf_confusion_matrix_epoch{epoch}.png" if epoch is not None else f"{out_dir}/confusion_matrix.png"
    plt.savefig(filename)
    plt.close()
    #print(f"Saved confusion matrix to {filename}")

# Full training loop with plotting and early stopping
def train_with_plot(model, train_words, train_labels, test_words, test_labels, epochs=1000, lr=1e-3, batch_size=64, patience=20):
    train_loader = make_dataloader(train_words, train_labels, batch_size)
    test_loader  = make_dataloader(test_words, test_labels, batch_size)
    model.to(device)

    criterion = LabelSmoothingBCELoss(smoothing=.05)  # Use smoothed BCE loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # Optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5)  # Reduces LR on plateau

    train_losses, test_losses, test_accuracies = [], [], []

    best_loss = float('inf')  # Track best test loss
    trigger_times = 0  # Counter for early stopping

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x).squeeze()  # Model predictions
            loss = criterion(preds, y.float())  # Compute loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))  # Avg train loss

        # Evaluate on test set
        model.eval()
        t_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).squeeze()
                t_loss += criterion(preds, y.float()).item()
                probs = torch.sigmoid(preds)  # Convert logits to probabilities
                predicted = (probs > 0.5).int()  # Convert to binary predictions
                correct += (predicted == y).sum().item()
                total += y.size(0)
        avg_test_loss = t_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accuracies.append(correct / total)  # Compute accuracy
        scheduler.step(avg_test_loss)  # Adjust LR if test loss plateaus

        print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, "
              f"Test Loss={test_losses[-1]:.4f}, "
              f"Test Acc={test_accuracies[-1]:.2%}")
        
        if (epoch + 1) % 1 == 0:
            plot_confusion_matrix(model, test_loader, epoch=epoch+1, out_dir="/projects/expmmllab/RNN/confusion/bin_confusion")

        # Early stopping logic
        '''if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break'''

    # Plot loss and accuracy curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(moving_average(train_losses), label='Train Loss')
    plt.plot(moving_average(test_losses), label='Test Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig("/projects/expmmllab/RNN/loss&accuracy/bin_accuracy.png")
    print("Plots saved as bin_accuracy.png")

    torch.save(model.state_dict(), "inf_bin_final.pth") # Saves trained weights
    print("Model saved as inf_bin_final.pth")

# Load tokenized dataset from CSV file
def load_dataset_from_csv(path):
    df = pd.read_csv(path, header=None, names=['tokens', 'label'])  # Read CSV file
    df['tokens'] = df['tokens'].apply(ast.literal_eval)  # Convert string repr to list
    df['tokens'] = df['tokens'].apply(lambda lst: [int(x) for x in lst])  # Convert to ints
    words = [torch.tensor(seq) for seq in df['tokens']]  # Convert to tensors
    labels = torch.tensor(df['label'].astype(int).values)  # Convert labels to tensor
    return words, labels

# Main execution logic
if __name__ == "__main__":
    train_path = "/projects/expmmllab/RNN/datasets/56.9kbtrain.csv"
    test_path  = "/projects/expmmllab/RNN/datasets/56.9kbtest.csv"

    print("Loading datasets…")
    train_words, train_labels = load_dataset_from_csv(train_path)
    test_words,  test_labels  = load_dataset_from_csv(test_path)

    # Flatten all token sequences into a single list to compute vocab range
    all_tokens = [token.item() if isinstance(token, torch.Tensor) else token
                  for word in (train_words + test_words) for token in word]

    # Sanity check for token values
    if not all_tokens:
        print("No tokens found in dataset!")
    else:
        print("Min token value in dataset:", min(all_tokens))
        print("Max token value in dataset:", max(all_tokens))

        # Create RNN model
        model = RNNClassifier(
            vocab_size=max(all_tokens),
            embed_dim=32,
            hidden_dim=64,
            num_layers=1,
        ).to(device)

        # Evaluate immediately, without training
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in make_dataloader(test_words, test_labels):
                x, y = x.to(device), y.to(device)
                preds = model(x).squeeze()
                probs = torch.sigmoid(preds)
                predicted = (probs > 0.5).int()
                correct += (predicted == y).sum().item()
                total += y.size(0)
        print("Accuracy before training:", correct / total)

        print("Training…")
        start = time.time()
        train_with_plot(model, train_words, train_labels, test_words, test_labels, epochs=1000, lr=8e-4, batch_size=512, patience=50)

# Print training time
start = time.time()
if time.time() - start > 60:
    print("Training took", (time.time() - start) / 60, "minutes")
else:
    print("Training took", time.time() - start, "seconds")

'''
Command-line GPU usage:
srun --gres=gpu:1 --time=06:00:00 --pty bash
nvidia-smi
python /projects/expmmllab/RNN/binary_RNN.py
'''
