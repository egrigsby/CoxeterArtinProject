import torch
import json
import random
import sys
import numpy as np
sys.path.append('/projects/expmmllab/RNN')
from binary_RNN import load_dataset_from_csv
test_words, test_labels = load_dataset_from_csv("/projects/expmmllab/RNN/datasets/129.3btest.csv")
from binary_RNN import RNNClassifier

random.seed(None)
np.random.seed(None)
torch.manual_seed(torch.initial_seed())
print("Random seed set")

with open("/projects/expmmllab/RNN/hidden/json/bin_hidden_states.json") as f:
    data = json.load(f)
print("Talked to json")

hidden = data["hidden"]   # List of (seq_len, hidden_dim) sequences
labels = data["labels"]

index = random.randint(0, len(hidden) - 1)

csv_path = "/projects/expmmllab/RNN/datasets/129.3btest.csv"
test_words, test_labels = load_dataset_from_csv(csv_path)
sequence = test_words[index].tolist()

vocab_size = 3
model = RNNClassifier(vocab_size=vocab_size, embed_dim=32, hidden_dim=64, num_layers=1)
model.load_state_dict(torch.load("/projects/expmmllab/RNN/bin_final.pth", map_location="cuda"))
model.eval()
print("Model loaded")

sequence_hidden = torch.tensor(hidden[index])  # Shape: (seq_len, hidden_dim)
# Apply output layer to each hidden state
with torch.no_grad():
    logits = model.fc(sequence_hidden).squeeze(-1)  # Shape: (seq_len,)
    preds = (torch.sigmoid(logits) > 0.5).int()     # Shape: (seq_len,)
print(f"Sequence and label at index {index+1}: {sequence}, {test_labels[index].item()}")
print(f"Predicted class at each timestep: {preds.tolist()}")

# python /projects/expmmllab/RNN/hidden/bin_trajectory.py
