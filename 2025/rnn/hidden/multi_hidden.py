import torch
import json
import sys
sys.path.append('/projects/expmmllab/RNN')
from multi_RNN import RNNClassifier, load_dataset_from_csv, get_all_hidden_states
import time
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "/projects/expmmllab/RNN/multi_final.pth"
test_path = "/projects/expmmllab/RNN/datasets/25kmtest.csv"
vocab_size = 2
embed_dim = 32
hidden_dim = 64
num_layers = 1
max_samples = 100000

# Load trained model
model = RNNClassifier(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=num_layers)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load test data
test_words, test_labels = load_dataset_from_csv(test_path)

# Sanity check on labels
all_flat_labels = [int(lab) for seq in test_labels for lab in seq if lab != -100]
print("Unique labels in test set:", sorted(set(all_flat_labels)))
print("Label counts in test set:", Counter(all_flat_labels))

# Extract hidden states
print("Extracting hidden states...")
hidden_states, labels = get_all_hidden_states(model, test_words, test_labels, max_samples=max_samples)

# Save to JSON
output_path = "/projects/expmmllab/RNN/hidden/json/multi_hidden_states.json"
with open(output_path, "w") as f:
    json.dump({
        "hidden": [h.tolist() for h in hidden_states],
        "labels": [lbl.tolist() for lbl in labels]
    }, f, indent=2)

print(f"Saved {len(hidden_states)} sequences to 'multi_hidden_states.json'")

# python /projects/expmmllab/RNN/hidden/multi_hidden.py
