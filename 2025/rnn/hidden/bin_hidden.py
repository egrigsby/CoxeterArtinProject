import torch
import json
import sys
sys.path.append('/projects/expmmllab/RNN')
from binary_RNN import RNNClassifier, load_dataset_from_csv, get_all_hidden_states
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "/projects/expmmllab/RNN/inf_bin_final.pth"
test_path = "/projects/expmmllab/RNN/datasets/56.9kbtest.csv"
vocab_size = 3
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

# Extract hidden states
print("Extracting hidden states...")
hidden_states, labels = get_all_hidden_states(model, test_words, test_labels, max_samples=max_samples)

# Save to JSON
output_path = "/projects/expmmllab/RNN/hidden/json/inf_bin_hidden_states.json"
with open(output_path, "w") as f:
    json.dump({
        "hidden": [h.tolist() for h in hidden_states],
        "labels": [lbl.item() for lbl in labels]
    }, f, indent=2)

print(f"Saved {len(hidden_states)} sequences to {output_path}")

# python /projects/expmmllab/RNN/hidden/bin_hidden.py
