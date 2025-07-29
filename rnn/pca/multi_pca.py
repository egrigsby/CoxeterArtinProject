import torch
import json
import numpy as np
from sklearn.decomposition import PCA
from collections import Counter
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load JSON data
'''json_path = "/projects/expmmllab/RNN/hidden/json/multi_hidden_states.json"
with open(json_path, "r") as f:
    data = json.load(f)

hidden_states = data["hidden"]  # List of [seq_len, hidden_dim]
labels = data["labels"]         # List of label sequences per sample

# Mean pool hidden states per sequence → fixed-size vector (hidden_dim)
pooled_hidden = [np.mean(np.array(seq), axis=0) for seq in hidden_states]
X = np.stack(pooled_hidden)  # Shape: (num_samples, hidden_dim)

# Extract single label per sequence (most common valid label)
flat_labels = []
for seq in labels:
    arr = np.array(seq, dtype=int)
    valid = arr[arr != -100]
    if len(valid) > 0:
        most_common = Counter(valid).most_common(1)[0][0]
        flat_labels.append(int(most_common))
    else:
        flat_labels.append(-1)

y = np.array(flat_labels)

# Filter out invalid (-1) labels
valid_mask = (y != -1)
X = X[valid_mask]
y = y[valid_mask]

# Print label distribution
unique_labels = sorted(set(y))
print("Unique labels:", unique_labels)
print("Label counts:", Counter(y))

# PCA to 3D
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)

# Plotly Interactive 3D Scatter
df = pd.DataFrame(X_reduced, columns=["PC1", "PC2", "PC3"])
df["label"] = y.astype(str)

fig = px.scatter_3d(
    df, x="PC1", y="PC2", z="PC3",
    color="label",
    title="3D PCA of Hidden States (All Classes 0-5)",
    opacity=1,
    color_discrete_sequence=px.colors.qualitative.T10
)

fig.write_html("/projects/expmmllab/RNN/hidden/pca/big_multi_hidden_states_3D_interactive.html")
print("Interactive 3D PCA saved as HTML")'''

# Load hidden states from JSON
with open('/projects/expmmllab/RNN/hidden/json/multi_hidden_states.json', 'r') as f:
    data = json.load(f)
print("Talked to Json")

# Flatten into a 2D array (num_hidden_vectors, hidden_dim)
hidden_states = []
for seq in data["hidden"]:
    for timestep_vector in seq:
        hidden_states.append(timestep_vector)
hidden_states = np.array(hidden_states, dtype=np.float32)  # Shape: (N, hidden_dim)
hidden_states -= np.mean(hidden_states, axis=0)
print("Shape of hidden states:", hidden_states.shape)
cov_matrix = np.cov(hidden_states, rowvar=False)
print("Covariance matrix shape:", cov_matrix.shape)
eigenvalues = np.linalg.eigvals(cov_matrix)
eigenvalues = np.sort(eigenvalues)[::-1]

print("Top 10 eigenvalues:")
print(eigenvalues[:10])

# python /projects/expmmllab/RNN/hidden/pca/multi_pca.py
