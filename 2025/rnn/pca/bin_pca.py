import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px

# Load JSON data
json_path = "/projects/expmmllab/RNN/hidden/json/inf_bin_hidden_states.json"
with open(json_path, "r") as f:
    data = json.load(f)

hidden_states = data["hidden"]  # List of [seq_len, hidden_dim]
labels = data["labels"]

# Convert to fixed-size vectors (mean pooling)
pooled_hidden = [np.mean(np.array(seq), axis=0) for seq in hidden_states]  # shape: (num_samples, hidden_dim)
X = np.stack(pooled_hidden)
y = np.array(labels)

# Apply PCA to reduce to 3 components
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)

# Create DataFrame for Plotly
df = pd.DataFrame(X_reduced, columns=["PC1", "PC2", "PC3"])
df["label"] = y

# Create interactive 3D scatter plot
fig = px.scatter_3d(
    df, x="PC1", y="PC2", z="PC3",
    color=df["label"].astype(str),  # Convert labels to strings for categorical coloring
    title="3D PCA of Hidden States (Inf Triangle)",
    opacity=1,
    color_discrete_sequence=px.colors.qualitative.T10
)

# Save to HTML
fig.write_html("/projects/expmmllab/RNN/hidden/pca/inf_bin_hidden_states_3D_interactive.html")
print("Interactive 3D PCA saved as HTML")

# Load hidden states from JSON
with open('/projects/expmmllab/RNN/hidden/json/inf_bin_hidden_states.json', 'r') as f:
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

#  python /projects/expmmllab/RNN/hidden/pca/bin_pca.py
