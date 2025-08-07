import torch
import json
import sys
sys.path.append('/projects/expmmllab/LSTMcx')      #change as needed

from binary_LSTM import WordDataset, LSTMCell
from torch.utils.data import DataLoader, Dataset
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import kaleido
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""Paths and params"""      #change all as needed
model_path = "/projects/expmmllab/LSTMcx/logs/bLSTM.pth"
test_path = "/projects/expmmllab/LSTMcx/datasets/hyptestb.csv"
out_path_h = "/projects/expmmllab/LSTMcx/logs/hiddenstatesb.json"
out_path_c = "/projects/expmmllab/LSTMcx/logs/cellstatesb.json"

embedding_dim = 4          #params MUST match saved model; else everything breaks
hidden_size = 16
vocab_size = 4
batch_size = 512
num_classes = 2


"""Loading model data"""
model = LSTMCell(vocab_size, embedding_dim, hidden_size)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

#load test data
test_set = WordDataset(test_path)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

#extract hidden states
print("Extracting hidden and cell states...")
hidden = []
cell = []
labels = []

start = time.time()
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out, logits, h, c = model(x)
        hidden.append(h.cpu()) 
        cell.append(c.cpu())
        labels.append(y.cpu())
        pred_labels_seq = (logits.squeeze(0) > 0).long().tolist()

print(f"Extraction took {time.time()-start:.4f} seconds.")


"""Saving options"""
save_json = False
if save_json:
    #convert tensors to lists
    hidden = torch.cat(hidden, dim=0).tolist()
    cell = torch.cat(cell, dim=0).tolist()
    labels = torch.cat(labels, dim=0).tolist()

    print("Writing hidden states to JSON file...")
    start = time.time()
    with open(out_path_h, "w") as f:
        json.dump({"hidden": hidden, "labels": labels}, f, indent=2)
    print(f"Saved {len(hidden)} hidden sequences in {time.time()-start:.4f} seconds :)")

    print("Writing cell states to JSON file...")
    start = time.time()
    with open(out_path_c, "w") as f:
        json.dump({"cell": cell, "labels": labels}, f, indent=2)
    print(f"Saved {len(cell)} cell sequences in {time.time()-start:.4f} seconds :)")

#save to pt file
torch.save({"hidden": hidden, "cell": cell, "labels": labels}, "/projects/expmmllab/bstates.pt")             #change as needed
print("Saved to bstates.pt.")


"""Processing"""
#concatenate along batch dimension
hidden_tensor = torch.cat(hidden, dim=0)  #shape: (total_sequences, seq_len, hidden_dim)
cell_tensor = torch.cat(cell, dim=0) 
labels_tensor = torch.cat(labels, dim=0)
#print("Combined hidden shape:", hidden_tensor.shape)
#print("Combined cell shape:", cell_tensor.shape)
#print("Combined labels shape:", cell_tensor.shape)

#mean pooling
hidden_tensor = hidden_tensor.cpu()
cell_tensor = cell_tensor.cpu()
labels_tensor = labels_tensor.cpu()

#mean pool over sequence length, turning shape from [seq_len, hidden_dim] to just [hidden_dim]??
pooled_hidden = [seq.mean(dim=0).numpy() for seq in hidden_tensor]
X = np.stack(pooled_hidden)
#print("Pooled X shape:", X.shape)

# Print label distribution to confirm all classes present
unique_labels = np.unique(labels_tensor)
#print("Unique labels:", unique_labels)


"""Individual sequence trajectory"""
#select a sequence
seq_id = 1
seq_hidden = hidden_tensor[seq_id] #shape [seq_len, hidden_dim]
seq_pred = pred_labels_seq[seq_id]

#pca on selected sequence
print("Starting PCA...")
start = time.time()
pca = PCA(n_components=2)
ht_pca = pca.fit_transform(seq_hidden)     #fits model and applies dimensionality reduction to seq_hidden
print(f"PCA took {time.time()-start:.4f} seconds.")
#print(pca.explained_variance_ratio_)        #debugging

timesteps = list(range(len(ht_pca)))

#plotting
print("Plotting hidden state trajectory...")
start = time.time()
fig = go.Figure()
fig.add_trace(go.Scatter(
    x = ht_pca[:, 0],
    y = ht_pca[:, 1],
    mode = "lines+markers",
    marker = dict(
        size = 10,
        color = seq_pred,        
        colorscale = "Viridis",
        ),
    line=dict(color='blue', width=1),
    name=f"Sequence {seq_id}",
    # text = [f'Time step {t+1}' for t in timesteps],
    text = [f'Time step {t+1}, Predicted label {seq_pred[t]}' for t in timesteps],
    hoverinfo='text',
))

fig.update_layout(
    title=f"Hidden State Trajectory for Sequence {seq_id}",
    xaxis_title="PC1",
    yaxis_title="PC2"
    ),

print(f"Plotted in {time.time()-start:.4f} seconds.")
fig.write_image('/projects/expmmllab/bht.png')         #change as needed
fig.write_html('/projects/expmmllab/bht.html')         #change as needed
print("Hidden state trajectory saved :)")


#select a sequence
seq_cell = cell_tensor[seq_id] #shape [seq_len, hidden_dim]

#pca on selected sequence
print("Starting PCA...")
start = time.time()
pca = PCA(n_components=2)
ct_pca = pca.fit_transform(seq_cell)         #fits model and applies dimensionality reduction to seq_cell
print(f"PCA took {time.time()-start:.4f} seconds.")
#print(pca.explained_variance_ratio_)        #debugging

# timesteps = list(range(len(ct_pca)))

#plotting
print("Plotting cell state trajectory...")
start = time.time()
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ct_pca[:,0], 
    y=ct_pca[:,1],
    mode='lines+markers',
    # text=[f'Time step {t+1}' for t in timesteps],
    # hoverinfo='text',
    marker=dict(
        size=11, 
        color=seq_pred,
        # color=timesteps,
        # colorscale='Sunset',
        # colorbar=dict(title="Time step")
    ),
    text = [f'Time step {t+1}, Predicted label {seq_pred[t]}' for t in timesteps],
    hoverinfo='text',
    line=dict(color='red', width=1),
    name=f"Sequence {seq_id}"
))

fig.update_layout(
    title=f"Cell State Trajectory for Sequence {seq_id}",
    xaxis_title="PC1",
    yaxis_title="PC2"
)
print(f"Plotted in {time.time()-start:.4f} seconds.")
fig.write_image('/projects/expmmllab/bct.png')             #change as needed
fig.write_html('/projects/expmmllab/bct.html')             #change as needed
print("Cell state trajectory saved :)")


"""3D Individual Sequence Hidden State Trajectory"""
pca = PCA(n_components=3)
ht3_pca = pca.fit_transform(seq_hidden)   

#3D plotting
print("Plotting 3D hidden state trajectory...")
start = time.time()
fig = go.Figure(data=[go.Scatter3d(
    x=ht3_pca[:,0], 
    y=ht3_pca[:,1],
    z=ht3_pca[:,2],
    mode='lines+markers',
    text=[f'Time step {t+1}' for t in timesteps],
    hoverinfo='text',
    marker=dict(
        size=7, 
        color=timesteps,
        colorscale='Viridis',
        colorbar=dict(title="Time step"),
        opacity=0.8
    ),
    line=dict(color='blue', width=1),
    name=f"Sequence {seq_id}"
)])

fig.update_layout(
    title=f"3D Hidden State Trajectory for Sequence {seq_id}",
    xaxis_title="PC1",
    yaxis_title="PC2"
)
print(f"Plotted in {time.time()-start:.4f} seconds.")
fig.write_image('/projects/expmmllab/bht3d.png')               #change as needed
fig.write_html('/projects/expmmllab/bht3d.html')               #change as needed
print("3D hidden state trajectory saved :)")


"""3D Individual Sequence Hidden State Trajectory"""
pca = PCA(n_components=3)
ct3_pca = pca.fit_transform(seq_cell)   

#3D plotting
print("Plotting 3D cell state trajectory...")
start = time.time()
fig = go.Figure(data=[go.Scatter3d(
    x=ct3_pca[:,0], 
    y=ct3_pca[:,1],
    z=ct3_pca[:,2],
    mode='lines+markers',
    text=[f'Time step {t+1}' for t in timesteps],
    hoverinfo='text',
    marker=dict(
        size=7, 
        color=timesteps,
        colorscale='Sunset',
        colorbar=dict(title="Time step"),
        opacity=0.8
    ),
    line=dict(color='red', width=1),
    name=f"Sequence {seq_id}"
)])

fig.update_layout(
    title=f"3D Cell State Trajectory for Sequence {seq_id}",
    xaxis_title="PC1",
    yaxis_title="PC2"
)
print(f"Plotted in {time.time()-start:.4f} seconds.")
fig.write_image('/projects/expmmllab/bct3d.png')             #change as needed
fig.write_html('/projects/expmmllab/bct3d.html')             #change as needed
print("3D cell state trajectory saved :)")


"""Scatterplot graphing"""
#pca
print("Starting PCA...")
start = time.time()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)     #fits model and applies dimensionality reduction to X
print(f"PCA took {time.time()-start:.4f} seconds.")
#print(pca.explained_variance_ratio_)        #debugging

fig = go.Figure()
colors = px.colors.qualitative.T10
y = labels_tensor.detach().cpu().numpy()

#one scatter trace per label class
for i, label in enumerate(unique_labels):
    mask = y == label
    fig.add_trace(go.Scatter(
        x=X_pca[mask, 0],
        y=X_pca[mask, 1],
        mode='markers',
        name=f"{label}",
        marker=dict(color=colors[i % len(colors)], size=6, line=dict(width=0.5, color='black'))
    ))

fig.update_layout(
    title="2D PCA of Hidden States",
    xaxis_title="PC1",
    yaxis_title="PC2",
    legend_title="Classes",
    width=800,
    height=600
)
fig.write_image("/projects/expmmllab/bhPCA.png")         #change as needed
fig.write_html("/projects/expmmllab/bhPCA.html")         #change as needed
print("PCA plot saved :)")

"""3D scatterplot graphing"""
#pca
print("Starting PCA...")
start = time.time()
pca = PCA(n_components=3)
hs3_pca = pca.fit_transform(X)     #fits model and applies dimensionality reduction to X
print(f"PCA took {time.time()-start:.4f} seconds.")
print(pca.explained_variance_ratio_)        #debugging

fig = go.Figure()
colors = px.colors.qualitative.T10
y = labels_tensor.detach().cpu().numpy()

#one scatter trace per label class
for i, label in enumerate(unique_labels):
    mask = y == label
    fig.add_trace(go.Scatter3d(
        x=hs3_pca[mask, 0],
        y=hs3_pca[mask, 1],
        z=hs3_pca[mask, 2],
        mode='markers',
        name=f"{label}",
        marker=dict(color=colors[i % len(colors)], size=6, line=dict(width=0.5, color='black'))
    ))

fig.update_layout(
    title="3D PCA of Hidden States",
    xaxis_title="PC1",
    yaxis_title="PC2",
    legend_title="Classes",
)
fig.write_image("/projects/expmmllab/bh3dPCA.png")         #change as needed
fig.write_html("/projects/expmmllab/bh3dPCA.html")         #change as needed
print("PCA plot saved :)")

#debugging more
# var_per_time = hidden_tensor.var(dim=1)  # shape: (num_seqs, hidden_dim)
# mean_var = var_per_time.mean().item()
# print(f"Avg variance across time (per neuron): {mean_var:.6f}")

X = np.array(X, dtype=np.float32)  # Shape: (N, hidden_dim)
X -= np.mean(X, axis=0)
print("Shape of hidden states:", X.shape)
cov_matrix = np.cov(X, rowvar=False)
print("Covariance matrix shape:", cov_matrix.shape)
eigenvalues = np.linalg.eigvals(cov_matrix)
eigenvalues = np.sort(eigenvalues)[::-1]

print(f"Top 10 eigenvalues:{eigenvalues[:10]}")

"""Request gpus"""
#srun --gres=gpu:1 --time=06:00:00 --pty bash   
#nvidia-smi
#python /CoxeterArtinProject/lstm/bLSTM_getstates.py
