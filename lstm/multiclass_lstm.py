import torch
import torch.nn as nn   #neural network functions
import torch.optim as optim   #optimizers
from torch.utils.data import DataLoader, Dataset, random_split

import random
import numpy as np
import pandas as pd
import ast
import time  #for simple timer

import matplotlib.pyplot as plt  #data visualization
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plotly.graph_objects as go


#set device to GPU if available and use CPU otherwise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}...")


"""Random seed"""
def set_seed(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
set_seed(42)
# torch.backends.cudnn.deterministic = True


"""Wrong predictions"""
def save_wrong(name):   #save the incorrect predictions to a CSV file
  wrong_csv=f'/projects/expmmllab/{name}.csv'             #change file path as needed!!
  wrong_df = pd.DataFrame(wrong)
  wrong_df.to_csv(wrong_csv, index=False)
  print("Saved wrongs :)")


"""Graphing functions"""
def log_graph():
  plt.figure(figsize=(10,5))

  if validation == True:
    plt.title("Training, Validation, and Testing Loss")
    plt.plot(train_losses,label="Training Loss")
    plt.plot(val_losses,label="Validation Loss")
  else:
    plt.title("Training and Testing Loss")
    plt.plot(train_losses,label="Training Loss")
  plt.plot(test_losses,label="Testing Loss")

  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.yscale('log')
  plt.legend()

  plt.savefig("/projects/expmmllab/logscalelossm.png")      #change file path as needed!!
  print("Log scale loss graph saved :)")
  plt.show()

def lin_graph():
  plt.figure(figsize=(10,5))

  if validation == True:
    plt.title("Training, Validation, and Testing Loss")
    plt.plot(train_losses,label="Training Loss")
    plt.plot(val_losses,label="Validation Loss")
  else:
    plt.title("Training and Testing Loss")
    plt.plot(train_losses,label="Training Loss")
  plt.plot(test_losses,label="Testing Loss")

  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend()

  plt.savefig("/projects/expmmllab/linscalelossm.png")      #change file path as needed!!
  print("Linear scale loss graph saved :)")
  plt.show()

def acc_graph():
  plt.figure(figsize=(10,5))

  if validation == True:
    plt.title("Training, Validation, and Testing Accuracy")
    plt.plot(train_accs,label="Training Accuracy")
    plt.plot(val_accs,label="Validation Accuracy")
  else:
    plt.title("Training and Testing Accuracy")
    plt.plot(train_accs,label="Training Accuracy")
  plt.plot(test_accs,label="Testing Accuracy")

  plt.xlabel("Epochs")
  plt.ylabel("Accuracy (%)")
  plt.legend()

  plt.savefig("/projects/expmmllab/accuracym.png")        #change file path as needed!!
  print("Accuracy graph saved :)")
  plt.show()


"""Confusion matrices"""
def get_confusion_matrix(model, dataloader, num_classes):
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
           # x, y = x.to(device), y.to(device)
            logits = model(x)
            if isinstance(logits, tuple):
              logits = logits[1]
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().view(-1).tolist())
            all_targets.extend(y.cpu().view(-1).tolist())

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    return cm

def plot_confusion_matrix(model, dataloader, num_classes):
    cm = get_confusion_matrix(model, dataloader, num_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["e", "1", "2", "12", "21", "121"])
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"mLSTM Confusion Matrix (Epoch {epoch+1})" if epoch is not None else "mLSTM Confusion Matrix")
    plt.tight_layout()
    filename = f"/projects/expmmllab/mcm_epoch{epoch+1}.png" if epoch is not None else f"/projects/expmmllab/mcm.png"      #change file path as needed!!
    plt.savefig(filename)
    plt.close()
    print("Confusion matrix saved :)")

def matrix_slider(matrices, class_labels):
    frames = []
    for i, cm in enumerate(matrices):
      heatmap = go.Heatmap(
        z=cm, 
        colorscale='Blues', 
        zmin=0, zmax=np.max(matrices), 
        text=cm.astype(str), 
        texttemplate="%{text}", 
        hovertemplate='Actual: %{y}'
      )
      frames.append(go.Frame(data=[heatmap], name=str(i)))

    #initial frame
    init_cm = matrices[0]
    heatmap = go.Heatmap(
      z=init_cm, 
      colorscale='Blues', 
      zmin=0, 
      zmax=np.max(matrices),    #normalise across matrices?
      text=init_cm.astype(str),
      texttemplate="%{text}",
      hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    )
    
    layout = go.Layout(
      title='mLSTM Confusion Matrices',
      width=600,
      height=600,
      yaxis_scaleanchor="x",
      yaxis_scaleratio=1,
      xaxis=dict(
        title='Predicted Label',
        tickmode='array',
        tickvals=list(range(num_classes)),
        ticktext=class_labels,
      ),
      yaxis=dict(
        title='True Label',
        tickmode='array',
        tickvals=list(range(num_classes)),
        ticktext=class_labels,
        autorange='reversed'
      ),
      updatemenus=[dict(
        type='buttons',
        buttons=[dict(label='Play',
                      method='animate',
                      args=[None, {"frame": {"duration": 500, "redraw": True},
                                   "fromcurrent": True}]),
                 dict(label='Stop',
                      method='animate',
                      args=[[None], {"frame": {"duration": 0, "redraw": False},
                                     "mode": "immediate",
                                     "transition": {"duration": 0}}])]
    )],
      sliders=[dict(
        steps=[dict(method='animate',
                    args=[[str(k)], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate"}],
                    label=f'Epoch {k+1}') for k in range(num_epochs)],
        transition={"duration": 0},
        x=0.1, y=0,
        len=0.9
    )]
  )
    fig = go.Figure(data=[heatmap], layout=layout, frames=frames)
    fig.write_html("/projects/expmmllab/mcm_slider.html")         #change file path as needed!!
    print("Matrix slider saved :)")


"""Prep dataset"""
class WordDataset(Dataset):
  def __init__(self, data_dir):
    df = pd.read_csv(data_dir)
    df['tokens'] = df['tokens'].apply(ast.literal_eval)   #turns string representations of list under column 'tokens' into actual lists
    df['label'] = df['label'].apply(ast.literal_eval)

    self.data = []
    for seq in df['tokens']:
      int_seq = [int(token) for token in seq]    #turns list of strings into integer sequence (list of ints)
      self.data.append(int_seq)

    self.labels = []
    for seq in df['label']:
      int_seq = [int(label) for label in seq]    
      self.labels.append(int_seq)

    self.seq_len = max(len(seq) for seq in self.data)  #read seq length from padded data

    #find max number of tokens to determine vocab_size
    all_tokens = [token for seq in self.data for token in seq]
    self.vocab_size = max(all_tokens) + 1

  def __len__(self):
    return len(self.data)    #returns the total number of samples in the dataset

  def __getitem__(self, index):
    words = torch.tensor(self.data[index], dtype=torch.long)
    labels = torch.tensor(self.labels[index], dtype=torch.long)
    return words, labels     #return word tensor and label tensor

"""Define multi-class classification model"""
class MultiClassLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
      super().__init__()
      self.hidden_size = hidden_size
      self.num_layers = num_layers
      self.embedding = nn.Embedding(vocab_size, embedding_dim)   #embedding layer
      self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
      self.fc = nn.Linear(hidden_size, num_classes)     #fully connected layer for multiclass classification

    def forward(self, x):
      embedded = self.embedding(x)    #passing through embedding layer

      out, _ = self.lstm(embedded)    #shape: (batch_size, seq_len, hidden_size)
      out = self.fc(out)              #shape: (batch_size, seq_len, num_classes)

      return out     #raw logits for cross entropy loss

class LSTMCell(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
      super().__init__()
      self.hidden_size = hidden_size
      self.embedding = nn.Embedding(vocab_size, embedding_dim)  
      self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_size)
      self.fc = nn.Linear(hidden_size, num_classes) 

    def forward(self, x):
      embedded = self.embedding(x)
      batch_size, seq_len = x.size()   #get batch size from embedded tensor

      #manually initialize hidden and cell states to zero and proper dimensions
      h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
      c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)

      hidden_states = []
      cell_states = []

      for t in range(seq_len):
        h_t, c_t = self.lstm_cell(embedded[:,t,:], (h_t, c_t))
        hidden_states.append(h_t)
        cell_states.append(c_t)

      hidden_states = torch.stack(hidden_states, dim=1)
      cell_states = torch.stack(cell_states, dim=1)

      final_hidden = hidden_states[:,t,:]
      out = self.fc(final_hidden)            
      logits = self.fc(hidden_states)  

      return out, logits, hidden_states, cell_states     


"""Options"""
#LSTMCell toggle: tracks and returns all hidden and cell states for later analysis if True; otherwise run with normal module
return_states = False

#validation toggle: True for on, False for off
validation = False

#plot confusion matrices as individual pngs if True; otherwise a slider html is created
print_cm = False

#wrong prediction csv generation
save_wrong_preds = False


"""Model parameters"""
embedding_dim = 32                  #token embedding dimension
hidden_size = 6                     #size of LSTM hidden state
num_layers = 1                      #number of LSTM layers
num_classes = 6                     #number of label classes
class_labels = ['e', '1', '2', '12', '21', '121']
batch_size = 512
num_epochs = 1000


"""Load datasets"""
#training dataset
train_set = WordDataset(data_dir='/CoxeterArtinProject/datasets/15ktrainm.csv') #change file path as needed; if a syntax error is raised, make sure paths contain backslashes, NOT forward slashes
train_loader = DataLoader(train_set, batch_size, shuffle=True)

vocab_size = train_set.vocab_size       #build vocab size on training data

#testing datasets
testing_sets = WordDataset(data_dir='/CoxeterArtinProject/datasets/15ktestm.csv')    #change file path as needed!!
test_loader = DataLoader(testing_sets, batch_size, shuffle=False)

#split testing dataset into validation and test sets
if validation == True:
  val_size = int(len(testing_sets) * 0.2)   #20% for validation, 80% for testing
  test_size = len(testing_sets) - val_size
  val_set, test_set = random_split(testing_sets, [val_size, test_size]) 
  val_loader = DataLoader(val_set, batch_size, shuffle=False)
  test_loader = DataLoader(test_set, batch_size, shuffle=False)


"""Load datasets"""
if return_states == True:
  model = LSTMCell(vocab_size, embedding_dim, hidden_size, num_classes)
else:
  model = MultiClassLSTM(vocab_size, embedding_dim, hidden_size, num_layers, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


"""Storage"""
#loss storage
if validation == True:
  train_losses, val_losses, test_losses = [],[],[]
  train_accs, val_accs, test_accs = [],[],[]
else:
  train_losses, test_losses = [],[]
  train_accs, test_accs = [],[]

#cm storage
confusion_matrices = []

#wrong prediction storage
wrong=[] 
wrong_set = set()  


"""Training loop"""
if __name__ == "__main__":
  print(model)
  print(f"Running {num_epochs} epochs...")

  start = time.time()  #timer start

  for epoch in range(num_epochs):
    train_correct = 0
    train_total = 0
    model.train()
    training_loss = 0.0    #running loss

    #training
    for batch_item in train_loader:
        X_batch = batch_item[0]    #input token seq
        y_batch = batch_item[1]    #label seq
        outputs = model(X_batch)   #(batch_size, seq_len, num_classes)

        if isinstance(outputs, tuple):
          outputs = outputs[1]           #if out is returned as a tuple, as in LSTMCell, take only logits
      
        outputs = outputs.view(-1, num_classes)     #flatten tokens
        y_batch = y_batch.view(-1)                  #flatten tokens
        loss = criterion(outputs, y_batch)

        loss.backward()    #run backpropagation
        optimizer.step()
        optimizer.zero_grad()

        training_loss += loss.item()

        with torch.no_grad():
          _, predicted = torch.max(outputs, dim=1)                 #max over classes dim
          train_total += y_batch.numel()                           #total tokens (batch_size * seq_len)
          train_correct += (predicted == y_batch).sum().item()

    torch.save(model.state_dict(), "/projects/expmmllab/mLSTM.pth")       #saves trained weights; change file path as needed

    avg_train_loss = training_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accs.append(100 * train_correct / train_total)


    #validation
    if validation == True:
      val_correct = 0
      val_total = 0
      model.eval()
      validation_loss = 0.0

      for batch_item in val_loader:
          X_batch = batch_item[0]
          y_batch = batch_item[1]
          outputs = model(X_batch) 

          if isinstance(outputs, tuple):
            outputs = outputs[1]    
      
          outputs = outputs.view(-1, num_classes)  
          y_batch = y_batch.view(-1)  
          loss = criterion(outputs, y_batch)
          validation_loss += loss.item()

          with torch.no_grad():
            _, predicted = torch.max(outputs, dim=1)
            val_total += y_batch.numel()
            val_correct += (predicted == y_batch).sum().item()

      avg_val_loss = validation_loss / len(val_loader)
      val_losses.append(avg_val_loss)
      val_accs.append(100 * val_correct / val_total)

    #testing
    test_correct = 0
    test_total = 0
    model.eval()
    testing_loss = 0.0

    for batch_item in test_loader:
      X_batch = batch_item[0]
      y_batch = batch_item[1]
      outputs = model(X_batch)

      if isinstance(outputs, tuple):
        outputs = outputs[1]    
      
      outputs = outputs.view(-1, num_classes)  
      y_batch = y_batch.view(-1)  
      loss = criterion(outputs, y_batch)
      testing_loss += loss.item()

      with torch.no_grad():
        _, predicted = torch.max(outputs, dim=1)
        test_total += y_batch.numel()
        test_correct += (predicted == y_batch).sum().item()

        batch_sz, seq_l = X_batch.shape             #get batch size and sequence length
        preds=predicted.view(batch_sz, seq_l)       #reshape predicted to match batch size and sequence length
        y_batch_seq=y_batch.view(batch_sz, seq_l)   #reshape y_batch to match
        
        #check wrong predictions at each epoch
        for i in range(batch_sz):  
            pred_seq=preds[i].tolist()
            label_seq=y_batch_seq[i].tolist()
            in_seq = X_batch[i].tolist()

            if pred_seq != label_seq:
              entry = {'epoch': epoch + 1, 'tokens': in_seq, 'label': label_seq, 'predicted': pred_seq}
              entry_key = (epoch + 1, tuple(in_seq), tuple(label_seq), tuple(pred_seq))  #create a unique key for entry to add to wrong_set
              if entry_key not in wrong_set:
                wrong.append(entry)
                wrong_set.add(entry_key)

    avg_test_loss = testing_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    test_accs.append(100 * test_correct / test_total)
    
    
    if print_cm == True:
      if (epoch + 1) % 10 == 0:    #currently printing every 10 epochs
        plot_confusion_matrix(model, test_loader, num_classes=num_classes)

    cm = get_confusion_matrix(model, test_loader, num_classes=num_classes)
    confusion_matrices.append(cm)

    #write data to a file (change path as needed); manually clear out.txt after saving a copy
    with open("/projects/expmmllab/outm.txt", "a") as f: 
      if validation == True:
        data = f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Train Accuracy: {100 * train_correct / train_total:.4f}%, Validation Accuracy: {100 * val_correct / val_total:.4f}%, Test Accuracy: {100 * test_correct / test_total:.4f}%'
        f.write(data + "\n")
      else:
        data = f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Train Accuracy: {100 * train_correct / train_total:.4f}%, Test Accuracy: {100 * test_correct / test_total:.4f}%'
        f.write(data + "\n")
    print(data)

    end = time.time()  #timer end

  elapsed = end - start
  print(f'Process completed in {elapsed:.4f} seconds.')

  log_graph()
  lin_graph()
  acc_graph()
  matrix_slider(confusion_matrices, class_labels)

  if save_wrong_preds:
    filename = input("Name wrong file (no slashes): ").strip().lower()
    save_wrong(filename)


"""Request gpus"""
#srun --gres=gpu:1 --time=08:00:00 --pty bash             #to request time
#nvidia-smi                                               #for more details
#python /CoxeterArtinProject/lstm/multiclass_LSTM.py      #to run file
