import torch
import torch.nn as nn   #neural networks functions
import torch.optim as optim   #optimizers
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import ast
import time  #for simple timer
import matplotlib.pyplot as plt  #data visualization

"""Prep dataset"""

class WordDataset(Dataset):
  def __init__(self, data_dir, sequence_length=26):  #update with dataset for the time being
    df = pd.read_csv(data_dir)
    df['tokens'] = df['tokens'].apply(ast.literal_eval)   #turns string representations of list under column 'tokens' into actual lists

    self.data = []
    for seq in df['tokens']:
      int_seq = [int(token) for token in seq]    #turns list of strings into integer sequence (list of ints)
      self.data.append(int_seq)

    self.labels = df['label'].tolist()
    self.sequence_length = sequence_length

    #find max number of tokens to determine vocab_size
    all_tokens = [token for seq in self.data for token in seq]
    self.vocab_size = max(all_tokens) + 1

  def __len__(self):
    return len(self.data)  #returns the total number of samples in the dataset

  def __getitem__(self, index):
    words = torch.tensor(self.data[index], dtype=torch.long)
    labels = torch.tensor(self.labels[index], dtype=torch.float32)
    return words, labels #return word tensor and label tensor


"""Define binary classification model"""

class TrivialWordLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
      super().__init__()
      self.hidden_size = hidden_size
      self.num_layers = num_layers
      self.embedding_dim = embedding_dim

      self.embedding = nn.Embedding(vocab_size, embedding_dim)   #embedding layer
      self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
      self.fc = nn.Linear(hidden_size, 1) #fully connected layer for binary classification

    def forward(self, x):
      embedded = self.embedding(x) #passing through embedding layer
      batch_size = embedded.size(0) #get batch size from embedded tensor

      #initialize hidden and cell states to zero and proper dimensions
      h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(embedded.device)
      c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(embedded.device)

      out, _ = self.lstm(embedded, (h0, c0))    #forward propagate LSTM
      out = self.fc(out[:, -1, :])              #take output of the last time step

      return torch.sigmoid(out)     #apply sigmoid to output for binary classification probability (between 0 and 1)

#training dataset
train_set = WordDataset(data_dir='C:/Users/xiongce/CoxeterArtinProject/generated_datasets/1_2025-06-25-train.csv') #need to upload files and check paths every time; if a syntax error is raised, make sure paths contain backslashes, NOT forward slashes
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

#validation dataset
val_set = WordDataset(data_dir='C:/Users/xiongce/CoxeterArtinProject/generated_datasets/1_2025-06-25-test.csv')
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

#model parameters
vocab_size = train_set.vocab_size   #build vocab size on training data
embedding_dim = 128   #token embedding dimension
hidden_size = 256    #size of LSTM hidden state
num_layers = 1      #number of LSTM layers

model = TrivialWordLSTM(vocab_size, embedding_dim, hidden_size, num_layers)

criterion = nn.BCELoss()    #binary cross entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)   #takes in model parameters, learning rate, weight decay
#optimizer = optim.AdamW(model.parameters(), lr=0.001)

"""Training loop"""

num_epochs = 10000
sequence_length = 26  #update with dataset for the time being

train_losses, val_losses = [],[]
val_accuracies = []  #list to store accuracies

start = time.time()  #timer start

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_item in train_loader:
        X_batch = batch_item[0]
        y_batch = batch_item[1]
        y_batch = y_batch.unsqueeze(-1)  #match output shape for BCELoss (batch_size, 1) by adding a dimension at the -1 position

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()    #runs backpropagation
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():   #zeros out gradients
      for batch_item in val_loader:
        X_batch = batch_item[0]
        y_batch = batch_item[1]

        y_batch = y_batch.unsqueeze(-1)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        val_running_loss += loss.item()

    avg_val_loss = val_running_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    #accuracy calculations
    correct = 0
    total = 0
    with torch.no_grad():
      for data in val_loader:
        tokens, labels = data
        outputs = model(tokens)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels.unsqueeze(-1)).sum().item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_val_loss:.4f}, Accuracy: {100 * correct / total:.4f}%')
    end = time.time()  #timer end

    accuracy = 100 * correct / total
    val_accuracies.append(accuracy)

elapsed = end - start
print(f'Process completed in {elapsed:.4f} seconds.')

#graphing
plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(train_losses,label="Training Loss")
plt.plot(val_losses,label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale('log') #toggle on/off as needed
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.title("Accuracy")
plt.plot(val_accuracies)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
