import torch
import torch.nn as nn   #neural networks functions
import torch.optim as optim   #optimizers
from torch.utils.data import DataLoader, Dataset, random_split

import numpy as np
import pandas as pd
import ast
import time  #for simple timer
import matplotlib.pyplot as plt  #data visualization

"""Prep dataset"""

class WordDataset(Dataset):
  def __init__(self, data_dir):
    df = pd.read_csv(data_dir)
    df['tokens'] = df['tokens'].apply(ast.literal_eval)   #turns string representations of list under column 'tokens' into actual lists

    self.data = []
    for seq in df['tokens']:
      int_seq = [int(token) for token in seq]    #turns list of strings into integer sequence (list of ints)
      self.data.append(int_seq)

    self.labels = df['label'].tolist()
    self.sequence_length = max(len(seq) for seq in self.data)  #read seq length from padded data

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
      self.embedding = nn.Embedding(vocab_size, embedding_dim)   #embedding layer
      self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
      self.fc = nn.Linear(hidden_size, 1) #fully connected layer for binary classification

    def forward(self, x):
      embedded = self.embedding(x) #passing through embedding layer
      batch_size = embedded.size(0) #get batch size from embedded tensor

      #initialize hidden and cell states to zero and proper dimensions
      #h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(embedded.device)
      #c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(embedded.device)

      out, _ = self.lstm(embedded)    #forward propagate LSTM
      out = self.fc(out[:, -1, :])              #take output of the last time step

      return torch.sigmoid(out)     #apply sigmoid to output for binary classification probability (between 0 and 1)

#training dataset
train_set = WordDataset(data_dir='C:/Users/xiongce/Downloads/CoxeterArtinProject/generated_datasets/1_2025-06-25-train.csv') #need to upload files and check paths every time; if a syntax error is raised, make sure paths contain backslashes, NOT forward slashes
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

#testing and validation datasets
testing_sets = WordDataset(data_dir='C:/Users/xiongce/Downloads/CoxeterArtinProject/generated_datasets/1_2025-06-25-test.csv')

#split testing dataset into validation and test sets
val_size = int(len(testing_sets) * 0.2)   #20% for validation, 80% for testing
test_size = len(testing_sets) - val_size
val_set, test_set = random_split(testing_sets, [val_size, test_size]) 
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
#test_loader = DataLoader(testing_sets, batch_size=100000, shuffle=True)

#model parameters
vocab_size = train_set.vocab_size   #build vocab size on training data
embedding_dim = 128   #token embedding dimension
hidden_size = 256    #size of LSTM hidden state
num_layers = 1      #number of LSTM layers

model = TrivialWordLSTM(vocab_size, embedding_dim, hidden_size, num_layers)

criterion = nn.BCELoss()    #binary cross entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) 
#optimizer = optim.AdamW(model.parameters(), lr=0.001)

"""Training loop"""

num_epochs = 1000

train_losses, val_losses, test_losses = [],[],[]
train_accs, val_accs, test_accs = [],[],[]
#train_losses, test_losses = [],[]
#train_accs, test_accs = [],[]

start = time.time()  #timer start

for epoch in range(num_epochs):
    train_correct = 0
    train_total = 0
    model.train()
    training_loss = 0.0    #running loss

    #training
    for batch_item in train_loader:
        X_batch = batch_item[0]    #input token seq
        y_batch = batch_item[1]    #binary labels
        y_batch = y_batch.unsqueeze(-1)  #match output shape for BCELoss (batch_size, 1) by adding a dimension at the -1 position

        outputs = model(X_batch)   #predicted probabilities
        loss = criterion(outputs, y_batch)

        loss.backward()    #runs backpropagation
        optimizer.step()
        optimizer.zero_grad()

        training_loss += loss.item()

        with torch.no_grad():
          predicted = (outputs > 0.5).float()
          train_total += y_batch.size(0)
          train_correct += (predicted == y_batch).sum().item()

    avg_train_loss = training_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accs.append(100 * train_correct / train_total)

    #validation
    val_correct = 0
    val_total = 0
    model.eval()
    validation_loss = 0.0
  
    for batch_item in val_loader:
        X_batch = batch_item[0]
        y_batch = batch_item[1]

        y_batch = y_batch.unsqueeze(-1) 
        outputs = model(X_batch)
        
        loss = criterion(outputs, y_batch)
        validation_loss += loss.item()

        with torch.no_grad():
          predicted = (outputs > 0.5).float()
          val_total += y_batch.size(0)
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

      y_batch = y_batch.unsqueeze(-1)
      outputs = model(X_batch)
      
      loss = criterion(outputs, y_batch)
      testing_loss += loss.item()

      with torch.no_grad():
          predicted = (outputs > 0.5).float()
          test_total += y_batch.size(0)
          test_correct += (predicted == y_batch).sum().item()

    avg_test_loss = testing_loss / len(test_loader)
    test_losses.append(loss.item())
    test_accs.append(100 * test_correct / test_total)

   with open("C:/Users/xiongce/Downloads/test/out.txt", "a") as f: #write data to a file (specify path); manually clear out.txt after saving a copy (for now)
      data = f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Train Accuracy: {100 * train_correct / train_total:.4f}%, Validation Accuracy: {100 * val_correct / val_total:.4f}%, Test Accuracy: {100 * test_correct / test_total:.4f}%'
      #data = f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Train Accuracy: {100 * train_correct / train_total:.4f}%, Test Accuracy: {100 * test_correct / test_total:.4f}%'
      f.write(data + "\n")
    print(data)
    end = time.time()  #timer end

elapsed = end - start
print(f'Process completed in {elapsed:.4f} seconds.')

#graphing
plt.figure(figsize=(10,5))
plt.title("Training, Validation, and Testing Loss")
plt.plot(train_losses,label="Training Loss")
plt.plot(val_losses,label="Validation Loss")
plt.plot(test_losses,label="Testing Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale('log') #toggle on/off as needed
plt.legend()
plt.savefig('logscaleloss')
print("Log scale loss graph saved :)")
plt.show()

plt.figure(figsize=(10,5))
plt.title("Training, Validation, and Testing Loss")
plt.plot(train_losses,label="Training Loss")
plt.plot(val_losses,label="Validation Loss")
plt.plot(test_losses,label="Testing Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('linscaleloss')
print("Linear scale loss graph saved :)")
plt.show()

plt.figure(figsize=(10,5))
plt.title("Training, Validation, and Testing Accuracy")
plt.plot(train_accs,label="Training Accuracy")
plt.plot(val_accs,label="Validation Accuracy")
plt.plot(test_accs,label="Testing Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig('accuracy')
print("Accuracy graph saved :)")
plt.show()

#srun --gres=gpu:1 --time=01:30:00 --pty bash   #failsafe for requesting gpus, time
