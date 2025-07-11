import torch
import torch.nn as nn   #neural networks functions
import torch.optim as optim   #optimizers
from torch.utils.data import DataLoader, Dataset, random_split

import random
import numpy as np
import pandas as pd
import ast
import time  #for simple timer
import matplotlib.pyplot as plt  #data visualization
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def set_seed(seed):
  seed = 42   #convention
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
set_seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

#set device to GPU if available and use CPU otherwise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

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

    self.sequence_length = max(len(seq) for seq in self.data)  #read seq length from padded data

    #find max number of tokens to determine vocab_size
    all_tokens = [token for seq in self.data for token in seq]
    self.vocab_size = max(all_tokens) + 1

  def __len__(self):
    return len(self.data)  #returns the total number of samples in the dataset

  def __getitem__(self, index):
    words = torch.tensor(self.data[index], dtype=torch.long)
    labels = torch.tensor(self.labels[index], dtype=torch.long)
    return words, labels #return word tensor and label tensor

"""Define multi-class classification model"""
class MultiClassLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
      super().__init__()
      self.hidden_size = hidden_size
      self.num_layers = num_layers
      self.embedding = nn.Embedding(vocab_size, embedding_dim)   #embedding layer
      self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
      self.fc = nn.Linear(hidden_size, num_classes) #fully connected layer for multiclass classification

    def forward(self, x):
      embedded = self.embedding(x) #passing through embedding layer
      batch_size = embedded.size(0) #get batch size from embedded tensor

      #initialize hidden and cell states to zero and proper dimensions
      #h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(embedded.device)
      #c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(embedded.device)

      out, _ = self.lstm(embedded)    #shape: (batch_size, seq_len, hidden_size)
      out = self.fc(out)             #shape: (batch_size, seq_len, num_classes)

      return out     #raw logits for cross entropy loss


"""Load datasets"""
#training dataset
train_set = WordDataset(data_dir='/projects/expmmllab/LSTMcx/1ktrainm.csv') #check paths every time; if a syntax error is raised, make sure paths contain backslashes, NOT forward slashes
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

#testing datasets
testing_sets = WordDataset(data_dir='/projects/expmmllab/LSTMcx/1ktestm.csv')
test_loader = DataLoader(testing_sets, batch_size=64, shuffle=True)

#validation toggle: True for on, False for off
validation = True

#split testing dataset into validation and test sets
if validation == True:
  val_size = int(len(testing_sets) * 0.2)   #20% for validation, 80% for testing
  test_size = len(testing_sets) - val_size
  val_set, test_set = random_split(testing_sets, [val_size, test_size]) 
  val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
  test_loader = DataLoader(test_set, batch_size=64, shuffle=False)


"""Model parameters"""
vocab_size = train_set.vocab_size   #build vocab size on training data
embedding_dim = 4                   #token embedding dimension
hidden_size = 16                    #size of LSTM hidden state
num_layers = 1                      #number of LSTM layers
num_classes = 6                     #number of label classes
num_epochs = 10

model = MultiClassLSTM(vocab_size, embedding_dim, hidden_size, num_layers, num_classes)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0) 
#optimizer = optim.AdamW(model.parameters(), lr=0.001)


"""Training loop"""
#loss storage
if validation == True:
  train_losses, val_losses, test_losses = [],[],[]
  train_accs, val_accs, test_accs = [],[],[]
else:
  train_losses, test_losses = [],[]
  train_accs, test_accs = [],[]

start = time.time() 

for epoch in range(num_epochs):
    train_correct = 0
    train_total = 0
    model.train()
    training_loss = 0.0    #running loss

    #training
    for batch_item in train_loader:
        X_batch = batch_item[0]    #input token seq
        y_batch = batch_item[1]    #label seq

        outputs = model(X_batch)                    #(batch_size, seq_len, num_classes)
        outputs = outputs.view(-1, num_classes)     #flatten tokens
        y_batch = y_batch.view(-1)                  #flatten tokens

        loss = criterion(outputs, y_batch)

        loss.backward()    #runs backpropagation
        optimizer.step()
        optimizer.zero_grad()

        training_loss += loss.item()

        with torch.no_grad():
          _, predicted = torch.max(outputs, dim=1)   #max over classes dim
          train_total += y_batch.numel()             #total tokens (batch_size * seq_len)
          train_correct += (predicted == y_batch).sum().item()

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
      outputs = outputs.view(-1, num_classes) 
      y_batch = y_batch.view(-1) 

      loss = criterion(outputs, y_batch)
      testing_loss += loss.item()

      with torch.no_grad():
        _, predicted = torch.max(outputs, dim=1)
        test_total += y_batch.numel()
        test_correct += (predicted == y_batch).sum().item()

    avg_test_loss = testing_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    test_accs.append(100 * test_correct / test_total)

    #write data to a file (change path as needed); manually clear out.txt after saving a copy
    with open("/projects/expmmllab/LSTMcx/out.txt", "a") as f: 
      if validation == True:
        data = f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Train Accuracy: {100 * train_correct / train_total:.4f}%, Validation Accuracy: {100 * val_correct / val_total:.4f}%, Test Accuracy: {100 * test_correct / test_total:.4f}%'
      else:
        data = f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Train Accuracy: {100 * train_correct / train_total:.4f}%, Test Accuracy: {100 * test_correct / test_total:.4f}%'
        f.write(data + "\n")
    print(data)

    end = time.time()  #timer end

elapsed = end - start
print(f'Process completed in {elapsed:.4f} seconds.')


"""Graphing functions"""
def log_graph():
  plt.figure(figsize=(10,5))
  plt.title("Training, Validation, and Testing Loss")
  plt.plot(train_losses,label="Training Loss")
  if validation == True:
    plt.plot(val_losses,label="Validation Loss")
  plt.plot(test_losses,label="Testing Loss")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.yscale('log')
  plt.legend()
  plt.savefig("/projects/expmmllab/LSTMcx/logscaleloss.png")
  print("Log scale loss graph saved :)")
  plt.show()

def lin_graph():
  plt.figure(figsize=(10,5))
  plt.title("Training, Validation, and Testing Loss")
  plt.plot(train_losses,label="Training Loss")
  if validation == True:
    plt.plot(val_losses,label="Validation Loss")
  plt.plot(test_losses,label="Testing Loss")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig("/projects/expmmllab/LSTMcx/linscaleloss.png")
  print("Linear scale loss graph saved :)")
  plt.show()

def acc_graph():
  plt.figure(figsize=(10,5))
  plt.title("Training, Validation, and Testing Accuracy")
  plt.plot(train_accs,label="Training Accuracy")
  if validation == True:
    plt.plot(val_accs,label="Validation Accuracy")
  plt.plot(test_accs,label="Testing Accuracy")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy (%)")
  plt.legend()
  plt.savefig("/projects/expmmllab/LSTMcx/accuracy.png")
  print("Accuracy graph saved :)")
  plt.show()

def plot_confusion_matrix(model, dataloader, num_classes):
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            mask = (y != -100)
            all_preds.extend(preds[mask].cpu().tolist())
            all_targets.extend(y[mask].cpu().tolist())

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["e", "1", "2", "12", "21", "121"])
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix (Token-level)")
    plt.tight_layout()
    plt.savefig("/projects/expmmllab/LSTMcx/confusion_matrix.png")
    print("Confusion matrix saved :)")

log_graph()
lin_graph()
acc_graph()
plot_confusion_matrix(model, test_loader, num_classes)


"""Request gpus"""
#srun --gres=gpu:1 --time=02:00:00 --pty bash   
#nvidia-smi
#python /projects/expmmllab/LSTMcx/multiclass-lstm.py
