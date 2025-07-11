import torch
import torch.nn as nn   #neural networks functions
import torch.optim as optim   #optimizers
from torch.utils.data import DataLoader, Dataset, random_split

#from lstmcell pytorch documentation
lstm = nn.LSTMCell(10, 20)  # (input_size, hidden_size)
input = torch.randn(2, 3, 10)  # (time_steps, batch, input_size)
hx = torch.randn(3, 20)  # (batch, hidden_size)
cx = torch.randn(3, 20)
output = []
for i in range(input.size()[0]):
    hx, cx = lstm(input[i], (hx, cx))
    output.append(hx)
    print(hx, cx)
output = torch.stack(output, dim=0)
