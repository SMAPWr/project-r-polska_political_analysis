import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        hidden_size = 128
        output_size = 5
        input_size = input_size
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.dense2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.dropout(x)
        d1 = self.dense1(x)
        x = self.activation(d1)
        x = self.dense2(x)
        return x, d1

class PoliticalTextsEncoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        hidden_size = 128
        output_size = 64
        input_size = input_size
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.dense2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.dropout(x)
        d1 = self.dense1(x)
        x = self.activation(d1)
        x = self.dense2(x)
        nn.functional.normalize(x, p=2, dim=1)
        return x

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(64, 64)
        self.activation = nn.ReLU()
        self.dense2 = nn.Linear(64, 2)
        self.dense3 = nn.Linear(2, 64)
        self.activation = nn.ReLU()
        self.dense4 = nn.Linear(64, 64)

    def forward(self, x):
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        middle = x
        x = self.dense3(x)
        x = self.activation(x)
        x = self.dense4(x)        
        return x, middle

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y, dtype=np.int64)
        assert self.X.shape[0]==self.y.shape[0]

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]