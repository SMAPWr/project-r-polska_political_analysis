from numpy.core.numeric import indices
from sentimentpl.models import SentimentPLModel
import os
import pandas as pd
import pickle
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from models import RegressionMCDropoutModel, RobertaRegressionModel
from scipy.stats import spearmanr

device = torch.device("cuda")

def datagen(X, y, batch_size):
    indices = np.random.permutation(np.arange(X.shape[0]))
    # indices = np.random.permutation(np.arange(300))
    for i in range(0, X.shape[0], batch_size):
        idx = indices[i:i+batch_size]
        yield X[idx].tolist(), torch.tensor(y[idx], dtype=torch.float32).to(device)


def rmse(y, y_hat):
    """Compute root mean squared error"""
    return np.sqrt(np.mean((y - y_hat)**2))


def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y        

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]



# model = RobertaForSequenceClassification.from_pretrained(
#     "models/politicalBERT", config = PretrainedConfig(num_labels=2))
model = RobertaRegressionModel(device)
for p in model.model.embeddings.parameters():
    p.requires_grad = False

for p in model.model.encoder.parameters():
    p.requires_grad = False

model = model.to(device)

train_ds = pd.read_csv('twitter_data/data/dataset2.csv')
X_train = train_ds['texts'].to_numpy()
y_train = train_ds[['economic', 'worldview']].to_numpy()

test_ds = pd.read_csv('twitter_data/data/validation_data.csv')
X_test = test_ds['tweet'].to_numpy()
y_test = test_ds[['economic', 'worldview']].to_numpy()


criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []
val_maes = []

BEST = float('inf')

BATCH_SIZE = 32

for epoch in range(50):  # loop over the dataset multiple times
    total_loss = 0
    model.train(True)
    t = tqdm(enumerate(datagen(X_train, y_train, BATCH_SIZE)))
    for i, data in t:
        input, target = data
        target = target.cuda()
        optimizer.zero_grad()
        pred = model(input)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.cpu().item()

    train_loss = total_loss/(i+1)
    losses.append(train_loss)

    total_loss = 0
    predictions = []
    targets = []
    model.train(False)
    for i, data in enumerate(datagen(X_test, y_test, BATCH_SIZE)):
        input, target = data
        target = target.cuda()
        pred = model(input)
        # loss = criterion(pred, target)
        predictions.append(pred.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    val_mae = mae(predictions, targets)
    val_maes.append(val_mae)

    # t.set_description(f'loss : {train_loss:.6f}, val_mae: {val_mae:.6f}')
    print(f'epoch: {epoch}, loss : {train_loss:.6f}, val_mae: {val_mae:.6f}')
    print(f'Spearman correlation (economic): {spearmanr(predictions[:, 0], targets[:, 0])}')
    print(f'Spearman correlation (worldview): {spearmanr(predictions[:, 1], targets[:, 1])}')
    if val_mae<BEST:
        BEST = val_mae        
        torch.save(model.state_dict(), f'fitted_bert_epoch_{epoch+1}_mae_{val_mae}')

print('Finished Training')
