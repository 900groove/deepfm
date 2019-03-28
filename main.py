import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import CrickDataset
from model import DeepFM


train = pd.read_csv('./data/processed/train.csv', nrows=10000)
target = train.iloc[:, 0]
train = train.iloc[:, 1:]
f = open('./data/processed/feature_size.pkl', 'rb')
feature_size = pickle.load(f)

dataset = CrickDataset(train.values, target.values)
dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)

model = DeepFM(feature_size).to('cuda')
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
criterion = nn.BCEWithLogitsLoss()

loss_score = []
for _ in range(1):
    for t, (xi, xv, y) in enumerate(dataloader):
        xi = xi.to(device='cuda', dtype=torch.long)
        xv = xv.to(device='cuda', dtype=torch.float)
        y = y.to(device='cuda', dtype=torch.float)

        total = model(xi, xv)
        loss = criterion(total, y)
        loss_score.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 500 == 0:
            print('Iteration %d, loss = %.4f' % (t, loss.item()))

