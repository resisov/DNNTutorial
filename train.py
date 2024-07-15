# -*- coding: utf-8 -*-
import os, sys
import argparse
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import myDataset # type: ignore
import myModel # type: ignore
import testGPU # type: ignore

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--optimizer', type=str, default='adam')
args = parser.parse_args()

optimizer = args.optimizer
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr

# check GPU availability
mps = testGPU.check_mps_avail()
gpu = testGPU.check_gpu_avail()
if mps and gpu:
    device = torch.device('cuda')
elif not mps and gpu:
    device = torch.device('cuda')
elif not gpu and mps:
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f'device: {device}')

# load dataset
train_df = myDataset.TrainDataset(data_path='train_data.npz')
train_loader = DataLoader(train_df, batch_size=batch_size, shuffle=True)
test_df = myDataset.TestDataset(data_path='test_data.npz')
test_loader = DataLoader(test_df, batch_size=batch_size, shuffle=True)
valid_df = myDataset.ValidDataset(data_path='valid_data.npz')
valid_loader = DataLoader(valid_df, batch_size=batch_size, shuffle=True)

# load model
input_node = 10
model = myModel.myDNN(input_node).to(device)
print(model)

# loss function and optimizer
criterion = nn.BCELoss()
if optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr)
else:
    optimizer = optim.Adam(model.parameters(), lr=lr)

# training
for epoch in range(epochs):
    model.train()
    for i, (x, y) in enumerate(train_loader):
        x, y = Variable(x).to(device), Variable(y).to(device)
        optimizer.zero_grad()
        y_pred = model(x.float())
        loss = criterion(y_pred, y.float())
        loss.backward()
        optimizer.step()
    print(f'epoch: {epoch}, loss: {loss.item()}')

# testing
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
        x, y = Variable(x).to(device), Variable(y).to(device)
        y_pred = model(x.float())
        y_pred = torch.round(y_pred)
        total += y.size(0)
        correct += (y_pred == y).sum().item()
print(f'test accuracy: {correct / total}')

# validation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for i, (x, y) in enumerate(valid_loader):
        x, y = Variable(x).to(device), Variable(y).to(device)
        y_pred = model(x.float())
        y_pred = torch.round(y_pred)
        total += y.size(0)
        correct += (y_pred == y).sum().item()
print(f'validation accuracy: {correct / total}')

# save model
torch.save(model.state_dict(), 'model.pth')