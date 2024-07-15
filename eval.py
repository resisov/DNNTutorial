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
test_df = myDataset.TestDataset(data_path='test_data.npz')
test_loader = DataLoader(test_df, batch_size=batch_size, shuffle=True)

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

# evaluation
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        y_pred = torch.round(y_pred)
        total += y.size(0)
        correct += (y_pred == y).sum().item()
        loss = criterion(y_pred, y.float())
        print(f'Loss: {loss.item()}', f'Accuracy: {100 * correct / total:.2f}%')

    print(f'Accuracy: {100 * correct / total:.2f}%')

# save result
with open('result.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Accuracy', 'Loss'])
    writer.writerow([100 * correct / total, loss.item()])
    
