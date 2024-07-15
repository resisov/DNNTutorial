# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class myDNN(nn.Module):
    def __init__(self, input_node):
        self.input_node = input_node
        super(myDNN, self).__init__()
        ## neural network layers
        self.fc = nn.Sequential(
            # first layer
            nn.Linear(self.input_node, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            # second layer
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            # third layer
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            # output layer
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        y_pred = self.fc(x)
        return y_pred
