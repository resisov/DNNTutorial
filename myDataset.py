# -*- coding: utf-8 -*-
import os, sys
import torch
import numpy as np

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = np.load(self.data_path)
        self.x_data = self.data['x_data']
        self.y_data = self.data['y_data']
        self.len = len(self.x_data)
    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        return x, y
    def __len__(self):
        return self.len

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = np.load(self.data_path)
        self.x_data = self.data['x_data']
        self.y_data = self.data['y_data']
        self.len = len(self.x_data)
    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        return x, y
    def __len__(self):
        return self.len

class ValidDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = np.load(self.data_path)
        self.x_data = self.data['x_data']
        self.y_data = self.data['y_data']
        self.len = len(self.x_data)
    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        return x, y
    def __len__(self):
        return self.len

train_df = TrainDataset(data_path='train_data.npz')
train_x = train_df.x_data
train_y = train_df.y_data

test_df = TestDataset(data_path='test_data.npz')
test_x = test_df.x_data
test_y = test_df.y_data

valid_df = ValidDataset(data_path='valid_data.npz')
valid_x = valid_df.x_data
valid_y = valid_df.y_data