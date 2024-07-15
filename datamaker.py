# -*- coding: utf-8 -*-
import numpy as np

datatype = np.float32

def make_data(data_size, input_node):
    x_data = np.random.rand(data_size, input_node).astype(datatype)
    y_data = np.random.randint(0, 2, (data_size, 1)).astype(datatype)
    return x_data, y_data

def save_data(data, data_path):
    np.savez(data_path, x_data=data[0], y_data=data[1])

def main():
    datas = ['train_data.npz', 'test_data.npz', 'valid_data.npz']
    data_size = 100000
    input_node = 10
    for data_path in datas:
        data = make_data(data_size, input_node)
        save_data(data, data_path)
        print(f'{data_path} is saved')

if __name__ == '__main__':
    main()