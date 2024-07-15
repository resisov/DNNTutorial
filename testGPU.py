# -*- coding: utf-8 -*-
import torch

def check_mps_avail():
    if torch.backends.mps.is_available():
        print("MPS is available")
    else:
        print("MPS is not available")
    return torch.backends.mps.is_available()

def check_gpu_avail():
    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")
    return torch.cuda.is_available()