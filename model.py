import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os


class IonNet(nn.Module):
    def __init__(self, n_in=6, activation='ReLU'):
        super(IonNet, self).__init__()
        if activation == 'ReLU':
            self.act = F.relu
        else:
            self.act = torch.sigmoid
        # an affine operation: y = Wx + b
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(n_in, 1024))
        self.linears.append(nn.Linear(1024, 512))
        self.linears.append(nn.Linear(512, 256))
        self.linears.append(nn.Linear(256, 128))
        self.linears.append(nn.Linear(128, 32))  # 5*5 from image dimension
        self.linears.append(nn.Linear(32, 1))  

    def forward(self, x):
        for i in range(len(self.linears)-1):
            x = self.act(self.linears[i](x))
        x = self.linears[-1](x)

        return torch.sigmoid(x)


class IonDataset(Dataset):
    def __init__(self, data):
        self.feat = data[:, :-1]
        self.y = data[:, -1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.feat[idx], dtype=torch.float)
        y = torch.tensor(self.y[idx], dtype=torch.float)
        return x, y.view(-1)


def split_data(data, valid_ratio, randomSeed = None):
    total_size = len(data)
    train_ratio = 1 - valid_ratio
    indices = list(range(total_size))
    print("The random seed is: ", randomSeed)
    np.random.seed(randomSeed)
    np.random.shuffle(indices)
    train_size = int(train_ratio * total_size)
    valid_size = int(valid_ratio * total_size)
    print('Train size: {}, Validation size: {}'.format(
    train_size, valid_size
    ))
    train_idx, valid_idx = indices[:train_size], indices[-valid_size:]
    return data[train_idx], data[valid_idx]

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mae(prediction, target):
    return torch.mean(torch.abs(target - prediction))