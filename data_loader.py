import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from pyro.infer import MCMC, NUTS, Predictive, SVI, TraceMeanField_ELBO



class NashAllDataset(DataLoader):
    "Nash dataset"

    def __init__(self, split):
        self.X, self.Y = self.load_data(split)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def load_data(self, split):
        if split == 'train':
            print(os.getcwd())
            X = np.load('/home/cdsw/data/x_train_all.npy')
            y = np.load('/home/cdsw/data/y_train_all.npy')
            # y = to_categorical(y)
        elif split == 'val':
            X = np.load('/home/cdsw/data/x_val_all.npy')
            y = np.load('/home/cdsw/data/y_val_all.npy')
            # y = to_categorical(y)
        else:
            X = np.load('/home/cdsw/data/x_test_all.npy')
            y = np.load('/home/cdsw/data/y_test_all.npy')
            # y = to_categorical(y)
        return torch.from_numpy(X).type(torch.FloatTensor),torch.from_numpy(y).type(torch.LongTensor)

class ALZAllDataset(DataLoader):
    "Nash dataset"

    def __init__(self, split):
        self.X, self.Y = self.load_data(split)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def load_data(self, split):
        if split == 'train':
            print(os.getcwd())
            X = np.load('/home/cdsw/data/ALZ/x_train_fix_leakage_ratio.npy')
            y = np.load('/home/cdsw/data/ALZ/y_train_fix_leakage_ratio.npy')
            # y = to_categorical(y)
        elif split == 'val':
            X = np.load('/home/cdsw/data/ALZ/x_val_fix_leakage_ratio.npy')
            y = np.load('/home/cdsw/data/ALZ/y_val_fix_leakage_ratio.npy')
            # y = to_categorical(y)
        else:
            X = np.load('/home/cdsw/data/ALZ/x_test_fix_leakage_ratio.npy')
            y = np.load('/home/cdsw/data/ALZ/y_test_fix_leakage_ratio.npy')
            # y = to_categorical(y)
        return torch.from_numpy(X).type(torch.FloatTensor),torch.from_numpy(y).type(torch.LongTensor)

class DebugDataset(DataLoader):
    "Nash dataset"

    def __init__(self, split):
        self.X, self.Y = self.load_data(split)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def load_data(self, split):
        if split == 'train':
            X = np.load('./data/ALZ/x_train_debug.npy')
            y = np.load('./data/ALZ/y_train_debug.npy')
            # y = to_categorical(y)
        elif split == 'val':
            X = np.load('./data/ALZ/x_val_debug.npy')
            y = np.load('./data/ALZ/y_val_debug.npy')
            # y = to_categorical(y)

        else:
            X = np.load('./data/ALZ/x_test_debug.npy')
            y = np.load('./data/ALZ/y_test_debug.npy')
            # y = to_categorical(y)


        return torch.from_numpy(X).type(torch.LongTensor),torch.from_numpy(y).type(torch.LongTensor)



def fetch_dataloaders_nash_all_features(splits, batch_size=1):
    """
    Args:
        splits (list): A list of strings containing train/val/test.
        batch size
    """
    dataloaders = {}
    for split in splits:
        dataset = NashAllDataset(split)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloaders[split] = dataloader
    return dataloaders


def fetch_dataloaders_ALZ_all_features(splits, batch_size=1):
    """
    Args:
        splits (list): A list of strings containing train/val/test.
        batch size
    """
    dataloaders = {}
    for split in splits:
        dataset = ALZAllDataset(split)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloaders[split] = dataloader
    return dataloaders


def fetch_dataloaders_debug(splits, batch_size=1):
    """
    Args:
        splits (list): A list of strings containing train/val/test.
        batch size
    """
    dataloaders = {}
    for split in splits:
        dataset = DebugDataset(split)
        # if 'test' in split:
        #     dataloader = DataLoader(dataset, batch_size=test_batch_size, shuffle=True)
        # else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        dataloaders[split] = dataloader
    return dataloaders






def test():

    dataloaders = fetch_dataloaders_ALZ_all_features(['train','test','val'],64)


    # dataloader = DataLoader(dataset, shuffle=True)
    for X, y in dataloaders['train'].dataset:
        print(X.shape)
        print(y.shape)
        print(X.dtype)
        print(y.dtype)
        break


if __name__ == '__main__':
    test()