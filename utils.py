import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler


class CrickDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        dataI, targetI = self.X[idx, :], self.y[idx]
        Xi = torch.from_numpy(dataI.astype(np.int32)).unsqueeze(-1)
        Xv = torch.from_numpy(np.ones_like(dataI))
        return Xi, Xv, targetI
