# src/datasets/tri_pairs.py
import torch
from torch.utils.data import Dataset

class TriPairsV2(Dataset):
    def __init__(self, base_dataset):
        self.data = base_dataset.samples_v2
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        v1, v2 = self.data[i]
        return torch.tensor(v1, dtype=torch.float32), torch.tensor(v2, dtype=torch.float32)

class TriPairsV3(Dataset):
    def __init__(self, base_dataset):
        self.data = base_dataset.samples_v3
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        (v1, v2), v3 = self.data[i]
        return torch.tensor(v1, dtype=torch.float32), torch.tensor(v2, dtype=torch.float32), torch.tensor(v3, dtype=torch.float32)