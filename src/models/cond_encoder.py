# src/models/cond_encoder.py
import torch, torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, depth, act="gelu", out_dim=128):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.GELU() if act=="gelu" else nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)