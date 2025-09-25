# src/models/flow_core.py
import torch, torch.nn as nn

class VelocityField(nn.Module):
    def __init__(self, in_dim, hidden, depth, act="gelu"):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.GELU() if act=="gelu" else nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, 3)]
        self.net = nn.Sequential(*layers)
    def forward(self, x_t, t, cond):
        # x_t: (B,3), t: (B,1), cond: (B,C)
        h = torch.cat([x_t, t, cond], dim=-1)
        return self.net(h)