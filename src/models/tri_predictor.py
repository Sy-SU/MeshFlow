# src/models/tri_predictor.py
import torch, torch.nn as nn
from .cond_encoder import MLP
from .flow_core import VelocityField

class TriFlow(nn.Module):
    def __init__(self, cond_in, cond_hidden, cond_depth, flow_in, flow_hidden, flow_depth, act="gelu"):
        super().__init__()
        self.cond = MLP(cond_in, cond_hidden, cond_depth, act=act, out_dim=128)
        self.field = VelocityField(flow_in, flow_hidden, flow_depth, act=act)

    def forward(self, x_t, t, cond_in):
        c = self.cond(cond_in)
        return self.field(x_t, t, c)

    @staticmethod
    def cfm_loss(model, x0, x1, cond_in, t):
        # straight line x(t) = (1-t)*x0 + t*x1 ; v* = x1 - x0
        x_t = (1.0 - t) * x0 + t * x1
        v_star = x1 - x0
        v_pred = model(x_t, t, cond_in)
        return ((v_pred - v_star)**2).sum(dim=-1).mean()

    @torch.no_grad()
    def sample(self, x0, cond_in, steps=32, method="euler"):
        # integrate v over t in [0,1]
        dt = 1.0 / steps
        x = x0
        for s in range(steps):
            t = torch.full_like(x0[:, :1], (s+0.5)*dt)
            v = self.forward(x, t, cond_in)
            x = x + v * dt
        return x