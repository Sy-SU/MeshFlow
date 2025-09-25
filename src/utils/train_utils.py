# src/utils/train_utils.py
import torch
from contextlib import nullcontext

def get_autocast(enabled=True):
    return torch.cuda.amp.autocast() if enabled else nullcontext()

def grad_clip_(params, max_norm=None):
    if max_norm is None or max_norm <= 0:
        return
    torch.nn.utils.clip_grad_norm_(params, max_norm)