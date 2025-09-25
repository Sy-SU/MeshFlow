# src/datasets/transforms.py
import numpy as np

def normalize_vertices(V, center=True, scale=True):
    V = V.copy()
    if center:
        V -= V.mean(axis=0, keepdims=True)
    if scale:
        mx = np.abs(V).max()
        if mx > 0:
            V /= mx
    return V