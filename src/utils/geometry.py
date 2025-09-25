# src/utils/geometry.py
import numpy as np

def quantize_vertices(V, tol=1e-8):
    Q = np.round(V / tol).astype(np.int64)
    unique, inv = np.unique(Q, axis=0, return_inverse=True)
    V_unique = unique.astype(np.float64) * tol
    return V_unique, inv