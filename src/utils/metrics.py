# src/utils/metrics.py
import torch
import numpy as np

def l2_mean(a, b):
    return ((a - b) ** 2).sum(dim=-1).sqrt().mean().item()

try:
    import open3d as o3d
except Exception:
    o3d = None

def chamfer(P, Q):
    if o3d is None:
        return np.nan
    pcd1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Q))
    kdt1 = o3d.geometry.KDTreeFlann(pcd1)
    kdt2 = o3d.geometry.KDTreeFlann(pcd2)
    def nn_sum(a, b, kdt_b):
        s = 0.0
        for x in a:
            _, idx, dist2 = kdt_b.search_knn_vector_3d(x, 1)
            s += dist2[0]
        return s / len(a)
    return nn_sum(P, Q, kdt2) + nn_sum(Q, P, kdt1)