# src/datasets/transforms.py
import numpy as np
from scipy.spatial import cKDTree
from ..utils.geometry import safe_local_frame

def precompute_vertex_knn_and_frames(V: np.ndarray, k: int = 24, use_local_frame: bool = True):
    """
    为每个顶点一次性查询 kNN，并可选计算每个顶点的局部坐标系。
    return:
      nbr_idx: (N,k) 每个顶点的邻居索引（包含自己）
      frames:  (N,3,3) 每个顶点的局部旋转矩阵（若 use_local_frame=False 则全 I）
    """
    N = V.shape[0]
    tree = cKDTree(V)
    # query 包含自身最近邻；若不希望包含自身，可取 idx[:,1:k+1]
    d, idx = tree.query(V, k=min(k, N))  # (N,k)
    # 计算每个顶点的局部系
    frames = np.zeros((N, 3, 3), dtype=np.float32)
    for i in range(N):
        rel = V[idx[i]] - V[i]  # (k,3)
        if use_local_frame:
            R = safe_local_frame(rel)
        else:
            R = np.eye(3, dtype=np.float32)
        frames[i] = R
    return idx.astype(np.int64), frames


def build_knn_context(V, anchor_idx, k=24):
    """
    Very simple kNN by brute-force (可后续换为 faiss/ball-tree).
    Return: ctx_idx (k,), ctx_rel (k,3) relative coords to anchor
    """
    anchor = V[anchor_idx]  # (3,)
    d2 = ((V - anchor)**2).sum(axis=1)
    # include self as first; take k nearest
    idx = np.argsort(d2)[:k]
    rel = V[idx] - anchor
    return idx.astype('int64'), rel.astype('float32')

def to_local_frame(rel_coords, R):
    # rel_coords: (k,3), R: (3,3)
    return (rel_coords @ R).astype('float32')
