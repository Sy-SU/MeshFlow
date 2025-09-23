import numpy as np

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
