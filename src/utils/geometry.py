import numpy as np
from sklearn.decomposition import PCA

def sort_triangle_vertices_zyx(V_f):
    """
    V_f: (3, 3) for a triangle's 3 vertices
    Return V_sorted: (3, 3) sorted by (z, y, x) ascending
    """
    order = np.lexsort((V_f[:,0], V_f[:,1], V_f[:,2]))  # keys from last -> first
    return V_f[order]

def normalize_unit_box(V):
    """
    Scale mesh into unit cube ([-0.5, 0.5]^3).
    Return V_norm, bbox (for de-normalization)
    """
    vmin = V.min(axis=0)
    vmax = V.max(axis=0)
    diag = (vmax - vmin).max()
    Vn = (V - (vmin + vmax)/2.0) / (diag + 1e-12)
    return Vn.astype('float32'), (vmin, vmax)

def local_frame(points):
    """
    PCA-based local frame. points: (K,3) centered at anchor (mean ~ 0).
    Return: R (3x3) rotation, where columns are principal directions.
    """
    if points.shape[0] < 3:
        return np.eye(3, dtype=np.float32)
    pca = PCA(n_components=3)
    pca.fit(points)
    R = pca.components_.T.astype('float32')  # shape (3,3)
    # 保证右手系（可选）
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return R
