# src/utils/geometry.py
import numpy as np
from sklearn.decomposition import PCA

def sort_triangle_vertices_zyx_with_idx(V, f):
    """
    V: (N,3) 全部顶点
    f: (3,) 该三角形的顶点索引
    return:
      V_sorted: (3,3) 排序后的坐标
      idx_sorted: (3,) 对应原始顶点索引（来自 f）
    """
    tri = V[f]  # (3,3)
    order = np.lexsort((tri[:,0], tri[:,1], tri[:,2]))  # keys=(x,y,z)
    return tri[order], f[order]

def normalize_unit_box(V):
    vmin = V.min(axis=0); vmax = V.max(axis=0)
    diag = (vmax - vmin).max()
    Vn = (V - (vmin + vmax)/2.0) / (diag + 1e-12)
    return Vn.astype(np.float32), (vmin, vmax)

def safe_local_frame(points: np.ndarray) -> np.ndarray:
    """
    对退化邻域更稳健的 PCA 局部系：若方差太小或秩不足，直接返回 I。
    points: (K,3) 相对坐标（均值~0）
    """
    K = points.shape[0]
    if K < 3:
        return np.eye(3, dtype=np.float32)
    # 零方差 / 全相同点检查
    var = points.var(axis=0).sum()
    if not np.isfinite(var) or var < 1e-12:
        return np.eye(3, dtype=np.float32)
    try:
        pca = PCA(n_components=3)
        pca.fit(points)
        R = pca.components_.T.astype(np.float32)
        if not np.isfinite(R).all() or abs(np.linalg.det(R)) < 1e-8:
            return np.eye(3, dtype=np.float32)
        # 保证右手系
        if np.linalg.det(R) < 0:
            R[:, -1] *= -1
        return R
    except Exception:
        return np.eye(3, dtype=np.float32)
