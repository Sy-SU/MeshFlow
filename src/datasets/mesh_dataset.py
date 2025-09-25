# src/datasets/mesh_dataset.py
import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
from tqdm import tqdm
import psutil, gc, time

def _mb(x):  # 简易内存格式化
    return f"{x/1024/1024:.1f} MB"

def _mem_usage():
    p = psutil.Process()
    return p.memory_info().rss

def sort_zyx_one(tri: np.ndarray):
    """
    tri: (3,3) float32, rows are three vertices [x,y,z]
    Return v1,v2,v3 sorted by z desc, then y desc, then x desc.
    """
    # keys: (-z, -y, -x) -> ascending sort equals desired desc on (z,y,x)
    keys = np.stack([-tri[:, 2], -tri[:, 1], -tri[:, 0]], axis=1)  # (3,3)
    # lexsort uses last key as primary,故顺序传入 (x_key, y_key, z_key)
    idx = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    v1, v2, v3 = tri[idx[0]], tri[idx[1]], tri[idx[2]]
    return v1, v2, v3


def load_trimesh_any(path: str) -> trimesh.Trimesh | None:
    """
    Robustly load a mesh from path. If it's a Scene, concatenate geometries.
    Returns a Trimesh or None if failed/empty.
    """
    obj = trimesh.load(path, process=False)
    if isinstance(obj, trimesh.Trimesh):
        mesh = obj
    elif isinstance(obj, trimesh.Scene):
        # concatenate all geometries in the scene
        geos = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geos:
            return None
        mesh = trimesh.util.concatenate(geos)
    else:
        return None

    if mesh.vertices is None or mesh.faces is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None
    return mesh


class TrianglePairs(Dataset):
    """
    Build triangle samples from ShapeNetCore.v2 meshes or from a precomputed NPZ.
    For each face (triangle), sort its three vertices by (z,y,x) desc -> (v1,v2,v3).
    samples_v2: list of (v1, v2)
    samples_v3: list of ((v1, v2), v3)
    """
    def __init__(self, data_root_or_npz, split="train", backend="trimesh",
                 max_meshes=None, shuffle_tris=True, heads=("v2","v3"),
                 debug=False, debug_limit_tris=None, debug_log_every=1000):
        """
        新增参数：
        - debug: 开启后显示 tqdm 进度条、异常打印、内存监控
        - debug_limit_tris: 仅收集前 N 个三角形（快速冒烟）
        - debug_log_every: 每收集多少个三角形打印一次小结
        """
        self.samples_v2, self.samples_v3 = [], []
        use_v2, use_v3 = "v2" in heads, "v3" in heads

        t0 = time.time()
        mem0 = _mem_usage()

        if backend == "trimesh":
            pat = os.path.join(data_root_or_npz, "**", "models", "model_normalized.obj")
            mesh_paths = sorted(glob.glob(pat, recursive=True))
            if max_meshes is not None:
                mesh_paths = mesh_paths[:max_meshes]

            it = mesh_paths
            if debug:
                it = tqdm(mesh_paths, desc=f"[{split}] loading meshes")

            tri_cnt = 0
            bad_mesh, ok_mesh = 0, 0

            for p in it:
                try:
                    mesh = load_trimesh_any(p)
                    if mesh is None:
                        bad_mesh += 1
                        if debug:
                            tqdm.write(f"[WARN] skip empty/invalid: {p}")
                        continue
                    V = np.asarray(mesh.vertices, dtype=np.float32)
                    F = np.asarray(mesh.faces, dtype=np.int64)
                    tris = V[F]  # (T,3,3)

                    # 内层进度条（可选）：大模型时会太刷屏，默认关；需要就放开下面两行
                    inner_iter = tris if not debug else tris
                    for i, t in enumerate(inner_iter):
                        v1, v2, v3 = sort_zyx_one(t)
                        if use_v2:
                            self.samples_v2.append((v1, v2))
                        if use_v3:
                            self.samples_v3.append(((v1, v2), v3))
                        tri_cnt += 1

                        # 调试：限制三角形上限
                        if debug_limit_tris and tri_cnt >= debug_limit_tris:
                            break

                        # 定期打印内存与样本数
                        if debug and (tri_cnt % debug_log_every == 0):
                            cur = _mem_usage()
                            tqdm.write(f"[{split}] tris={tri_cnt:,} | v2={len(self.samples_v2):,} v3={len(self.samples_v3):,} | RSS={_mb(cur)}")

                    ok_mesh += 1
                    if debug_limit_tris and tri_cnt >= debug_limit_tris:
                        break

                except Exception as e:
                    bad_mesh += 1
                    if debug:
                        tqdm.write(f"[ERROR] path={p} err={repr(e)}")
                    continue

            if debug:
                mem1 = _mem_usage()
                tqdm.write(f"[{split}] mesh ok/bad = {ok_mesh}/{bad_mesh} | tris={tri_cnt:,} | elapsed={time.time()-t0:.1f}s | RSS+={_mb(mem1-mem0)}")

        else:
            if debug:
                tqdm.write(f"[{split}] loading NPZ: {data_root_or_npz}")
            data = np.load(data_root_or_npz, allow_pickle=True, mmap_mode=None)
            v1 = data[f"{split}_v1"].astype(np.float32)
            v2 = data[f"{split}_v2"].astype(np.float32)
            v3 = data[f"{split}_v3"].astype(np.float32)

            N = len(v1)
            rng = range(N)
            if debug:
                rng = tqdm(range(N), desc=f"[{split}] building pairs (npz)")

            for i in rng:
                a, b, c = v1[i], v2[i], v3[i]
                if use_v2:
                    self.samples_v2.append((a, b))
                if use_v3:
                    self.samples_v3.append(((a, b), c))
                if debug_limit_tris and (i+1) >= debug_limit_tris:
                    break

            if debug:
                mem1 = _mem_usage()
                tqdm.write(f"[{split}] npz tris={len(self.samples_v2) or len(self.samples_v3):,} | RSS+={_mb(mem1-mem0)}")

        if shuffle_tris:
            rng = np.random.default_rng(42)
            if self.samples_v2:
                rng.shuffle(self.samples_v2)
            if self.samples_v3:
                rng.shuffle(self.samples_v3)

        # 主动触发一次 GC（大文件遍历后可回收临时对象）
        if debug:
            gc.collect()
            tqdm.write(f"[{split}] ready. v2={len(self.samples_v2):,} v3={len(self.samples_v3):,} | RSS={_mb(_mem_usage())}")

    def __len__(self):
        # allow independent sampling; choose the longer one as length
        return max(len(self.samples_v2), len(self.samples_v3))

    def __getitem__(self, idx):
        i2 = idx % len(self.samples_v2)
        i3 = idx % len(self.samples_v3)
        (v1, v2) = self.samples_v2[i2]
        ((a, b), v3) = self.samples_v3[i3]
        return (
            torch.from_numpy(np.asarray(v1)),
            torch.from_numpy(np.asarray(v2)),
            torch.from_numpy(np.asarray(v3)),
        )
