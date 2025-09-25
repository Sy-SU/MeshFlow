# src/datasets/mesh_dataset.py
import os, glob, numpy as np, torch
from torch.utils.data import Dataset
import trimesh

class TrianglePairs(Dataset):
    def __init__(self, data_root, split, backend="trimesh", max_meshes=None, shuffle_tris=True):
        self.samples_v2 = []  # each: (v1, v2)
        self.samples_v3 = []  # each: ((v1, v2), v3)

        if backend == "trimesh":
            mesh_paths = sorted(glob.glob(os.path.join(data_root, "**", "models", "model_normalized.obj"), recursive=True))
            if max_meshes is not None:
                mesh_paths = mesh_paths[:max_meshes]
            for p in mesh_paths:
                mesh = trimesh.load(p, process=False)
                V = np.asarray(mesh.vertices, dtype=np.float32)
                F = np.asarray(mesh.faces, dtype=np.int64)
                tris = V[F]  # (T,3,3)
                # sort by (z,y,x)
                order = np.lexsort((tris[...,0], tris[...,1], tris[...,2]))  # wrong: need per-tri sort
                # per-triangle sort
                tris_sorted = np.sort(tris, axis=1, order=None)  # placeholder; implement custom sorter
                for t in tris_sorted:
                    v1, v2, v3 = sort_zyx(t)
                    self.samples_v2.append((v1, v2))
                    self.samples_v3.append(((v1, v2), v3))
        else:
            data = np.load(backend, allow_pickle=True)
            # expect keys: {split}_v1, {split}_v2, {split}_v3
            v1 = data[f"{split}_v1"].astype(np.float32)
            v2 = data[f"{split}_v2"].astype(np.float32)
            v3 = data[f"{split}_v3"].astype(np.float32)
            for a, b, c in zip(v1, v2, v3):
                self.samples_v2.append((a, b))
                self.samples_v3.append(((a, b), c))

        if shuffle_tris:
            rng = np.random.default_rng(42)
            rng.shuffle(self.samples_v2)
            rng.shuffle(self.samples_v3)

    def __len__(self):
        return max(len(self.samples_v2), len(self.samples_v3))

    def __getitem__(self, idx):
        i2 = idx % len(self.samples_v2)
        i3 = idx % len(self.samples_v3)
        (v1, v2) = self.samples_v2[i2]
        ((a, b), v3) = self.samples_v3[i3]
        return {
            "v1": torch.from_numpy(np.asarray(v1)),
            "v2": torch.from_numpy(np.asarray(v2)),
            "v3": torch.from_numpy(np.asarray(v3)),
            "pair_v2": (torch.from_numpy(np.asarray(v1)), torch.from_numpy(np.asarray(v2))),
            "pair_v3": (torch.from_numpy(np.asarray(a)), torch.from_numpy(np.asarray(b)), torch.from_numpy(np.asarray(v3))),
        }

def sort_zyx(tri):
    # tri: (3,3) np array
    idx = np.lexsort((tri[:,0], tri[:,1], tri[:,2]))  # lex by x after y after z → use reversed order
    # we want z desc, y desc, x desc → negate
    key = np.column_stack([-tri[:,2], -tri[:,1], -tri[:,0]])
    idx = np.lexsort((key[:,2], key[:,1], key[:,0]))
    v1, v2, v3 = tri[idx[0]], tri[idx[1]], tri[idx[2]]
    return v1, v2, v3