import os, numpy as np
from tqdm import tqdm
from ..utils.io import load_mesh, iter_mesh_files
from ..utils.geometry import sort_triangle_vertices_zyx, normalize_unit_box, local_frame
from .transforms import build_knn_context, to_local_frame

def export_npz_from_mesh_root(
    root, out_npz, split_ratio=(0.8, 0.1, 0.1), 
    knn_k=24, use_local_frame=True, seed=42
):
    """
    Read all meshes under root, build (v1,v2,v3,ctx) dataset and save to NPZ.
    We split by mesh (not by triangles).
    """
    mesh_paths = list(iter_mesh_files(root))
    assert len(mesh_paths) > 0, f"No meshes found in {root}"
    rng = np.random.default_rng(seed)
    rng.shuffle(mesh_paths)

    n = len(mesh_paths)
    n_tr = int(n * split_ratio[0]); n_va = int(n * split_ratio[1])
    splits = {
        'train': mesh_paths[:n_tr],
        'val'  : mesh_paths[n_tr:n_tr+n_va],
        'test' : mesh_paths[n_tr+n_va:],
    }

    all_data = {k: {'v1':[], 'v2':[], 'v3':[], 'ctx':[], 'mesh_id':[]} for k in splits}

    for split, paths in splits.items():
        for mid, path in enumerate(tqdm(paths, desc=f'[{split}]')):
            V, F = load_mesh(path)
            Vn, _bbox = normalize_unit_box(V)
            # 枚举三角面片
            for f in F:
                tri = Vn[f]  # (3,3)
                tri_sorted = sort_triangle_vertices_zyx(tri)
                v1, v2, v3 = tri_sorted[0], tri_sorted[1], tri_sorted[2]

                # 选 v1 的全局索引（用于构造上下文）。这里简单地用最近点匹配回原 Vn
                # 也可以记录 f 对应排序后的原始索引（更精确），此处做最小可行版：
                anchor_idx = np.argmin(((Vn - v1)**2).sum(axis=1))

                # 上下文
                if knn_k > 0:
                    ctx_idx, ctx_rel = build_knn_context(Vn, anchor_idx, k=knn_k)
                    if use_local_frame:
                        R = local_frame(ctx_rel)      # PCA frame on neighborhood
                        ctx_local = to_local_frame(ctx_rel, R)  # (k,3)
                    else:
                        ctx_local = ctx_rel
                else:
                    ctx_local = np.zeros((0,3), dtype=np.float32)

                all_data[split]['v1'].append(v1.astype('float32'))
                all_data[split]['v2'].append(v2.astype('float32'))
                all_data[split]['v3'].append(v3.astype('float32'))
                all_data[split]['ctx'].append(ctx_local.astype('float32'))
                all_data[split]['mesh_id'].append(mid)

        # 堆叠为数组；ctx 是变长 k，可保存为 (N,k,3)
        for key in ['v1','v2','v3']:
            all_data[split][key] = np.stack(all_data[split][key], axis=0)  # (N,3)
        # ctx：若 k 固定可直接 stack
        if len(all_data[split]['ctx']) > 0:
            all_data[split]['ctx'] = np.stack(all_data[split]['ctx'], axis=0)  # (N,k,3)
        else:
            all_data[split]['ctx'] = np.zeros((0,0,3), dtype=np.float32)
        all_data[split]['mesh_id'] = np.array(all_data[split]['mesh_id'], dtype=np.int64)

    np.savez_compressed(out_npz,
        train_v1=all_data['train']['v1'],
        train_v2=all_data['train']['v2'],
        train_v3=all_data['train']['v3'],
        train_ctx=all_data['train']['ctx'],
        train_mesh_id=all_data['train']['mesh_id'],
        val_v1=all_data['val']['v1'],
        val_v2=all_data['val']['v2'],
        val_v3=all_data['val']['v3'],
        val_ctx=all_data['val']['ctx'],
        val_mesh_id=all_data['val']['mesh_id'],
        test_v1=all_data['test']['v1'],
        test_v2=all_data['test']['v2'],
        test_v3=all_data['test']['v3'],
        test_ctx=all_data['test']['ctx'],
        test_mesh_id=all_data['test']['mesh_id'],
        meta=dict(knn_k=knn_k, use_local_frame=use_local_frame, root=root)
    )

class MeshTriNPZ:
    """ 简单的可迭代数据集（训练时你可换成 PyTorch Dataset） """
    def __init__(self, npz_path, split='train'):
        z = np.load(npz_path, allow_pickle=True)
        self.v1  = z[f'{split}_v1']
        self.v2  = z[f'{split}_v2']
        self.v3  = z[f'{split}_v3']
        self.ctx = z[f'{split}_ctx']
        self.mid = z[f'{split}_mesh_id']
    def __len__(self): return self.v1.shape[0]
    def __getitem__(self, i):
        return dict(v1=self.v1[i], v2=self.v2[i], v3=self.v3[i], ctx=self.ctx[i], mesh_id=int(self.mid[i]))
