# src/datasets/mesh_dataset.py (关键片段)
import os, numpy as np
from tqdm import tqdm
from ..utils.io import load_mesh, iter_mesh_files
from ..utils.geometry import sort_triangle_vertices_zyx_with_idx, normalize_unit_box
from .transforms import precompute_vertex_knn_and_frames

def export_npz_from_mesh_root(root, out_npz, split_ratio=(0.8,0.1,0.1),
                              knn_k=24, use_local_frame=True, seed=42):
    mesh_paths = list(iter_mesh_files(root))
    assert len(mesh_paths) > 0, f"No meshes found in {root}"
    rng = np.random.default_rng(seed); rng.shuffle(mesh_paths)

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

            # ⬇️ 关键：为该 mesh 一次性预建每个顶点的邻居与局部系
            if knn_k > 0:
                nbr_idx, frames = precompute_vertex_knn_and_frames(
                    Vn, k=knn_k, use_local_frame=use_local_frame
                )

            for f in F:
                # 同时拿到排序后的“原始索引”
                tri_sorted, idx_sorted = sort_triangle_vertices_zyx_with_idx(Vn, f)
                v1, v2, v3 = tri_sorted[0], tri_sorted[1], tri_sorted[2]
                v1_idx = idx_sorted[0]  # 直接得到 anchor 索引，避免 O(|V|) 搜索

                # 上下文：直接查表
                if knn_k > 0:
                    nbr = nbr_idx[v1_idx]              # (k,)
                    rel = Vn[nbr] - Vn[v1_idx]         # (k,3)
                    R = frames[v1_idx]                 # (3,3)
                    ctx_local = (rel @ R).astype(np.float32)
                else:
                    ctx_local = np.zeros((0,3), dtype=np.float32)

                all_data[split]['v1'].append(v1.astype(np.float32))
                all_data[split]['v2'].append(v2.astype(np.float32))
                all_data[split]['v3'].append(v3.astype(np.float32))
                all_data[split]['ctx'].append(ctx_local)
                all_data[split]['mesh_id'].append(mid)

        # stack
        for key in ['v1','v2','v3']:
            all_data[split][key] = np.stack(all_data[split][key], axis=0)
        all_data[split]['ctx'] = (np.stack(all_data[split]['ctx'], axis=0)
                                  if len(all_data[split]['ctx'])>0
                                  else np.zeros((0,0,3), dtype=np.float32))
        all_data[split]['mesh_id'] = np.array(all_data[split]['mesh_id'], dtype=np.int64)

    np.savez_compressed(out_npz, 
        train_v1=all_data['train']['v1'], train_v2=all_data['train']['v2'], train_v3=all_data['train']['v3'],
        train_ctx=all_data['train']['ctx'], train_mesh_id=all_data['train']['mesh_id'],
        val_v1=all_data['val']['v1'],     val_v2=all_data['val']['v2'],     val_v3=all_data['val']['v3'],
        val_ctx=all_data['val']['ctx'],   val_mesh_id=all_data['val']['mesh_id'],
        test_v1=all_data['test']['v1'],   test_v2=all_data['test']['v2'],   test_v3=all_data['test']['v3'],
        test_ctx=all_data['test']['ctx'], test_mesh_id=all_data['test']['mesh_id'],
        meta=dict(knn_k=knn_k, use_local_frame=use_local_frame, root=root)
    )
