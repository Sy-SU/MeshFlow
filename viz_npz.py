#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render meshes directly from tri_dataset.npz to PNG images (offscreen).
- Groups triangles by mesh_id per split (train/val/test)
- Deduplicates vertices (round or epsilon grid)
- Builds open3d TriangleMesh and renders to outs/renders/<split>/*.png

Usage:
  python render_from_npz.py --npz outs/tri_dataset.npz --split train --limit 12
"""

import os
import argparse
import numpy as np
import open3d as o3d


def dedup_vertices(triangles: np.ndarray, mode="round", precision=6, epsilon=1e-6):
    """
    triangles: (M,3,3) float32
    Return: V (K,3) float32, F (M,3) int32 (0-based indices)
    """
    if mode == "round":
        key = lambda p: (round(float(p[0]), precision),
                         round(float(p[1]), precision),
                         round(float(p[2]), precision))
    elif mode == "eps":
        inv = 1.0 / float(epsilon)
        key = lambda p: (int(np.floor(float(p[0]) * inv)),
                         int(np.floor(float(p[1]) * inv)),
                         int(np.floor(float(p[2]) * inv)))
    else:
        raise ValueError("mode must be 'round' or 'eps'")

    vmap = {}
    V = []
    F = np.empty((triangles.shape[0], 3), dtype=np.int32)

    for i, tri in enumerate(triangles):
        fi = []
        for j in range(3):
            kk = key(tri[j])
            idx = vmap.get(kk)
            if idx is None:
                idx = len(V)
                vmap[kk] = idx
                V.append((float(tri[j, 0]), float(tri[j, 1]), float(tri[j, 2])))
            fi.append(idx)
        F[i] = fi

    return np.asarray(V, dtype=np.float32), F


def make_camera_auto(mesh: o3d.geometry.TriangleMesh, fov_deg=35.0):
    aabb = mesh.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    extent = aabb.get_extent()
    radius = float(np.linalg.norm(extent) * 0.6 + 1e-6)
    eye = center + np.array([1.5, 1.2, 1.0]) * radius
    up = np.array([0.0, 0.0, 1.0])
    return eye, center, up, fov_deg


def render_one_mesh(V, F, out_path, width=1000, height=800, bg=(1,1,1,1)):
    import os, gc, numpy as np, open3d as o3d

    # --- build mesh ---
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F.astype(np.int32))
    if len(mesh.triangles) == 0:
        return False
    mesh.compute_vertex_normals()

    # --- offscreen renderer ---
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background(bg)

    # --- try lit material; fallback to unlit if anything fails ---
    use_lit = True
    try:
        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.shader = "defaultLit"  # 有光照
        # 基础颜色总是可用
        mtl.base_color = (0.78, 0.80, 0.85, 1.0)
        # 不同版本字段名可能不同：有则设置、无则跳过
        if hasattr(mtl, "base_roughness"):
            mtl.base_roughness = 0.4
        elif hasattr(mtl, "roughness"):
            mtl.roughness = 0.4
        if hasattr(mtl, "base_metallic"):
            mtl.base_metallic = 0.0
        elif hasattr(mtl, "metallic"):
            mtl.metallic = 0.0
        if hasattr(mtl, "base_reflectance"):
            mtl.base_reflectance = 0.5
        scene.add_geometry("mesh", mesh, mtl)

        # 方向光（太阳光）+ 间接光（有 API 再调用）
        if hasattr(scene.scene, "set_sun_light"):
            # set_sun_light(direction(x,y,z), color(r,g,b), intensity)
            scene.scene.set_sun_light([1, 1, 1], [1.0, 1.0, 1.0], 200000)
        if hasattr(scene.scene, "enable_sun_light"):
            scene.scene.enable_sun_light(True)
        # 部分版本有间接光接口
        if hasattr(scene.scene, "set_indirect_light_intensity"):
            scene.scene.set_indirect_light_intensity(8000)
        # 新一些的版本有 IBL（环境贴图）加载/强度设置接口，存在就用
        if hasattr(scene.scene, "set_indirect_light"):
            # 这里不加载外部 HDRI，仅确保不会出错；若你有 .ibl/.hdr 可在此加载
            pass

    except Exception as e:
        # 回退到无光照，避免版本不兼容导致崩溃
        use_lit = False
        try:
            # 移除已添加的几何体（如果前面 add 成功过）
            scene.remove_geometry("mesh")
        except Exception:
            pass
        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.shader = "defaultUnlit"
        mtl.base_color = (0.80, 0.82, 0.87, 1.0)
        scene.add_geometry("mesh", mesh, mtl)

    # --- camera ---
    aabb   = mesh.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    extent = aabb.get_extent()
    radius = float(np.linalg.norm(extent) * 0.6 + 1e-6)
    eye    = center + np.array([1.5, 1.2, 1.0]) * radius
    up     = np.array([0.0, 0.0, 1.0])
    renderer.setup_camera(35.0, center, eye, up)

    # --- render & save ---
    img = renderer.render_to_image()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    o3d.io.write_image(out_path, img)

    # --- cleanup (旧版没有 .release) ---
    try:
        scene.remove_geometry("mesh")
    except Exception:
        pass
    del renderer
    gc.collect()
    return True

def render_from_npz(npz_path, split="train", out_dir="outs/renders",
                    limit=None, mode="round", precision=20, epsilon=1e-6,
                    width=1000, height=800):
    z = np.load(npz_path, allow_pickle=True)
    v1 = z[f"{split}_v1"]       # (N,3)
    v2 = z[f"{split}_v2"]       # (N,3)
    v3 = z[f"{split}_v3"]       # (N,3)
    mid = z[f"{split}_mesh_id"] # (N,)

    # sort by mesh_id, group consecutive ranges
    order = np.argsort(mid, kind="mergesort")
    v1, v2, v3, mid = v1[order], v2[order], v3[order], mid[order]

    uniq, start_idx, counts = np.unique(mid, return_index=True, return_counts=True)
    if limit is not None:
        uniq, start_idx, counts = uniq[:limit], start_idx[:limit], counts[:limit]

    out_split = os.path.join(out_dir, split)
    os.makedirs(out_split, exist_ok=True)

    total = len(uniq)
    print(f"[{split}] rendering {total} mesh(es) from {npz_path}")

    for i, (m, s, c) in enumerate(zip(uniq, start_idx, counts), 1):
        tris = np.stack([v1[s:s+c], v2[s:s+c], v3[s:s+c]], axis=1)  # (M,3,3)
        V, F = dedup_vertices(tris, mode=mode, precision=precision, epsilon=epsilon)
        out_path = os.path.join(out_split, f"mesh_{int(m)}.png")
        ok = render_one_mesh(V, F, out_path, width=width, height=height)
        tag = "OK" if ok else "SKIP"
        print(f"  [{i}/{total}] {tag}: {out_path}  (V={len(V)}, F={len(F)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="outs/tri_dataset.npz", help="path to tri_dataset.npz")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--out", default="outs/renders", help="output image root dir")
    ap.add_argument("--limit", type=int, default=8, help="render first N meshes")
    ap.add_argument("--mode", choices=["round", "eps"], default="round",
                    help="vertex merge mode (round=decimals / eps=grid)")
    ap.add_argument("--precision", type=int, default=6, help="round decimals (1e-6)")
    ap.add_argument("--epsilon", type=float, default=1e-6, help="grid size for eps mode")
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--height", type=int, default=800)
    args = ap.parse_args()

    render_from_npz(args.npz, args.split, args.out, args.limit,
                    args.mode, args.precision, args.epsilon,
                    width=args.width, height=args.height)
