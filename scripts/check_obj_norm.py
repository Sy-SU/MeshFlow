#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查 OBJ 顶点是否已归一化：
- 读取 v 行（v x y z），忽略注释/法线/纹理/面等
- 指标：坐标范围、均值(中心)、最大范数(半径)
- 判定：更像 [-1,1] or [0,1]，或由用户指定
"""

import os
import sys
import argparse
import math
import gzip
from typing import Tuple, List, Optional

import numpy as np


def iter_obj_vertices(path: str):
    """逐行读取 OBJ，产出 (x,y,z)。支持 .gz"""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 只取几何顶点 v（不含vt/vn）
            if not line or line[0] != 'v' or (len(line) > 1 and line[1] not in (' ', '\t')):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                yield x, y, z
            except ValueError:
                continue


def load_vertices(path: str) -> Optional[np.ndarray]:
    vs = list(iter_obj_vertices(path))
    if not vs:
        return None
    return np.asarray(vs, dtype=np.float64)  # [N,3]


def stats_for_vertices(v: np.ndarray):
    """返回统计信息"""
    mean = v.mean(axis=0)                     # (3,)
    vmin = v.min(axis=0)
    vmax = v.max(axis=0)
    max_abs = np.abs(v).max()
    radii = np.linalg.norm(v, axis=1)
    rmax = radii.max()
    bbox_size = (vmax - vmin)
    return {
        "count": v.shape[0],
        "mean": mean,
        "min": vmin,
        "max": vmax,
        "max_abs": max_abs,
        "rmax": rmax,
        "bbox_size": bbox_size,
    }


def judge_normalization(s: dict, tol: float = 1e-3) -> Tuple[str, List[str]]:
    """
    自动判断更像哪种归一化：
      - 'neg1to1'  ：坐标落在 [-1,1]，中心接近 0，最大半径≈1
      - 'zero1'    ：坐标落在 [0,1]，中心接近 0.5，边长≈1（至少一个维度）
      - 'none'     ：都不像
    返回 (label, reasons)
    """
    reasons = []
    vmin, vmax = s["min"], s["max"]
    mean = s["mean"]; rmax = s["rmax"]; max_abs = s["max_abs"]; bbox = s["bbox_size"]

    # 条件组（带容差）
    in_neg1to1 = (vmin >= (-1 - tol)).all() and (vmax <= (1 + tol)).all()
    center_near_0 = np.linalg.norm(mean, ord=2) <= (5 * tol)  # 稍微放宽些
    rmax_close_1 = abs(rmax - 1.0) <= 0.05  # ±5%

    in_zero1 = (vmin >= (0 - tol)).all() and (vmax <= (1 + tol)).all()
    center_near_05 = np.linalg.norm(mean - 0.5, ord=2) <= 0.05  # 0.5±0.05
    any_edge_close_1 = np.any(np.abs(bbox - 1.0) <= 0.05)  # 任一维边长≈1

    # 规则优先：先看 [-1,1] 套件是否全部满足；否则尝试 [0,1]
    if in_neg1to1 and center_near_0 and rmax_close_1:
        reasons.append("范围在[-1,1]内、中心≈0、最大半径≈1")
        return "neg1to1", reasons
    if in_zero1 and (center_near_05 or any_edge_close_1):
        reasons.append("范围在[0,1]内，且中心≈0.5或边长≈1")
        return "zero1", reasons

    # 若不能严格判定，尝试弱匹配（只看范围）
    if in_neg1to1:
        reasons.append("范围在[-1,1]内，但中心/半径不够符合")
        return "neg1to1?", reasons
    if in_zero1:
        reasons.append("范围在[0,1]内，但中心/边长不够符合")
        return "zero1?", reasons

    # 都不像
    if max_abs > 1 + 1e-2 and (vmin.min() < -1 - 1e-2 or vmax.max() > 1 + 1e-2):
        reasons.append("坐标超出[-1,1]较多")
    return "none", reasons


def pretty_vec(a: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.5f}" for x in a.tolist()) + "]"


def check_one(path: str, assume: str, tol: float) -> Tuple[str, Optional[dict], str]:
    """
    返回 (path, stats or None, verdict_text)
    assume: auto / neg1to1 / zero1
    """
    v = load_vertices(path)
    if v is None:
        return path, None, "无顶点(v)或无法解析"
    s = stats_for_vertices(v)

    if assume == "auto":
        label, reasons = judge_normalization(s, tol)
        verdict = f"{label} ({'; '.join(reasons)})" if reasons else label
    elif assume == "neg1to1":
        cond = (s["min"] >= (-1 - tol)).all() and (s["max"] <= (1 + tol)).all()
        center = np.linalg.norm(s["mean"], ord=2) <= (5 * tol)
        rmax_close_1 = abs(s["rmax"] - 1.0) <= 0.05
        verdict = "OK" if (cond and center and rmax_close_1) else "NOT_OK"
    elif assume == "zero1":
        cond = (s["min"] >= (0 - tol)).all() and (s["max"] <= (1 + tol)).all()
        center = np.linalg.norm(s["mean"] - 0.5, ord=2) <= 0.05
        any_edge_close_1 = np.any(np.abs(s["bbox_size"] - 1.0) <= 0.05)
        verdict = "OK" if (cond and (center or any_edge_close_1)) else "NOT_OK"
    else:
        verdict = "unknown"

    # 打印细节
    print(f"[{path}]")
    print(f"  顶点数: {s['count']}")
    print(f"  min: {pretty_vec(s['min'])}")
    print(f"  max: {pretty_vec(s['max'])}")
    print(f"  mean(center): {pretty_vec(s['mean'])}")
    print(f"  max|coord|: {s['max_abs']:.6f}   max_radius: {s['rmax']:.6f}")
    print(f"  bbox_size: {pretty_vec(s['bbox_size'])}")
    print(f"  判定: {verdict}")
    print()
    return path, s, verdict


def walk_objs(root: str, recursive: bool) -> List[str]:
    files = []
    if os.path.isdir(root):
        for dirpath, dirnames, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith((".obj", ".obj.gz")):
                    files.append(os.path.join(dirpath, f))
            if not recursive:
                break
    else:
        if root.lower().endswith((".obj", ".obj.gz")):
            files.append(root)
    return sorted(files)


def main():
    ap = argparse.ArgumentParser(description="检查 OBJ 数据是否归一化")
    ap.add_argument("path", type=str, help="OBJ 文件或包含 OBJ 的目录（支持 .obj.gz）")
    ap.add_argument("--recursive", action="store_true", help="目录时递归遍历")
    ap.add_argument("--assume", type=str, default="auto",
                    choices=["auto", "neg1to1", "zero1"],
                    help="判定模式：auto 自动选择；或强制按某种规范检查")
    ap.add_argument("--tol", type=float, default=1e-3, help="范围与中心的容差")
    args = ap.parse_args()

    files = walk_objs(args.path, args.recursive)
    if not files:
        print("未找到 OBJ 文件")
        sys.exit(1)

    # 汇总
    counts = {}
    global_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    global_max = -global_min
    global_sum = np.zeros(3, dtype=np.float64)
    global_n = 0

    for p in files:
        _, s, verdict = check_one(p, args.assume, args.tol)
        counts[verdict] = counts.get(verdict, 0) + 1
        if s is not None:
            global_min = np.minimum(global_min, s["min"])
            global_max = np.maximum(global_max, s["max"])
            global_sum += s["mean"] * s["count"]
            global_n += s["count"]

    print("====== 汇总 ======")
    for k in sorted(counts.keys()):
        print(f"{k:>10s}: {counts[k]}")
    if global_n > 0:
        dataset_mean = global_sum / global_n
        print(f"全局 min: {pretty_vec(global_min)}")
        print(f"全局 max: {pretty_vec(global_max)}")
        print(f"加权均值(所有点): {pretty_vec(dataset_mean)}")


if __name__ == "__main__":
    main()
