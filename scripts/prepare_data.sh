#!/usr/bin/env bash
set -e
ROOT=${1:-"/root/autodl-fs/ShapeNetCore.v2"}
OUT=${2:-"outs/tri_dataset.npz"}
K=${3:-24}
USE_LOCAL=${4:-"true"}

# 关键：参数写在 <<'PY' 之前
PYTHONPATH=./ python - "$ROOT" "$OUT" "$K" "$USE_LOCAL" <<'PY'
import os, sys
from src.datasets.mesh_dataset import export_npz_from_mesh_root

root = sys.argv[1]
out  = sys.argv[2]
k    = int(sys.argv[3])
use_local = sys.argv[4].lower() == "true"

os.makedirs(os.path.dirname(out), exist_ok=True)
export_npz_from_mesh_root(root, out, knn_k=k, use_local_frame=use_local)
print("Saved to", out)
PY
