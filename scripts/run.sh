#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <run_name>"
  exit 1
fi

RUN_NAME="$1"

# 可以通过环境变量覆盖超参数
: "${EPOCHS:=9999}"
: "${BATCH_SIZE:=4}"
: "${LR:=1e-3}"
: "${D_MODEL:=256}"
: "${NHEAD:=8}"
: "${TOK_DEPTH:=2}"
: "${FLOW_LAYERS:=4}"
: "${HEUN_STEPS:=20}"
: "${MAX_EDGES:=4096}"

export EPOCHS BATCH_SIZE LR D_MODEL NHEAD TOK_DEPTH FLOW_LAYERS HEUN_STEPS MAX_EDGES

echo "[run] 开始训练: ${RUN_NAME}"
bash scripts/train.sh "${RUN_NAME}"

echo "[run] 开始重建评估: ${RUN_NAME}"
bash scripts/reconstruct.sh "${RUN_NAME}"

echo "[run] 全流程完成: ${RUN_NAME}"
