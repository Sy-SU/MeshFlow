#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <run_prefix>"
  exit 1
fi

RUN_PREFIX="$1"
DATE=$(date +%Y%m%d_%H%M%S)

# ====== 搜索空间 ======
LR_LIST=("1e-3" "7e-4" "5e-4" "2e-4" "1e-4")
WARMUP_LIST=(5)  # warmup 搜索空间
DIRW_LIST=("0.0005" "0.0001" "0.005" "0.001" "0.05" "0.01")
BATCH=4

# ====== 保存结果 ======
RESULT_DIR="outs/${RUN_PREFIX}_${DATE}"
RESULT_FILE="${RESULT_DIR}/tune_results.txt"
mkdir -p "${RESULT_DIR}"
echo "[AutoTune] 保存结果到 ${RESULT_FILE}"
echo "run_name, lr, batch, warmup, dir_weight, best_val, recon_error" > "${RESULT_FILE}"

EXP_ID=0
for LR in "${LR_LIST[@]}"; do
  for DW in "${DIRW_LIST[@]}"; do
    for WARMUP in "${WARMUP_LIST[@]}"; do
      EXP_ID=$((EXP_ID+1))

      lr_tag="${LR//./p}"
      dw_tag="${DW//./p}"
      RUN_NAME="${RUN_PREFIX}_${DATE}_exp${EXP_ID}_lr${lr_tag}_bs${BATCH}_wu${WARMUP}_dw${dw_tag}"

      echo
      echo "====== 运行实验 ${EXP_ID}: lr=${LR}, batch=${BATCH}, warmup=${WARMUP}, dir_weight=${DW} ======"

      # 通过 run.sh 一键执行（训练 + 重建）
      LR="${LR}" BATCH_SIZE="${BATCH}" WARMUP_EPOCHS="${WARMUP}" DIR_WEIGHT="${DW}" \
        bash scripts/run.sh "${RUN_NAME}"

      # 从训练阶段保存的 early_stop.json 或 ckpt_best.pt 里读 best_val
      CKPT_DIR="outs/${RUN_NAME}/ckpts"
      BEST_JSON="${CKPT_DIR}/early_stop.json"
      if [[ -f "${BEST_JSON}" ]]; then
        BEST_VAL=$(python - <<PY
import json
with open("${BEST_JSON}", "r", encoding="utf-8") as f:
    print(json.load(f).get("best_val", "nan"))
PY
)
      else
        BEST_VAL=$(python - <<PY
import torch
ckpt = torch.load("${CKPT_DIR}/ckpt_best.pt", map_location="cpu")
print(ckpt.get("best_metric", float("nan")))
PY
)
      fi

      # 从重建脚本保存的结果文件里取 recon_error（假设 reconstruct.sh 会保存结果）
      RECON_FILE="outs/${RUN_NAME}/eval/reconstruct_test.txt"
      if [[ -f "${RECON_FILE}" ]]; then
        RECON_ERROR=$(tail -n 1 "${RECON_FILE}")
      else
        RECON_ERROR="nan"
      fi

      echo "${RUN_NAME}, ${LR}, ${BATCH}, ${WARMUP}, ${DW}, ${BEST_VAL}, ${RECON_ERROR}" \
        | tee -a "${RESULT_FILE}"
    done
  done
done

echo
echo "[AutoTune] 所有实验完成，结果见: ${RESULT_FILE}"
