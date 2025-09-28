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
BATCH_LIST=(4)
WARMUP_LIST=(0 1 2 3 5 7 10)

# ====== 保存结果 ======
RESULT_FILE="outs/${RUN_PREFIX}_${DATE}/tune_results.txt"
mkdir -p "$(dirname "${RESULT_FILE}")"
echo "[AutoTune] 保存结果到 ${RESULT_FILE}"
echo "run_name, lr, batch, warmup, best_val" > "${RESULT_FILE}"

EXP_ID=0
for LR in "${LR_LIST[@]}"; do
  for BATCH in "${BATCH_LIST[@]}"; do
    for WARMUP in "${WARMUP_LIST[@]}"; do
      EXP_ID=$((EXP_ID+1))
      RUN_NAME="${RUN_PREFIX}_${DATE}_exp${EXP_ID}"

      echo
      echo "====== 运行实验 ${EXP_ID}: lr=${LR}, batch=${BATCH}, warmup=${WARMUP} ======"

      # 调用已有 train.sh
      LR=${LR} BATCH_SIZE=${BATCH} WARMUP_EPOCHS=${WARMUP} ./scripts/train.sh "${RUN_NAME}"

      # 读取 val 最优指标
      CKPT_DIR="outs/${RUN_NAME}/ckpts"
      BEST_JSON="${CKPT_DIR}/early_stop.json"
      if [[ -f "${BEST_JSON}" ]]; then
        BEST_VAL=$(jq -r '.best_val' "${BEST_JSON}")
      else
        # 如果没早停，就看 ckpt_best.pt 里面的 best_metric
        BEST_VAL=$(python -c "import torch;print(torch.load('${CKPT_DIR}/ckpt_best.pt')['best_metric'])")
      fi

      echo "${RUN_NAME}, ${LR}, ${BATCH}, ${WARMUP}, ${BEST_VAL}" | tee -a "${RESULT_FILE}"
    done
  done
done

echo
echo "[AutoTune] 所有实验完成，结果见: ${RESULT_FILE}"
