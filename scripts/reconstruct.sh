#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <run_name>"
  exit 1
fi

RUN_NAME="$1"
BASE_DIR="outs/${RUN_NAME}"
CKPT_DIR="${BASE_DIR}/ckpts"
LOG_DIR="${BASE_DIR}/logs"
EVAL_DIR="${BASE_DIR}/eval"
CFG_DIR="${BASE_DIR}/configs"

mkdir -p "${EVAL_DIR}" "${LOG_DIR}"

# 读取超参数
if [[ -f "${CFG_DIR}/hparams.sh" ]]; then
  source "${CFG_DIR}/hparams.sh"
else
  echo "未找到 ${CFG_DIR}/hparams.sh，请先训练或手动提供。"
  exit 1
fi

DATA_PATH="./outs/data/test"
JOINT_CKPT="${CKPT_DIR}/ckpt_best.pt"

if [[ ! -f "${JOINT_CKPT}" ]]; then
  echo "未找到权重: ${JOINT_CKPT}"
  exit 1
fi

LOG_FILE="${LOG_DIR}/reconstruct_$(date +%Y%m%d_%H%M%S).log"
OUT_TXT="${EVAL_DIR}/reconstruct_test.txt"

echo "[reconstruct] run_name=${RUN_NAME}"
echo "[reconstruct] 使用权重: ${JOINT_CKPT}"
echo "[reconstruct] 数据目录: ${DATA_PATH}"

PYTHONPATH=. \
python -m src.tools.reconstruct_eval_tokens \
  --data "${DATA_PATH}" \
  --joint_ckpt "${JOINT_CKPT}" \
  --steps "${HEUN_STEPS}" \
  --max_edges "${MAX_EDGES}" \
  2>&1 | tee "${LOG_FILE}"

grep -E "^\[OVERALL\]" "${LOG_FILE}" | tail -n 1 > "${OUT_TXT}" || true

echo "[reconstruct] 完成。日志: ${LOG_FILE}"
echo "[reconstruct] 汇总结果: ${OUT_TXT}"
