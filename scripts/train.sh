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
CFG_DIR="${BASE_DIR}/configs"

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${CFG_DIR}"

# ===== 可通过环境变量覆盖（训练）=====
EPOCHS=${EPOCHS:-9999}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-2e-4}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_EDGES=${MAX_EDGES:-2048}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
TRAIN_DT=${TRAIN_DT:-0.01}
DIR_WEIGHT=${DIR_WEIGHT:-0.01}
GRAD_CLIP=${GRAD_CLIP:-1.0}
AMP_FLAG=${AMP_FLAG:-1}          # 1 开启AMP，0关闭
VAL_STEPS=${VAL_STEPS:-100}
DEVICE=${DEVICE:-cuda}

# ===== 早停 =====
EARLY_MIN_EPOCHS=${EARLY_MIN_EPOCHS:-5}
EARLY_PATIENCE=${EARLY_PATIENCE:-15}
EARLY_MIN_DELTA=${EARLY_MIN_DELTA:-0}
EARLY_COOLDOWN=${EARLY_COOLDOWN:-5}

# ===== 学习率调度（与 train.py 对齐）=====
SCHEDULER=${SCHEDULER:-none}                       # none / cosine / cosine_warm_restarts / step / multistep / plateau / onecycle
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
MIN_LR=${MIN_LR:-1e-6}
TMAX=${TMAX:-$EPOCHS}                              # 对 cosine/onecycle，通常设置为总 epoch
RESTART_PERIOD=${RESTART_PERIOD:-10}               # cosine_warm_restarts: T_0
RESTART_MULT=${RESTART_MULT:-2.0}                  # cosine_warm_restarts: T_mult
STEP_SIZE=${STEP_SIZE:-30}                         # step
GAMMA=${GAMMA:-0.1}                                # step/multistep
MILESTONES=${MILESTONES:-""}                       # multistep: 例如 "60,80"
PLATEAU_PATIENCE=${PLATEAU_PATIENCE:-5}            # plateau
PLATEAU_FACTOR=${PLATEAU_FACTOR:-0.5}              # plateau
ONECYCLE_PCT_START=${ONECYCLE_PCT_START:-0.3}      # onecycle

# 是否用伪 TTY 包裹以保留漂亮的 tqdm 进度条（1=开启，0=关闭）
TTY_WRAP=${TTY_WRAP:-1}

# ===== 模型结构 =====
D_MODEL=${D_MODEL:-256}
NHEAD=${NHEAD:-8}
TOK_NUM_TOKENS=${TOK_NUM_TOKENS:-32}
TOK_DEPTH=${TOK_DEPTH:-2}
FLOW_LAYERS=${FLOW_LAYERS:-4}
PE_NUM_FREQ=${PE_NUM_FREQ:-16}
PE_INCLUDE_INPUT=${PE_INCLUDE_INPUT:-1}  # 1 表示传 --pe_include_input

# ===== 记录超参数（可供重建脚本复用）=====
cat > "${CFG_DIR}/hparams.sh" <<EOF
export EPOCHS=${EPOCHS}
export BATCH_SIZE=${BATCH_SIZE}
export LR=${LR}
export NUM_WORKERS=${NUM_WORKERS}
export MAX_EDGES=${MAX_EDGES}
export WEIGHT_DECAY=${WEIGHT_DECAY}
export TRAIN_DT=${TRAIN_DT}
export DIR_WEIGHT=${DIR_WEIGHT}
export GRAD_CLIP=${GRAD_CLIP}
export AMP_FLAG=${AMP_FLAG}
export VAL_STEPS=${VAL_STEPS}
export DEVICE=${DEVICE}
export D_MODEL=${D_MODEL}
export NHEAD=${NHEAD}
export TOK_NUM_TOKENS=${TOK_NUM_TOKENS}
export TOK_DEPTH=${TOK_DEPTH}
export FLOW_LAYERS=${FLOW_LAYERS}
export PE_NUM_FREQ=${PE_NUM_FREQ}
export PE_INCLUDE_INPUT=${PE_INCLUDE_INPUT}
export EARLY_MIN_EPOCHS=${EARLY_MIN_EPOCHS}
export EARLY_PATIENCE=${EARLY_PATIENCE}
export EARLY_MIN_DELTA=${EARLY_MIN_DELTA}
export EARLY_COOLDOWN=${EARLY_COOLDOWN}
export SCHEDULER=${SCHEDULER}
export WARMUP_EPOCHS=${WARMUP_EPOCHS}
export MIN_LR=${MIN_LR}
export TMAX=${TMAX}
export RESTART_PERIOD=${RESTART_PERIOD}
export RESTART_MULT=${RESTART_MULT}
export STEP_SIZE=${STEP_SIZE}
export GAMMA=${GAMMA}
export MILESTONES="${MILESTONES}"
export PLATEAU_PATIENCE=${PLATEAU_PATIENCE}
export PLATEAU_FACTOR=${PLATEAU_FACTOR}
export ONECYCLE_PCT_START=${ONECYCLE_PCT_START}
EOF

LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "[train] run_name=${RUN_NAME}"
echo "[train] ckpt_dir=${CKPT_DIR}"

# ===== 组装命令 =====
CMD="PYTHONPATH=. python -m src.tools.train \
  --data_root ./outs/data \
  --batch_size \"${BATCH_SIZE}\" \
  --num_workers \"${NUM_WORKERS}\" \
  --max_edges_per_mesh \"${MAX_EDGES}\" \
  --epochs \"${EPOCHS}\" \
  --lr \"${LR}\" \
  --weight_decay \"${WEIGHT_DECAY}\" \
  --train_dt \"${TRAIN_DT}\" \
  --dir_weight \"${DIR_WEIGHT}\" \
  --grad_clip \"${GRAD_CLIP}\" \
  --val_steps \"${VAL_STEPS}\" \
  --device \"${DEVICE}\" \
  --ckpt_dir \"${CKPT_DIR}\" \
  --run_name \"${RUN_NAME}\" \
  --d_model \"${D_MODEL}\" \
  --nhead \"${NHEAD}\" \
  --tok_num_tokens \"${TOK_NUM_TOKENS}\" \
  --tok_depth \"${TOK_DEPTH}\" \
  --flow_layers \"${FLOW_LAYERS}\" \
  --pe_num_frequencies \"${PE_NUM_FREQ}\" \
  --early_min_epochs \"${EARLY_MIN_EPOCHS}\" \
  --early_patience \"${EARLY_PATIENCE}\" \
  --early_min_delta \"${EARLY_MIN_DELTA}\" \
  --early_cooldown \"${EARLY_COOLDOWN}\" \
  --scheduler \"${SCHEDULER}\" \
  --warmup_epochs \"${WARMUP_EPOCHS}\" \
  --min_lr \"${MIN_LR}\" \
  --tmax \"${TMAX}\" \
  --restart_period \"${RESTART_PERIOD}\" \
  --restart_mult \"${RESTART_MULT}\" \
  --step_size \"${STEP_SIZE}\" \
  --gamma \"${GAMMA}\" \
  --plateau_patience \"${PLATEAU_PATIENCE}\" \
  --plateau_factor \"${PLATEAU_FACTOR}\" \
  --onecycle_pct_start \"${ONECYCLE_PCT_START}\""

# 仅当有里程碑时再传（避免传空串）
if [[ -n "${MILESTONES}" ]]; then
  CMD="${CMD} --milestones \"${MILESTONES}\""
fi

# 可选开关
if [[ "${AMP_FLAG}" == "1" ]]; then
  CMD="${CMD} --amp"
fi
if [[ "${PE_INCLUDE_INPUT}" == "1" ]]; then
  CMD="${CMD} --pe_include_input"
fi

# ===== 执行并同时写日志 =====
if [[ "${TTY_WRAP}" == "1" ]]; then
  # 保留 tqdm 宽条
  # shellcheck disable=SC2086
  script -q -c "${CMD}" /dev/null | tee "${LOG_FILE}"
else
  # shellcheck disable=SC2086
  bash -lc "${CMD}" 2>&1 | tee "${LOG_FILE}"
fi

echo "[train] 完成。日志: ${LOG_FILE}"
echo "[train] 最优权重: ${CKPT_DIR}/ckpt_best.pt（训练过程中由脚本保存）"
