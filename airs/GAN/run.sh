#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET="${DATASET:-tn3k}"
EXPID="${EXPID:-1}"
NGPU="${NGPU:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-12}"
NITER="${NITER:-3001}"
LR_D="${LR_D:-0.00005}"
LR_G="${LR_G:-0.00005}"
PRECISION="${PRECISION:-auto}"
TF32="${TF32:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"
SAVE_EVERY="${SAVE_EVERY:-500}"
WARMUP_GEN_ITERATIONS="${WARMUP_GEN_ITERATIONS:-5}"
WARMUP_DITERS="${WARMUP_DITERS:-20}"
EXTRA_DITERS_EVERY="${EXTRA_DITERS_EVERY:-0}"
USE_GP="${USE_GP:-1}"
USE_ADAM="${USE_ADAM:-1}"
LOG_FILE="${LOG_FILE:-train_gan.log}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

read -r -a EXTRA_ARGS_ARRAY <<< "${EXTRA_ARGS}"

cmd=(
  "${PYTHON_BIN}" -u main.py
  --dataset "${DATASET}"
  --expID "${EXPID}"
  --cuda
  --ngpu "${NGPU}"
  --batchSize "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --niter "${NITER}"
  --lrD "${LR_D}"
  --lrG "${LR_G}"
  --precision "${PRECISION}"
  --log_interval "${LOG_INTERVAL}"
  --save_every "${SAVE_EVERY}"
  --warmup_gen_iterations "${WARMUP_GEN_ITERATIONS}"
  --warmup_diters "${WARMUP_DITERS}"
  --extra_diters_every "${EXTRA_DITERS_EVERY}"
)

if [[ "${TF32}" == "1" ]]; then
  cmd+=(--tf32)
fi
if [[ "${USE_GP}" == "1" ]]; then
  cmd+=(--gradient_penalty)
fi
if [[ "${USE_ADAM}" == "1" ]]; then
  cmd+=(--adam)
fi
if [[ ${#EXTRA_ARGS_ARRAY[@]} -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS_ARRAY[@]}")
fi

nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}" > "${LOG_FILE}" 2>&1 &
