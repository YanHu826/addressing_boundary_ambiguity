#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-help}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}}"
DATASET="${DATASET:-TN3K}"
EXPID="${EXPID:-1}"
GPUS="${GPUS:-0}"
CKPT_NAME="${CKPT_NAME:-brace_${DATASET,,}_exp${EXPID}}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-1e-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

print_help() {
  cat <<EOF
Usage:
  DATA_ROOT=/path/to/workspace DATASET=TN3K EXPID=1 bash airs/semi/run.sh train
  DATA_ROOT=/path/to/workspace DATASET=TN3K EXPID=1 bash airs/semi/run.sh test

Environment variables:
  DATA_ROOT     Root path containing the dataset folders expected by the loaders.
  DATASET       BUSI, TN3K, PSFH, HC18, or another supported dataset.
  EXPID         Low-label split identifier.
  GPUS          GPU ids passed to main.py.
  CKPT_NAME     Checkpoint name.
  EXTRA_ARGS    Additional arguments passed through to main.py.
EOF
}

run_main() {
  local manner="$1"
  local -a extra=()
  if [[ -n "${EXTRA_ARGS}" ]]; then
    read -r -a extra <<< "${EXTRA_ARGS}"
  fi

  cd "${PROJECT_DIR}/airs/semi/code"
  exec "${PYTHON_BIN}" main.py \
    --manner "${manner}" \
    --dataset "${DATASET}" \
    --expID "${EXPID}" \
    --GPUs "${GPUS}" \
    --root "${DATA_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --nEpoch "${EPOCHS}" \
    --lr "${LR}" \
    --ckpt_name "${CKPT_NAME}" \
    "${extra[@]}"
}

case "${MODE}" in
  train)
    run_main semi
    ;;
  test)
    run_main test
    ;;
  full)
    run_main full
    ;;
  help|-h|--help)
    print_help
    ;;
  *)
    printf 'Unknown mode: %s\n' "${MODE}" >&2
    print_help >&2
    exit 1
    ;;
esac
