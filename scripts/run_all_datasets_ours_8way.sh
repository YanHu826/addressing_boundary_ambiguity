#!/usr/bin/env bash
# Maximally parallel runner: 8 single-GPU sub-shells, all 10 experiments.
# Wall clock target: ~14h (HC18 single-exp bottleneck) vs ~30h for 4×2-GPU.
#
# Layout (8 H100 total, 1 each):
#   GPU 0 -> HC18 exp1  (GAN 1.8h + Semi 12h ≈ 14h)
#   GPU 1 -> HC18 exp2  (GAN 1.8h + Semi 12h ≈ 14h)
#   GPU 2 -> PSFH exp1  (Semi 10h, GAN reused)
#   GPU 3 -> PSFH exp2  (Semi 10h)
#   GPU 4 -> BUSI all 3 (1.7h × 3 ≈ 5h)
#   GPU 5 -> TN3K exp1  (Semi 8h)
#   GPU 6 -> TN3K exp2  (GAN 0.1h + Semi 8h)
#   GPU 7 -> TN3K exp3  (GAN 0.1h + Semi 8h)
#
# Why 1 GPU per exp instead of 2: removes DataParallel scatter/gather overhead
# and keeps every GPU busy (no idle GPU during GAN pretrain).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/outputs/logs/all_datasets"
mkdir -p "${LOG_DIR}"

# Theorem 1 + FixMatch + BCP — the "Ours" main-table configuration.
COMMON="--use_strong_aug --use_bcp --wsdice_mode adaptive"

launch_exp() {
  local dataset="$1"
  local config="$2"     # "1", "2", "3", or "all"
  local gpu="$3"        # single GPU id, e.g. "0"
  local prefix="$4"
  local timestamp
  timestamp="$(date '+%Y%m%d_%H%M%S')"
  local log_file="${LOG_DIR}/${prefix}_${timestamp}.log"
  echo "[$(date '+%F %T')] Launching ${dataset}:${config} on GPU ${gpu}"
  echo "                       prefix=${prefix}"
  echo "                       log=${log_file}"
  (
    export SEMI_BATCH_SIZE=16
    export SEMI_NUM_WORKERS=8
    export SEMI_AUTOTUNE=0
    export SEMI_VISIBLE_DEVICES="${gpu}"
    export SEMI_EXTRA_ARGS="${COMMON}"
    export PIPELINE_CONFIGS="${dataset}:${config}"
    export CKPT_PREFIX="${prefix}"
    bash scripts/run_training_pipeline.sh run \
      > "${log_file}" 2>&1
  ) &
  local pid=$!
  echo "${pid}" > "${LOG_DIR}/${prefix}.pid"
  echo "                       pid=${pid}"
}

echo "================================================================"
echo "8-way parallel ablation launcher  start=$(date '+%F %T')"
echo "Target wall clock: ~14h (HC18 single-exp bottleneck)"
echo "================================================================"

# HC18 — 2 ratios, each on its own GPU because each needs ~14h
launch_exp HC18 "1" "0" "brace_ours_hc18_exp1"
launch_exp HC18 "2" "1" "brace_ours_hc18_exp2"

# PSFH — 2 ratios, each on its own GPU
launch_exp PSFH "1" "2" "brace_ours_psfh_exp1"
launch_exp PSFH "2" "3" "brace_ours_psfh_exp2"

# BUSI — 3 ratios serial on 1 GPU (each only ~1.7h, no benefit splitting)
launch_exp BUSI "all" "4" "brace_ours_busi"

# TN3K — 3 ratios, each on own GPU (each ~8h, splitting saves 16h)
launch_exp TN3K "1" "5" "brace_ours_tn3k_exp1"
launch_exp TN3K "2" "6" "brace_ours_tn3k_exp2"
launch_exp TN3K "3" "7" "brace_ours_tn3k_exp3"

echo
echo "All 8 sub-shells launched. Wall clock target: ~14h"
echo

wait
echo "[$(date '+%F %T')] All experiments complete"
