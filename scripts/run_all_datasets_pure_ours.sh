#!/usr/bin/env bash
# Pure Ours ablation: only adaptive WSDice + EMA teacher + cross-teaching.
# Removes FixMatch strong-aug, BCP injection, and WGAN-GP DSR.
# Goal: validate "Less is More" claim — can a single closed-form loss
# match the performance of a multi-module SSL pipeline?
#
# Layout (8 H100 total, 1 each):
#   GPU 0 -> HC18 exp1  (Pure)
#   GPU 1 -> HC18 exp2  (Pure)
#   GPU 2 -> PSFH exp1  (Pure)
#   GPU 3 -> PSFH exp2  (Pure)
#   GPU 4 -> BUSI all 3 (Pure, serial)
#   GPU 5 -> TN3K exp1  (Pure)
#   GPU 6 -> TN3K exp2  (Pure)
#   GPU 7 -> TN3K exp3  (Pure)
#
# Compare to brace_ours_* prefixes (Full Ours) for the Less-is-More claim.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/outputs/logs/all_datasets"
mkdir -p "${LOG_DIR}"

# Pure: only adaptive WSDice. Disable FixMatch (--use_strong_aug omitted),
# BCP (--use_bcp omitted), and WGAN-GP shape critic (--adv_weight 0).
# Keep EMA teacher (always on), cross-teaching (default), boundary loss (default).
COMMON="--wsdice_mode adaptive --adv_weight 0"

launch_exp() {
  local dataset="$1"
  local config="$2"
  local gpu="$3"
  local prefix="$4"
  local timestamp
  timestamp="$(date '+%Y%m%d_%H%M%S')"
  local log_file="${LOG_DIR}/${prefix}_${timestamp}.log"
  echo "[$(date '+%F %T')] Launching ${dataset}:${config} on GPU ${gpu}"
  echo "                       prefix=${prefix}"
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
}

echo "================================================================"
echo "Pure Ours ablation (Less-is-More claim)  start=$(date '+%F %T')"
echo "Common args: ${COMMON}"
echo "================================================================"

launch_exp HC18 "1" "0" "brace_pure_hc18_exp1"
launch_exp HC18 "2" "1" "brace_pure_hc18_exp2"
launch_exp PSFH "1" "2" "brace_pure_psfh_exp1"
launch_exp PSFH "2" "3" "brace_pure_psfh_exp2"
launch_exp BUSI "all" "4" "brace_pure_busi"
launch_exp TN3K "1" "5" "brace_pure_tn3k_exp1"
launch_exp TN3K "2" "6" "brace_pure_tn3k_exp2"
launch_exp TN3K "3" "7" "brace_pure_tn3k_exp3"

echo
echo "All 8 Pure Ours sub-shells launched. Wall clock target: ~14h"
echo

wait
echo "[$(date '+%F %T')] All Pure Ours experiments complete"
