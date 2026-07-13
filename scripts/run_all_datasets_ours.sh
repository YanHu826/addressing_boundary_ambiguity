#!/usr/bin/env bash
# Run "Ours" (Theorem 1 adaptive + FixMatch + BCP) on all 4 datasets in parallel.
# Each dataset runs its ratios sequentially on a dedicated 2-GPU slot.
#
# Layout (8 H100 total):
#   GPU 0,1 -> HC18 ratios {1,2}     (~28h)
#   GPU 2,3 -> PSFH ratios {1,2}     (~28h)
#   GPU 4,5 -> BUSI ratios {1,2,3}   (~42h)
#   GPU 6,7 -> TN3K ratios {1,2,3}   (~42h)
# Wall clock: ~42h (limited by BUSI/TN3K's 3 ratios).
#
# Output: outputs/semi/checkpoints/brace_ours_<dataset>_exp<n>_abrnet/
#         outputs/logs/all_datasets/<prefix>_<timestamp>.log
#
# Purpose: validate Ours > BiPCC main claim before running ablations.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/outputs/logs/all_datasets"
mkdir -p "${LOG_DIR}"

# Theorem 1 + FixMatch + BCP — the "Ours" configuration for the main table.
COMMON="--use_strong_aug --use_bcp --wsdice_mode adaptive"

launch_dataset() {
  local dataset="$1"     # HC18 / PSFH / BUSI / TN3K
  local devices="$2"     # "0,1"
  local prefix="$3"      # e.g. brace_ours_hc18
  local timestamp
  timestamp="$(date '+%Y%m%d_%H%M%S')"
  local log_file="${LOG_DIR}/${prefix}_${timestamp}.log"
  echo "[$(date '+%F %T')] Launching ${dataset} on GPUs ${devices}"
  echo "                       prefix=${prefix}"
  echo "                       log=${log_file}"
  (
    export SEMI_BATCH_SIZE=16
    export SEMI_NUM_WORKERS=8
    export SEMI_VISIBLE_DEVICES="${devices}"
    export SEMI_EXTRA_ARGS="${COMMON}"
    export PIPELINE_CONFIGS="${dataset}:all"
    export CKPT_PREFIX="${prefix}"
    bash scripts/run_training_pipeline.sh run \
      > "${log_file}" 2>&1
  ) &
  local pid=$!
  echo "${pid}" > "${LOG_DIR}/${prefix}.pid"
  echo "                       pid=${pid}"
}

echo "================================================================"
echo "Ours-on-all-datasets launcher  start=$(date '+%F %T')"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "Common args: ${COMMON}"
echo "================================================================"

launch_dataset HC18 "0,1" "brace_ours_hc18"
launch_dataset PSFH "2,3" "brace_ours_psfh"
launch_dataset BUSI "4,5" "brace_ours_busi"
launch_dataset TN3K "6,7" "brace_ours_tn3k"

echo
echo "All 4 datasets launched. Waiting for completion (~42h)..."
echo "Tail a log: tail -f ${LOG_DIR}/brace_ours_<dataset>_<timestamp>.log"
echo

wait
echo "[$(date '+%F %T')] All 4 datasets complete"
