#!/usr/bin/env bash
# Run 4 PSFH WSDice ablations in parallel within one 8-GPU pod.
# Each ablation pins to 2 GPUs; total 8 GPUs used.
#
# Layout:
#   GPU 0,1 -> abl_thm1    (--wsdice_mode adaptive)        ★ Ours
#   GPU 2,3 -> abl_fixed   (--wsdice_mode fixed)           baseline
#   GPU 4,5 -> abl_invfreq (--wsdice_mode inverse_freq)    baseline
#   GPU 6,7 -> abl_effnum  (--wsdice_mode effective_number) baseline (Cui CVPR'19)
#
# Usage from scripts/run_training_pipeline.sh or a generic shell:
#   LAUNCH_CMD="bash scripts/run_parallel_ablations.sh" \
#     GPU_COUNT=8 SEMI_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
#     bash scripts/run_training_pipeline.sh run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/outputs/logs/ablation"
mkdir -p "${LOG_DIR}"

COMMON_BASE="--use_strong_aug --use_bcp"
PSFH_CONFIGS="${PSFH_CONFIGS:-PSFH:1 PSFH:2}"

launch_ablation() {
  local prefix="$1"
  local devices="$2"
  local extra="$3"
  local timestamp
  timestamp="$(date '+%Y%m%d_%H%M%S')"
  local log_file="${LOG_DIR}/${prefix}_${timestamp}.log"
  echo "[$(date '+%F %T')] Launching ${prefix}  GPUs=${devices}  extra=\"${extra}\""
  echo "                       log=${log_file}"
  (
    export SEMI_BATCH_SIZE=16
    export SEMI_NUM_WORKERS=8
    export SEMI_VISIBLE_DEVICES="${devices}"
    export SEMI_EXTRA_ARGS="${COMMON_BASE} ${extra}"
    export PIPELINE_CONFIGS="${PSFH_CONFIGS}"
    export CKPT_PREFIX="${prefix}"
    export SEMI_AUTOTUNE="${SEMI_AUTOTUNE:-0}"
    bash scripts/run_training_pipeline.sh run \
      > "${log_file}" 2>&1
  ) &
  local pid=$!
  echo "${pid}" > "${LOG_DIR}/${prefix}.pid"
  echo "                       pid=${pid}"
}

echo "================================================================"
echo "Parallel ablation launcher  start=$(date '+%F %T')"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "PSFH_CONFIGS=${PSFH_CONFIGS}"
echo "================================================================"

launch_ablation "abl_thm1"    "0,1" "--wsdice_mode adaptive"
launch_ablation "abl_fixed"   "2,3" "--wsdice_mode fixed"
launch_ablation "abl_invfreq" "4,5" "--wsdice_mode inverse_freq"
launch_ablation "abl_effnum"  "6,7" "--wsdice_mode effective_number --wsdice_eff_beta 0.9999"

echo
echo "All 4 ablations launched. Waiting for completion..."
echo "Tail any log with: tail -f ${LOG_DIR}/<prefix>_<timestamp>.log"
echo

wait
echo "[$(date '+%F %T')] All 4 ablations complete"
