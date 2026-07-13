#!/usr/bin/env bash
# Rerun the missing HC18 20% rows for Table 4.
#
# Historical Table 4 labels used the old names:
#   FBWA -> current SCD branch
#   SORP -> current SOR auxiliary branch
#   CAE  -> current CoordAttention encoder
#   RCT  -> reliability-calibrated transfer
#
# The current pipeline's built-in "baseline" only disables SCD/SOR/CA, so this
# runner keeps SEMI_VARIANTS=abrnet and injects the exact component switches per
# row. RCT-off is reproduced by removing reliability weighting, boundary
# calibration, and cross-teaching.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_DIR}"
git config --global --add safe.directory "${PROJECT_DIR}" >/dev/null 2>&1 || true

TAG="${TAG:-table4_hc18_abmiss_$(date '+%m%d%H%M')}"
LOG_DIR="${PROJECT_DIR}/outputs/logs/hc18_ablation_missing"
mkdir -p "${LOG_DIR}"

COMMON_EXPORTS=(
  SKIP_GAN=1
  GAN_RESUME_FROM_LATEST=1
  RUN_TEST=1
  SKIP_EXISTING=0
  SEMI_VARIANTS=abrnet
  PIPELINE_CONFIGS=HC18:2
  SEMI_BATCH_SIZE=16
  SEMI_NUM_WORKERS=6
  SEMI_EPOCHS=200
  SEMI_AUTOTUNE=0
  SEMI_LR=1e-4
  SEMI_EVAL_TTA=1
  SEMI_AUTO_TUNE_EVAL_POSTPROCESS=1
  SEMI_VAL_INTERVAL=5
)

run_variant() {
  local row_name="$1"
  local gpu="$2"
  local extra_args="$3"
  local prefix="${TAG}_hc18_20_${row_name}"
  local log_file="${LOG_DIR}/${prefix}_$(date '+%Y%m%d_%H%M%S').log"

  echo "[$(date '+%F %T')] Launching HC18 20% ${row_name} on GPU ${gpu}"
  echo "                       prefix=${prefix}"
  echo "                       extra_args=${extra_args}"
  echo "                       log=${log_file}"
  (
    set -euo pipefail
    for assignment in "${COMMON_EXPORTS[@]}"; do
      export "${assignment}"
    done
    export SEMI_VISIBLE_DEVICES="${gpu}"
    export CKPT_PREFIX="${prefix}"
    export SEMI_EXTRA_ARGS="${extra_args}"

    {
      echo "[RUNNER] row_name=${row_name}"
      echo "[RUNNER] gpu=${gpu}"
      echo "[RUNNER] ckpt_prefix=${CKPT_PREFIX}"
      echo "[RUNNER] semi_extra_args=${SEMI_EXTRA_ARGS}"
      git status --short --branch
      git log --oneline -1
      bash scripts/run_training_pipeline.sh run
    } > "${log_file}" 2>&1
  ) &
  local pid=$!
  echo "${pid}" > "${LOG_DIR}/${prefix}.pid"
  echo "                       pid=${pid}"
}

echo "================================================================"
echo "HC18 Table 4 missing ablation runner  start=$(date '+%F %T')"
echo "tag=${TAG}"
echo "project=${PROJECT_DIR}"
echo "================================================================"

# Baseline removes FBWA/SORP/CAE/RCT.
run_variant \
  "baseline" \
  "0" \
  "--no_scd --no_sor --no_ca --reliability_floor 1 --calibration_weight 0 --cross_weight 0"

# w/o FBWA: remove only the SCD/FBWA branch.
run_variant \
  "no_fbwa" \
  "1" \
  "--no_scd"

# w/o SORP: remove only the auxiliary SOR view.
run_variant \
  "no_sorp" \
  "2" \
  "--no_sor"

# w/o CAE: remove only CoordAttention encoder blocks.
run_variant \
  "no_cae" \
  "3" \
  "--no_ca"

echo
echo "All HC18 missing ablation jobs launched. Waiting for completion..."
wait
echo "[$(date '+%F %T')] HC18 missing ablation runner complete"
