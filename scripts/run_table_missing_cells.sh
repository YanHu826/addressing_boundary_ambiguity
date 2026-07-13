#!/usr/bin/env bash
# Fill the remaining manuscript cells that are still shown as "--".
#
# Current gaps:
#   1) Table 4 TN3K 1/8 w/o CAE: prior training finished, but the test log was
#      interrupted before emitting the final Valid Result. Re-run test from the
#      recorded checkpoint when it is available; otherwise train the same row.
#   2) PSFHS diagnostic BRACE default at 10%: run the current default objective
#      without the legacy adaptive PSFHS WSDice switch, matching the recorded
#      20% "BRACE default (w/o PSFHS-WSDice)" row.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_DIR}"
git config --global --add safe.directory "${PROJECT_DIR}" >/dev/null 2>&1 || true

TAG="${TAG:-table_missing_cells_$(date '+%m%d%H%M')}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/outputs/table_missing_cells/${TAG}}"
LAUNCH_LOG_DIR="${RUN_ROOT}/logs/launcher"
mkdir -p "${LAUNCH_LOG_DIR}"

COMMON_EXPORTS=(
  AUTO_INSTALL_DEPS=0
  FORCE_INSTALL_DEPS=0
  SKIP_GAN=1
  GAN_RESUME_FROM_LATEST=1
  RUN_TEST=1
  SEMI_VARIANTS=abrnet
  SEMI_BATCH_SIZE=16
  SEMI_NUM_WORKERS=6
  SEMI_EPOCHS=200
  SEMI_AUTOTUNE=0
  SEMI_LR=1e-4
  SEMI_EVAL_TTA=1
  SEMI_AUTO_TUNE_EVAL_POSTPROCESS=1
  SEMI_SAVE_BEST_TTA=0
  SEMI_SAVE_RELIABILITY_MAPS=0
  SEMI_VAL_INTERVAL=5
)

write_manifest() {
  mkdir -p "${RUN_ROOT}/metadata"
  {
    echo "tag=${TAG}"
    echo "run_root=${RUN_ROOT}"
    echo "start_time=$(date '+%F %T %Z')"
    echo "code=$(git log --oneline -1 2>/dev/null || echo unknown)"
    echo "branch=$(git branch --show-current 2>/dev/null || echo detached)"
    echo "status_begin"
    git status --short --branch || true
    echo "status_end"
    echo "missing_cells=TN3K_1_8_wo_CAE_test,PSFHS_10_default"
    echo "tn3k_no_cae_existing_ckpt=${TN3K_NO_CAE_CKPT:-}"
  } > "${RUN_ROOT}/metadata/manifest.txt"
}

run_case() {
  local row_name="$1"
  local gpu="$2"
  local pipeline_configs="$3"
  local ckpt_prefix="$4"
  local extra_args="$5"
  local skip_existing="$6"
  local log_file="${LAUNCH_LOG_DIR}/${row_name}_$(date '+%Y%m%d_%H%M%S').log"

  echo "[$(date '+%F %T')] Launching ${row_name} on GPU ${gpu}"
  echo "                       configs=${pipeline_configs}"
  echo "                       ckpt_prefix=${ckpt_prefix}"
  echo "                       skip_existing=${skip_existing}"
  echo "                       extra_args=${extra_args:-<none>}"
  echo "                       log=${log_file}"
  (
    set -euo pipefail
    for assignment in "${COMMON_EXPORTS[@]}"; do
      export "${assignment}"
    done

    export PIPELINE_CONFIGS="${pipeline_configs}"
    export CKPT_PREFIX="${ckpt_prefix}"
    export SKIP_EXISTING="${skip_existing}"
    export SEMI_VISIBLE_DEVICES="${gpu}"
    export GAN_VISIBLE_DEVICES="${gpu}"
    export SEMI_EXTRA_ARGS="${extra_args}"

    export LOG_ROOT="${RUN_ROOT}/logs/pipeline/${row_name}"
    export SUMMARY_ROOT="${RUN_ROOT}/summary/${row_name}"
    export TEST_SUMMARY_FILE="${SUMMARY_ROOT}/test_results.md"
    export PRETRAIN_ROOT="${RUN_ROOT}/pretrain/${row_name}"
    export SCD_ARCHIVE_ROOT="${PRETRAIN_ROOT}/scd_archive"
    export SCD_ACTIVE_PATH="${PRETRAIN_ROOT}/scd_active/netD_epoch_3000.pth"

    {
      echo "[RUNNER] row_name=${row_name}"
      echo "[RUNNER] gpu=${gpu}"
      echo "[RUNNER] pipeline_configs=${PIPELINE_CONFIGS}"
      echo "[RUNNER] ckpt_prefix=${CKPT_PREFIX}"
      echo "[RUNNER] skip_existing=${SKIP_EXISTING}"
      echo "[RUNNER] semi_extra_args=${SEMI_EXTRA_ARGS:-<none>}"
      git status --short --branch
      git log --oneline -1
      bash scripts/run_training_pipeline.sh run
    } > "${log_file}" 2>&1
  ) &
  local pid=$!
  echo "${pid}" > "${RUN_ROOT}/logs/launcher/${row_name}.pid"
  echo "                       pid=${pid}"
}

TN3K_NO_CAE_PREFIX="scrarcc_abgen_06151927_tn3k18_no_cae"
TN3K_NO_CAE_CKPT="${PROJECT_DIR}/outputs/semi/checkpoints/${TN3K_NO_CAE_PREFIX}_tn3k_exp1_abrnet/best.pth"
TN3K_SKIP_EXISTING=0
TN3K_CKPT_PREFIX="${TAG}_tn3k18_no_cae"
if [[ -f "${TN3K_NO_CAE_CKPT}" ]]; then
  TN3K_SKIP_EXISTING=1
  TN3K_CKPT_PREFIX="${TN3K_NO_CAE_PREFIX}"
fi

write_manifest

echo "================================================================"
echo "Table missing-cell runner  start=$(date '+%F %T')"
echo "tag=${TAG}"
echo "run_root=${RUN_ROOT}"
echo "project=${PROJECT_DIR}"
echo "================================================================"

run_case \
  "tn3k18_no_cae_test" \
  "0" \
  "TN3K:1" \
  "${TN3K_CKPT_PREFIX}" \
  "--no_ca" \
  "${TN3K_SKIP_EXISTING}"

run_case \
  "psfh10_default" \
  "1" \
  "PSFH:1" \
  "${TAG}_psfh10_default" \
  "" \
  "0"

echo
echo "All missing-cell jobs launched. Waiting for completion..."
wait

mkdir -p "${RUN_ROOT}/summary"
{
  echo "# Table missing-cell summaries"
  echo
  echo "tag=${TAG}"
  echo "run_root=${RUN_ROOT}"
  echo "finished=$(date '+%F %T %Z')"
  echo
  find "${RUN_ROOT}/logs/pipeline" -path '*/semi/*_test.log' -type f | sort | while read -r f; do
    echo "## ${f#${RUN_ROOT}/}"
    grep 'Valid Result' "${f}" | tail -1 || true
    grep '\[ValidDetail' "${f}" | tail -4 || true
    echo
  done
} > "${RUN_ROOT}/summary/all_test_results_extracted.md"

echo "[$(date '+%F %T')] Table missing-cell runner complete"
echo "Summary: ${RUN_ROOT}/summary/all_test_results_extracted.md"
