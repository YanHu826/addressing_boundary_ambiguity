#!/usr/bin/env bash
# Paper-record BRACE rerun from the d898bd1 training point.
#
# Protocol:
#   1) restore/keep the d898bd1 training logic;
#   2) train a fresh GAN/FBWA(SCD) critic for every dataset+expID;
#   3) hand each semi run its own SCD_ACTIVE_PATH to avoid cross-run races;
#   4) write all GAN/semi checkpoints, summaries and logs into one immutable
#      paper-run folder under outputs/paper_runs/${TAG};
#   5) run the same semi seed/protocol as the PSFH20=82% reference run: seed 3407,
#      batch 16, workers 6, lr 1e-4, TTA + auto postprocess, and
#      --use_strong_aug --use_bcp --wsdice_mode adaptive;
#   6) train GAN/FBWA to epoch 5000 for the paper run.
#
# Wave 1 uses all 8 GPUs for exp1/exp2 across HC18/PSFH/BUSI/TN3K.
# Wave 2 finishes the remaining BUSI/TN3K exp3 settings.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_DIR}"

git config --global --add safe.directory "${PROJECT_DIR}" 2>/dev/null || true

TAG="${TAG:-brace_paper_freshgan_$(date '+%m%d%H%M')}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/outputs/paper_runs/${TAG}}"
LAUNCH_LOG_DIR="${RUN_ROOT}/logs/launcher"
METADATA_DIR="${RUN_ROOT}/metadata"
mkdir -p "${LAUNCH_LOG_DIR}" "${METADATA_DIR}"

COMMON_ARGS="${COMMON_ARGS:---use_strong_aug --use_bcp --wsdice_mode adaptive}"

# Historical GAN seeds visible in outputs/logs/gan/*.log from the reference run.
# Keeping them fixed makes the fresh-GAN paper rerun reproducible instead of
# relying on random.randint inside airs/GAN/main.py.
gan_seed() {
  local dataset="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  local expid="$2"
  case "${dataset}:${expid}" in
    hc18:1) printf '9048' ;;
    hc18:2) printf '442' ;;
    psfh:1) printf '5403' ;;
    psfh:2) printf '188' ;;
    busi:1) printf '1239' ;;
    busi:2) printf '989' ;;
    busi:3) printf '3407' ;;
    tn3k:1) printf '4598' ;;
    tn3k:2) printf '3986' ;;
    tn3k:3) printf '3407' ;;
    *) printf '3407' ;;
  esac
}

write_metadata_once() {
  if [[ -f "${METADATA_DIR}/manifest.txt" ]]; then
    return 0
  fi
  {
    echo "tag=${TAG}"
    echo "run_root=${RUN_ROOT}"
    echo "start_time=$(date '+%F %T %Z')"
    echo "code=$(git log --oneline -1 2>/dev/null || echo unknown)"
    echo "branch=$(git branch --show-current 2>/dev/null || echo detached)"
    echo "status_begin"
    git status --short || true
    echo "status_end"
    echo "common_args=${COMMON_ARGS}"
    echo "semi_seed=3407"
    echo "semi_batch_size=16"
    echo "semi_num_workers=6"
    echo "semi_lr=1e-4"
    echo "gan_niter=5001"
    echo "gan_expected_epoch=5000"
    echo "gan_resume_from_latest=0"
    echo "phase_order=8way_exp1_exp2_all_datasets -> BUSI/TN3K_exp3"
  } > "${METADATA_DIR}/manifest.txt"

  find airs/data/splits -type f \
    \( -name 'labeled.txt' -o -name 'unlabeled.txt' -o -name 'val.txt' -o -name 'test.txt' \) \
    | sort | xargs sha256sum > "${METADATA_DIR}/split_sha256.txt"
}

run_exp() {
  local dataset="$1"
  local expid="$2"
  local gpu="$3"
  local suffix="$4"
  local seed prefix log_file active_path
  seed="$(gan_seed "${dataset}" "${expid}")"
  prefix="${TAG}_${suffix}"
  log_file="${LAUNCH_LOG_DIR}/${prefix}_$(date '+%Y%m%d_%H%M%S').log"
  active_path="${RUN_ROOT}/pretrain/scd_active/${prefix}/netD_epoch_5000.pth"

  echo "[$(date '+%F %T')] START ${dataset}:${expid} gpu=${gpu} prefix=${prefix} gan_seed=${seed} log=${log_file}"
  (
    set -euo pipefail
    export OUTPUT_ROOT="${RUN_ROOT}"
    export GAN_OUTPUT_ROOT="${RUN_ROOT}/gan"
    export SEMI_CHECKPOINT_ROOT="${RUN_ROOT}/semi/checkpoints"
    export SEMI_RESULT_ROOT="${RUN_ROOT}/semi/results"
    export SUMMARY_ROOT="${RUN_ROOT}/summary/${prefix}"
    export TEST_SUMMARY_FILE="${RUN_ROOT}/summary/${prefix}/test_results.md"
    export LOG_ROOT="${RUN_ROOT}/logs/pipeline/${prefix}"
    export PRETRAIN_ROOT="${RUN_ROOT}/pretrain"
    export SCD_ARCHIVE_ROOT="${RUN_ROOT}/pretrain/scd_archive"
    export SCD_ACTIVE_PATH="${active_path}"

    export PIPELINE_CONFIGS="${dataset}:${expid}"
    export CKPT_PREFIX="${prefix}"
    export SEMI_VARIANTS="abrnet"
    export RUN_TEST=1
    export SKIP_GAN=0
    export SKIP_EXISTING=0
    export GAN_RESUME_FROM_LATEST=0
    export GAN_NITER=5001
    export GAN_EXPECTED_EPOCH=5000
    export GAN_BATCH_SIZE=32
    export GAN_WORKERS=12
    export GAN_PRECISION=fp32
    export GAN_TF32=1
    export GAN_EXTRA_ARGS="--manualSeed ${seed}"

    export SEMI_BATCH_SIZE=16
    export SEMI_NUM_WORKERS=6
    export SEMI_AUTOTUNE=0
    export SEMI_EPOCHS=200
    export SEMI_LR=1e-4
    export SEMI_VISIBLE_DEVICES="${gpu}"
    export GAN_VISIBLE_DEVICES="${gpu}"
    export SEMI_SIGNAL_SEED=3407
    export SEMI_EXTRA_ARGS="${COMMON_ARGS}"
    export SEMI_EVAL_TTA=1
    export SEMI_AUTO_TUNE_EVAL_POSTPROCESS=1
    export SEMI_SAVE_BEST_TTA=0
    export SEMI_SAVE_RELIABILITY_MAPS=0

    bash scripts/run_training_pipeline.sh run

    test_log_dir="${RUN_ROOT}/logs/pipeline/${prefix}/semi"
    shopt -s nullglob
    test_logs=("${test_log_dir}/${prefix}_"*"_test.log")
    if [[ ${#test_logs[@]} -eq 0 ]]; then
      echo "[CHECK][ERROR] ${prefix} produced no test logs in ${test_log_dir}" >&2
      exit 1
    fi
    for test_log in "${test_logs[@]}"; do
      grep -q 'Valid Result' "${test_log}"
    done
    echo "[CHECK] ${prefix} produced ${#test_logs[@]} test log(s)"
  ) > "${log_file}" 2>&1
  echo "[$(date '+%F %T')] DONE ${dataset}:${expid} prefix=${prefix}"
}

launch_exp() {
  run_exp "$@" &
  PIDS+=("$!")
}

run_phase() {
  local phase_name="$1"
  shift
  PIDS=()
  echo "================================================================"
  echo "[$(date '+%F %T')] PHASE START ${phase_name}"
  echo "================================================================"
  while [[ $# -gt 0 ]]; do
    launch_exp "$1" "$2" "$3" "$4"
    shift 4
  done
  local pid
  for pid in "${PIDS[@]}"; do
    wait "${pid}"
  done
  echo "[$(date '+%F %T')] PHASE DONE ${phase_name}"
}

write_metadata_once

echo "================================================================"
echo "BRACE paper fresh-GAN rerun"
echo "tag=${TAG}"
echo "run_root=${RUN_ROOT}"
echo "code=$(git log --oneline -1 2>/dev/null || echo unknown)"
echo "common_args=${COMMON_ARGS}"
echo "================================================================"

# Wave 1: occupy all 8 GPUs.  Each dataset+expID has its own fresh GAN and
# experiment-local SCD_ACTIVE_PATH, so exp1/exp2 cannot read each other's critic.
run_phase "exp1_exp2_all_datasets_8way" \
  HC18 1 0 hc18_exp1 \
  HC18 2 1 hc18_exp2 \
  PSFH 1 2 psfh_exp1 \
  PSFH 2 3 psfh_exp2 \
  BUSI 1 4 busi_exp1 \
  BUSI 2 5 busi_exp2 \
  TN3K 1 6 tn3k_exp1 \
  TN3K 2 7 tn3k_exp2

# Wave 2: remaining BUSI/TN3K 1/2 settings for the main table.
run_phase "exp3_half_remaining" \
  BUSI 3 4 busi_exp3 \
  TN3K 3 5 tn3k_exp3

{
  echo "# BRACE paper fresh-GAN summaries"
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

echo "[$(date '+%F %T')] ALL DONE run_root=${RUN_ROOT}"
echo "Summary: ${RUN_ROOT}/summary/all_test_results_extracted.md"
