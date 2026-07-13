#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"

_GITDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
git -C "${_GITDIR}" config --global --add safe.directory "${_GITDIR}" 2>/dev/null || true
echo "[PIPELINE] Using code at: $(git -C "${_GITDIR}" log --oneline -1 2>/dev/null || echo unknown)"

PROJECT_DIR="${PROJECT_DIR:-${PROJECT_DIR_DEFAULT}}"
PROJECT_NAME="${PROJECT_NAME:-$(basename "${PROJECT_DIR}")}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${PROJECT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${PROJECT_DIR}/requirements.txt}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${WORKSPACE_ROOT}/.cache/pip}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"
GAN_OUTPUT_ROOT="${GAN_OUTPUT_ROOT:-${OUTPUT_ROOT}/gan}"
SEMI_CHECKPOINT_ROOT="${SEMI_CHECKPOINT_ROOT:-${OUTPUT_ROOT}/semi/checkpoints}"
SEMI_RESULT_ROOT="${SEMI_RESULT_ROOT:-${OUTPUT_ROOT}/semi/results}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${OUTPUT_ROOT}/summary}"
TEST_SUMMARY_FILE="${TEST_SUMMARY_FILE:-${SUMMARY_ROOT}/test_results.md}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-${PROJECT_DIR}/runtime/pretrain}"
BACKBONE_DEFAULT_PATH="${PROJECT_DIR}/airs/semi/code/pretrain/backbone/resnet34.pth"
BACKBONE_FALLBACK_PATH="${PRETRAIN_ROOT}/backbone/resnet34.pth"
if [[ -n "${BACKBONE_PATH:-}" ]]; then
  BACKBONE_PATH="${BACKBONE_PATH}"
elif [[ -f "${BACKBONE_DEFAULT_PATH}" ]]; then
  BACKBONE_PATH="${BACKBONE_DEFAULT_PATH}"
else
  BACKBONE_PATH="${BACKBONE_FALLBACK_PATH}"
fi
SCD_ARCHIVE_ROOT="${SCD_ARCHIVE_ROOT:-${PRETRAIN_ROOT}/scd}"
SCD_ACTIVE_PATH="${SCD_ACTIVE_PATH:-${SCD_ARCHIVE_ROOT}/active/netD_epoch_10000.pth}"

AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-1}"
FORCE_INSTALL_DEPS="${FORCE_INSTALL_DEPS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
SKIP_GAN="${SKIP_GAN:-0}"
RUN_TEST="${RUN_TEST:-1}"

GAN_BATCH_SIZE_WAS_SET="${GAN_BATCH_SIZE+x}"
GAN_WORKERS_WAS_SET="${GAN_WORKERS+x}"
SEMI_BATCH_SIZE_WAS_SET="${SEMI_BATCH_SIZE+x}"
SEMI_NUM_WORKERS_WAS_SET="${SEMI_NUM_WORKERS+x}"
SEMI_LR_WAS_SET="${SEMI_LR+x}"

PIPELINE_CONFIGS="${PIPELINE_CONFIGS:-PAPER_ALL}"
CKPT_PREFIX="${CKPT_PREFIX:-brace}"
# When RUN_ABLATION=1, run all variants; otherwise only run "full"
RUN_ABLATION="${RUN_ABLATION:-0}"
if [[ "${RUN_ABLATION}" == "1" ]]; then
  SEMI_VARIANTS="${SEMI_VARIANTS:-abrnet abrnet_wo_boundary abrnet_wo_anatomy abrnet_wo_reliability baseline}"
else
  SEMI_VARIANTS="${SEMI_VARIANTS:-abrnet}"
fi

SEMI_VISIBLE_DEVICES="${SEMI_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
GAN_VISIBLE_DEVICES_DEFAULT="${SEMI_VISIBLE_DEVICES%%,*}"
GAN_VISIBLE_DEVICES="${GAN_VISIBLE_DEVICES:-${GAN_VISIBLE_DEVICES_DEFAULT}}"

GAN_NGPU="${GAN_NGPU:-1}"
GAN_BATCH_SIZE="${GAN_BATCH_SIZE:-32}"
GAN_WORKERS="${GAN_WORKERS:-12}"
GAN_NITER="${GAN_NITER:-3001}"
GAN_EXPECTED_EPOCH="${GAN_EXPECTED_EPOCH:-3000}"
GAN_RESUME_FROM_LATEST="${GAN_RESUME_FROM_LATEST:-1}"
GAN_LR_D="${GAN_LR_D:-0.00005}"
GAN_LR_G="${GAN_LR_G:-0.00005}"
GAN_LAMBDA_ADV="${GAN_LAMBDA_ADV:-0.1}"
GAN_LAMBDA_EDGE="${GAN_LAMBDA_EDGE:-1.0}"
GAN_LAMBDA_FM="${GAN_LAMBDA_FM:-1.0}"
GAN_LAMBDA_MASK="${GAN_LAMBDA_MASK:-2.0}"
GAN_PRECISION="${GAN_PRECISION:-fp32}"
GAN_TF32="${GAN_TF32:-1}"
GAN_WARMUP_GEN_ITERATIONS="${GAN_WARMUP_GEN_ITERATIONS:-5}"
GAN_WARMUP_DITERS="${GAN_WARMUP_DITERS:-20}"
GAN_EXTRA_DITERS_EVERY="${GAN_EXTRA_DITERS_EVERY:-0}"
GAN_SAVE_EVERY="${GAN_SAVE_EVERY:-500}"
GAN_LOG_INTERVAL="${GAN_LOG_INTERVAL:-25}"
GAN_AUTOTUNE="${GAN_AUTOTUNE:-0}"
GAN_AUTOTUNE_STEPS="${GAN_AUTOTUNE_STEPS:-6}"
GAN_AUTOTUNE_WARMUP_STEPS="${GAN_AUTOTUNE_WARMUP_STEPS:-2}"
GAN_AUTOTUNE_BATCH_CANDIDATES="${GAN_AUTOTUNE_BATCH_CANDIDATES:-32,48,64}"
GAN_AUTOTUNE_WORKER_CANDIDATES="${GAN_AUTOTUNE_WORKER_CANDIDATES:-8,12,16}"

SEMI_RATIO="${SEMI_RATIO:-10}"
SEMI_BATCH_SIZE="${SEMI_BATCH_SIZE:-}"
SEMI_NUM_WORKERS="${SEMI_NUM_WORKERS:-}"
SEMI_EPOCHS="${SEMI_EPOCHS:-}"
SEMI_LR="${SEMI_LR:-}"
SEMI_WEIGHT_DECAY="${SEMI_WEIGHT_DECAY:-1e-5}"
SEMI_POWER="${SEMI_POWER:-0.9}"
SEMI_EPS="${SEMI_EPS:-1e-8}"
SEMI_MT="${SEMI_MT:-0.99}"
SEMI_BAND="${SEMI_BAND:-3}"
SEMI_AUX_SUPERVISION_WEIGHT="${SEMI_AUX_SUPERVISION_WEIGHT:-0.5}"
SEMI_BOUNDARY_WEIGHT="${SEMI_BOUNDARY_WEIGHT:-0.2}"
SEMI_PSEUDO_MAIN_WEIGHT="${SEMI_PSEUDO_MAIN_WEIGHT:-1.0}"
SEMI_PSEUDO_AUX_WEIGHT="${SEMI_PSEUDO_AUX_WEIGHT:-0.3}"
SEMI_SOR_WEIGHT="${SEMI_SOR_WEIGHT:-0.05}"
SEMI_EDGE_WEIGHT="${SEMI_EDGE_WEIGHT:-0.05}"
SEMI_ADV_WEIGHT="${SEMI_ADV_WEIGHT:-0.05}"
SEMI_FM_WEIGHT="${SEMI_FM_WEIGHT:-0.5}"
SEMI_TEACHER_CONSISTENCY_WEIGHT="${SEMI_TEACHER_CONSISTENCY_WEIGHT:-0.5}"
SEMI_CONSISTENCY_RAMPUP="${SEMI_CONSISTENCY_RAMPUP:-40}"
SEMI_TEACHER_RAMPUP="${SEMI_TEACHER_RAMPUP:-}"
SEMI_STRUCTURE_RAMPUP="${SEMI_STRUCTURE_RAMPUP:-}"
SEMI_SCD_RAMPUP="${SEMI_SCD_RAMPUP:-}"
SEMI_PS_COPY_PASTE_PROB="${SEMI_PS_COPY_PASTE_PROB:-}"
SEMI_SCD_START_EPOCH="${SEMI_SCD_START_EPOCH:-}"
SEMI_SCD_UPDATE_INTERVAL="${SEMI_SCD_UPDATE_INTERVAL:-2}"
SEMI_CALIBRATION_WEIGHT="${SEMI_CALIBRATION_WEIGHT:-0.1}"
SEMI_RELIABILITY_FLOOR="${SEMI_RELIABILITY_FLOOR:-}"
SEMI_SOR_ERASE="${SEMI_SOR_ERASE:-0.2}"
SEMI_VAL_INTERVAL="${SEMI_VAL_INTERVAL:-5}"
SEMI_DATALOADER_START_METHOD="${SEMI_DATALOADER_START_METHOD:-auto}"
SEMI_EVAL_TTA="${SEMI_EVAL_TTA:-}"
SEMI_EVAL_THRESHOLD="${SEMI_EVAL_THRESHOLD:-}"
SEMI_EVAL_MIN_AREA="${SEMI_EVAL_MIN_AREA:-}"
SEMI_EVAL_POSTPROCESS_MODE="${SEMI_EVAL_POSTPROCESS_MODE:-}"
SEMI_AUTO_TUNE_EVAL_POSTPROCESS="${SEMI_AUTO_TUNE_EVAL_POSTPROCESS:-}"
SEMI_SAVE_BEST_TTA="${SEMI_SAVE_BEST_TTA:-0}"
SEMI_SAVE_RELIABILITY_MAPS="${SEMI_SAVE_RELIABILITY_MAPS:-0}"
SEMI_SAVE_RELIABILITY_LIMIT="${SEMI_SAVE_RELIABILITY_LIMIT:-}"
SEMI_PRECISION="${SEMI_PRECISION:-auto}"
SEMI_TF32="${SEMI_TF32:-1}"
SEMI_AUTOTUNE="${SEMI_AUTOTUNE:-1}"
SEMI_AUTOTUNE_MODE="${SEMI_AUTOTUNE_MODE:-signal}"
SEMI_AUTOTUNE_METRIC="${SEMI_AUTOTUNE_METRIC:-F1}"
SEMI_AUTOTUNE_STEPS="${SEMI_AUTOTUNE_STEPS:-6}"
SEMI_AUTOTUNE_WARMUP_STEPS="${SEMI_AUTOTUNE_WARMUP_STEPS:-2}"
SEMI_RESULT_AUTOTUNE_EPOCHS="${SEMI_RESULT_AUTOTUNE_EPOCHS:-}"
SEMI_AUTOTUNE_BATCH_CANDIDATES="${SEMI_AUTOTUNE_BATCH_CANDIDATES:-}"
SEMI_AUTOTUNE_WORKER_CANDIDATES="${SEMI_AUTOTUNE_WORKER_CANDIDATES:-}"
SEMI_HP_TUNE_SPEC="${SEMI_HP_TUNE_SPEC:-}"
SEMI_SIGNAL_STEPS="${SEMI_SIGNAL_STEPS:-120}"
SEMI_SIGNAL_WARMUP_STEPS="${SEMI_SIGNAL_WARMUP_STEPS:-20}"
SEMI_SIGNAL_SEED="${SEMI_SIGNAL_SEED:-3407}"
SEMI_SIGNAL_LR_CANDIDATES="${SEMI_SIGNAL_LR_CANDIDATES:-}"
SEMI_SIGNAL_LR_MULTIPLIERS="${SEMI_SIGNAL_LR_MULTIPLIERS:-0.70,0.85,1.00,1.15}"

GAN_EXTRA_ARGS="${GAN_EXTRA_ARGS:-}"
SEMI_EXTRA_ARGS="${SEMI_EXTRA_ARGS:-}"

read -r -a GAN_EXTRA_ARRAY <<< "${GAN_EXTRA_ARGS}"
read -r -a SEMI_EXTRA_ARRAY <<< "${SEMI_EXTRA_ARGS}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

git_worktree_state() {
  local repo_dir="$1"
  local status_line=""

  status_line="$(git -C "${repo_dir}" status --short --untracked-files=normal 2>/dev/null | head -n 1 || true)"
  if [[ -n "${status_line}" ]]; then
    printf 'dirty\n'
  else
    printf 'clean\n'
  fi
}

log_repo_revision() {
  local repo_dir="$1"
  local revision=""
  local branch=""
  local worktree=""

  command -v git >/dev/null 2>&1 || {
    log "Repo state: git unavailable project_dir=${repo_dir}"
    return 0
  }
  [[ -d "${repo_dir}/.git" ]] || {
    log "Repo state: git metadata unavailable project_dir=${repo_dir}"
    return 0
  }

  revision="$(git -C "${repo_dir}" rev-parse --short HEAD 2>/dev/null || true)"
  branch="$(git -C "${repo_dir}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  if [[ -z "${revision}" ]]; then
    log "Repo state: unable to resolve git revision project_dir=${repo_dir}"
    return 0
  fi

  worktree="$(git_worktree_state "${repo_dir}")"
  log "Repo state: branch=${branch:-DETACHED} commit=${revision} worktree=${worktree} project_dir=${repo_dir}"
}

normalize_semi_autotune_mode() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    throughput|speed|perf) printf 'throughput\n' ;;
    result|metric|quality) printf 'result\n' ;;
    signal|warmup|probe) printf 'signal\n' ;;
    *) return 1 ;;
  esac
}

normalize_semi_autotune_metric() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    f1) printf 'best_f1\n' ;;
    dice|dsc) printf 'best_dice\n' ;;
    *) return 1 ;;
  esac
}

emit_export_hint() {
  local stage="$1"
  local dataset="$2"
  local expid="$3"
  local batch_var="$4"
  local batch_value="$5"
  local workers_var="$6"
  local workers_value="$7"
  log "[AUTOTUNE_EXPORT] ${stage} ${dataset} exp${expid}: export ${batch_var}=${batch_value} ${workers_var}=${workers_value}"
}

float_mul() {
  awk -v a="$1" -v b="$2" 'BEGIN { printf "%.12g\n", (a + 0) * (b + 0) }'
}

float_div() {
  awk -v a="$1" -v b="$2" 'BEGIN { if ((b + 0) == 0) exit 1; printf "%.12g\n", (a + 0) / (b + 0) }'
}

trim_whitespace() {
  sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

semi_tunable_env_vars() {
  cat <<'EOF'
SEMI_RATIO
SEMI_BATCH_SIZE
SEMI_NUM_WORKERS
SEMI_EPOCHS
SEMI_LR
SEMI_WEIGHT_DECAY
SEMI_POWER
SEMI_EPS
SEMI_MT
SEMI_BAND
SEMI_AUX_SUPERVISION_WEIGHT
SEMI_BOUNDARY_WEIGHT
SEMI_PSEUDO_MAIN_WEIGHT
SEMI_PSEUDO_AUX_WEIGHT
SEMI_SOR_WEIGHT
SEMI_EDGE_WEIGHT
SEMI_ADV_WEIGHT
SEMI_FM_WEIGHT
SEMI_TEACHER_CONSISTENCY_WEIGHT
SEMI_CONSISTENCY_RAMPUP
SEMI_TEACHER_RAMPUP
SEMI_STRUCTURE_RAMPUP
SEMI_SCD_RAMPUP
SEMI_PS_COPY_PASTE_PROB
SEMI_SCD_START_EPOCH
SEMI_SCD_UPDATE_INTERVAL
SEMI_CALIBRATION_WEIGHT
SEMI_RELIABILITY_FLOOR
SEMI_SOR_ERASE
SEMI_VAL_INTERVAL
SEMI_DATALOADER_START_METHOD
SEMI_EVAL_TTA
SEMI_EVAL_THRESHOLD
SEMI_EVAL_MIN_AREA
SEMI_EVAL_POSTPROCESS_MODE
SEMI_AUTO_TUNE_EVAL_POSTPROCESS
SEMI_SAVE_BEST_TTA
SEMI_SAVE_RELIABILITY_MAPS
SEMI_SAVE_RELIABILITY_LIMIT
EOF
}

semi_optional_override_env_vars() {
  cat <<'EOF'
SEMI_RATIO
SEMI_WEIGHT_DECAY
SEMI_POWER
SEMI_EPS
SEMI_MT
SEMI_BAND
SEMI_AUX_SUPERVISION_WEIGHT
SEMI_BOUNDARY_WEIGHT
SEMI_PSEUDO_MAIN_WEIGHT
SEMI_PSEUDO_AUX_WEIGHT
SEMI_SOR_WEIGHT
SEMI_EDGE_WEIGHT
SEMI_ADV_WEIGHT
SEMI_FM_WEIGHT
SEMI_TEACHER_CONSISTENCY_WEIGHT
SEMI_CONSISTENCY_RAMPUP
SEMI_TEACHER_RAMPUP
SEMI_STRUCTURE_RAMPUP
SEMI_SCD_RAMPUP
SEMI_PS_COPY_PASTE_PROB
SEMI_SCD_START_EPOCH
SEMI_SCD_UPDATE_INTERVAL
SEMI_CALIBRATION_WEIGHT
SEMI_RELIABILITY_FLOOR
SEMI_SOR_ERASE
SEMI_VAL_INTERVAL
SEMI_DATALOADER_START_METHOD
SEMI_EVAL_TTA
SEMI_EVAL_THRESHOLD
SEMI_EVAL_MIN_AREA
SEMI_EVAL_POSTPROCESS_MODE
SEMI_AUTO_TUNE_EVAL_POSTPROCESS
SEMI_SAVE_BEST_TTA
SEMI_SAVE_RELIABILITY_MAPS
SEMI_SAVE_RELIABILITY_LIMIT
EOF
}

normalize_semi_tune_var() {
  local key
  key="$(printf '%s' "$1" | sed 's/^--//' | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  case "${key}" in
    ratio|semi_ratio) printf 'SEMI_RATIO\n' ;;
    batch|batchsize|batch_size|semi_batch_size) printf 'SEMI_BATCH_SIZE\n' ;;
    workers|numworkers|num_workers|semi_num_workers) printf 'SEMI_NUM_WORKERS\n' ;;
    epochs|epoch|nepoch|nepochs|semi_epochs) printf 'SEMI_EPOCHS\n' ;;
    lr|learningrate|learning_rate|semi_lr) printf 'SEMI_LR\n' ;;
    weightdecay|weight_decay|wd|semi_weight_decay) printf 'SEMI_WEIGHT_DECAY\n' ;;
    power|semi_power) printf 'SEMI_POWER\n' ;;
    eps|epsilon|semi_eps) printf 'SEMI_EPS\n' ;;
    mt|ema|ema_momentum|semi_mt) printf 'SEMI_MT\n' ;;
    band|semi_band) printf 'SEMI_BAND\n' ;;
    auxsupervisionweight|aux_supervision_weight|semi_aux_supervision_weight) printf 'SEMI_AUX_SUPERVISION_WEIGHT\n' ;;
    boundaryweight|boundary_weight|semi_boundary_weight) printf 'SEMI_BOUNDARY_WEIGHT\n' ;;
    pseudomainweight|pseudo_main_weight|semi_pseudo_main_weight) printf 'SEMI_PSEUDO_MAIN_WEIGHT\n' ;;
    pseudoauxweight|pseudo_aux_weight|semi_pseudo_aux_weight) printf 'SEMI_PSEUDO_AUX_WEIGHT\n' ;;
    sorweight|sor_weight|semi_sor_weight) printf 'SEMI_SOR_WEIGHT\n' ;;
    edgeweight|edge_weight|semi_edge_weight) printf 'SEMI_EDGE_WEIGHT\n' ;;
    advweight|adv_weight|semi_adv_weight) printf 'SEMI_ADV_WEIGHT\n' ;;
    fmweight|fm_weight|semi_fm_weight) printf 'SEMI_FM_WEIGHT\n' ;;
    teacherconsistencyweight|teacher_consistency_weight|semi_teacher_consistency_weight) printf 'SEMI_TEACHER_CONSISTENCY_WEIGHT\n' ;;
    consistencyrampup|consistency_rampup|semi_consistency_rampup) printf 'SEMI_CONSISTENCY_RAMPUP\n' ;;
    teacherrampup|teacher_rampup|semi_teacher_rampup) printf 'SEMI_TEACHER_RAMPUP\n' ;;
    pscopypasteprob|ps_copy_paste_prob|semi_ps_copy_paste_prob) printf 'SEMI_PS_COPY_PASTE_PROB\n' ;;
    structurerampup|structure_rampup|semi_structure_rampup) printf 'SEMI_STRUCTURE_RAMPUP\n' ;;
    scdrampup|scd_rampup|semi_scd_rampup) printf 'SEMI_SCD_RAMPUP\n' ;;
    scdstartepoch|scd_start_epoch|semi_scd_start_epoch) printf 'SEMI_SCD_START_EPOCH\n' ;;
    scdupdateinterval|scd_update_interval|semi_scd_update_interval) printf 'SEMI_SCD_UPDATE_INTERVAL\n' ;;
    calibrationweight|calibration_weight|semi_calibration_weight) printf 'SEMI_CALIBRATION_WEIGHT\n' ;;
    reliabilityfloor|reliability_floor|semi_reliability_floor) printf 'SEMI_RELIABILITY_FLOOR\n' ;;
    sorerase|sor_erase|semi_sor_erase) printf 'SEMI_SOR_ERASE\n' ;;
    valinterval|val_interval|semi_val_interval) printf 'SEMI_VAL_INTERVAL\n' ;;
    dataloaderstartmethod|dataloader_start_method|loaderstart|loader_start|semi_dataloader_start_method) printf 'SEMI_DATALOADER_START_METHOD\n' ;;
    evaltta|eval_tta|semi_eval_tta) printf 'SEMI_EVAL_TTA\n' ;;
    evalthreshold|eval_threshold|threshold|semi_eval_threshold) printf 'SEMI_EVAL_THRESHOLD\n' ;;
    evalminarea|eval_min_area|minarea|min_area|semi_eval_min_area) printf 'SEMI_EVAL_MIN_AREA\n' ;;
    evalpostprocessmode|eval_postprocess_mode|postprocess_mode|semi_eval_postprocess_mode) printf 'SEMI_EVAL_POSTPROCESS_MODE\n' ;;
    autotuneevalpostprocess|auto_tune_eval_postprocess|semi_auto_tune_eval_postprocess) printf 'SEMI_AUTO_TUNE_EVAL_POSTPROCESS\n' ;;
    savebesttta|save_best_tta|semi_save_best_tta) printf 'SEMI_SAVE_BEST_TTA\n' ;;
    savereliabilitymaps|save_reliability_maps|semi_save_reliability_maps) printf 'SEMI_SAVE_RELIABILITY_MAPS\n' ;;
    savereliabilitylimit|save_reliability_limit|semi_save_reliability_limit) printf 'SEMI_SAVE_RELIABILITY_LIMIT\n' ;;
    *) return 1 ;;
  esac
}

semi_cli_arg_for_var() {
  case "$1" in
    SEMI_RATIO) printf '%s\n' --ratio ;;
    SEMI_BATCH_SIZE) printf '%s\n' --batch_size ;;
    SEMI_NUM_WORKERS) printf '%s\n' --num_workers ;;
    SEMI_EPOCHS) printf '%s\n' --nEpoch ;;
    SEMI_LR) printf '%s\n' --lr ;;
    SEMI_WEIGHT_DECAY) printf '%s\n' --weight_decay ;;
    SEMI_POWER) printf '%s\n' --power ;;
    SEMI_EPS) printf '%s\n' --eps ;;
    SEMI_MT) printf '%s\n' --mt ;;
    SEMI_BAND) printf '%s\n' --band ;;
    SEMI_AUX_SUPERVISION_WEIGHT) printf '%s\n' --aux_supervision_weight ;;
    SEMI_BOUNDARY_WEIGHT) printf '%s\n' --boundary_weight ;;
    SEMI_PSEUDO_MAIN_WEIGHT) printf '%s\n' --pseudo_main_weight ;;
    SEMI_PSEUDO_AUX_WEIGHT) printf '%s\n' --pseudo_aux_weight ;;
    SEMI_SOR_WEIGHT) printf '%s\n' --sor_weight ;;
    SEMI_EDGE_WEIGHT) printf '%s\n' --edge_weight ;;
    SEMI_ADV_WEIGHT) printf '%s\n' --adv_weight ;;
    SEMI_FM_WEIGHT) printf '%s\n' --fm_weight ;;
    SEMI_TEACHER_CONSISTENCY_WEIGHT) printf '%s\n' --teacher_consistency_weight ;;
    SEMI_CONSISTENCY_RAMPUP) printf '%s\n' --consistency_rampup ;;
    SEMI_TEACHER_RAMPUP) printf '%s\n' --teacher_rampup ;;
    SEMI_STRUCTURE_RAMPUP) printf '%s\n' --structure_rampup ;;
    SEMI_SCD_RAMPUP) printf '%s\n' --scd_rampup ;;
    SEMI_PS_COPY_PASTE_PROB) printf '%s\n' --ps_copy_paste_prob ;;
    SEMI_SCD_START_EPOCH) printf '%s\n' --scd_start_epoch ;;
    SEMI_SCD_UPDATE_INTERVAL) printf '%s\n' --scd_update_interval ;;
    SEMI_CALIBRATION_WEIGHT) printf '%s\n' --calibration_weight ;;
    SEMI_RELIABILITY_FLOOR) printf '%s\n' --reliability_floor ;;
    SEMI_SOR_ERASE) printf '%s\n' --sor_erase ;;
    SEMI_VAL_INTERVAL) printf '%s\n' --val_interval ;;
    SEMI_DATALOADER_START_METHOD) printf '%s\n' --dataloader_start_method ;;
    SEMI_EVAL_TTA) printf '%s\n' --eval_tta ;;
    SEMI_EVAL_THRESHOLD) printf '%s\n' --eval_threshold ;;
    SEMI_EVAL_MIN_AREA) printf '%s\n' --eval_min_area ;;
    SEMI_EVAL_POSTPROCESS_MODE) printf '%s\n' --eval_postprocess_mode ;;
    SEMI_AUTO_TUNE_EVAL_POSTPROCESS) printf '%s\n' --auto_tune_eval_postprocess ;;
    SEMI_SAVE_BEST_TTA) printf '%s\n' --save_best_tta ;;
    SEMI_SAVE_RELIABILITY_MAPS) printf '%s\n' --save_reliability_maps ;;
    SEMI_SAVE_RELIABILITY_LIMIT) printf '%s\n' --save_reliability_limit ;;
    *) return 1 ;;
  esac
}

semi_env_var_is_flag() {
  case "$1" in
    SEMI_EVAL_TTA|SEMI_AUTO_TUNE_EVAL_POSTPROCESS|SEMI_SAVE_BEST_TTA|SEMI_SAVE_RELIABILITY_MAPS) return 0 ;;
    *) return 1 ;;
  esac
}

normalize_bool_like() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | trim_whitespace
}

semi_flag_cli_arg_for_value() {
  local env_var="$1"
  local value="$2"
  local normalized=""

  normalized="$(normalize_bool_like "${value}")"
  case "${env_var}" in
    SEMI_EVAL_TTA)
      case "${normalized}" in
        1|true|yes|on) printf '%s\n' --eval_tta ;;
        0|false|no|off) printf '%s\n' --no_eval_tta ;;
        '') return 0 ;;
        *) die "unsupported boolean value for ${env_var}: ${value}" ;;
      esac
      ;;
    SEMI_AUTO_TUNE_EVAL_POSTPROCESS)
      case "${normalized}" in
        1|true|yes|on) printf '%s\n' --auto_tune_eval_postprocess ;;
        0|false|no|off) printf '%s\n' --no_auto_tune_eval_postprocess ;;
        '') return 0 ;;
        *) die "unsupported boolean value for ${env_var}: ${value}" ;;
      esac
      ;;
    SEMI_SAVE_BEST_TTA)
      case "${normalized}" in
        1|true|yes|on) printf '%s\n' --save_best_tta ;;
        0|false|no|off|'') return 0 ;;
        *) die "unsupported boolean value for ${env_var}: ${value}" ;;
      esac
      ;;
    SEMI_SAVE_RELIABILITY_MAPS)
      case "${normalized}" in
        1|true|yes|on) printf '%s\n' --save_reliability_maps ;;
        0|false|no|off|'') return 0 ;;
        *) die "unsupported boolean value for ${env_var}: ${value}" ;;
      esac
      ;;
    *)
      die "unsupported semi flag var ${env_var}"
      ;;
  esac
}

semi_append_optional_override_args() {
  local env_var
  local value
  local cli_arg
  while IFS= read -r env_var; do
    [[ -n "${env_var}" ]] || continue
    value="${!env_var:-}"
    [[ -n "${value}" ]] || continue
    cli_arg="$(semi_cli_arg_for_var "${env_var}")" || die "unsupported semi override var ${env_var}"
    if semi_env_var_is_flag "${env_var}"; then
      cli_arg="$(semi_flag_cli_arg_for_value "${env_var}" "${value}")"
      [[ -n "${cli_arg}" ]] && printf '%s\n' "${cli_arg}"
      continue
    fi
    printf '%s\n%s\n' "${cli_arg}" "${value}"
  done < <(semi_optional_override_env_vars)
}

semi_trial_value() {
  local trial="$1"
  local target="$2"
  local assignment
  local var
  local value
  [[ -n "${trial}" ]] || return 1
  IFS=';' read -r -a _trial_assignments <<< "${trial}"
  for assignment in "${_trial_assignments[@]}"; do
    [[ -n "${assignment}" ]] || continue
    var="${assignment%%=*}"
    value="${assignment#*=}"
    if [[ "${var}" == "${target}" ]]; then
      printf '%s\n' "${value}"
      return 0
    fi
  done
  return 1
}

semi_trial_override_args() {
  local trial="$1"
  local assignment
  local env_var
  local value
  local cli_arg
  [[ -n "${trial}" ]] || return 0
  IFS=';' read -r -a _trial_assignments <<< "${trial}"
  for assignment in "${_trial_assignments[@]}"; do
    [[ -n "${assignment}" ]] || continue
    env_var="${assignment%%=*}"
    value="${assignment#*=}"
    cli_arg="$(semi_cli_arg_for_var "${env_var}")" || die "unsupported semi tune var ${env_var}"
    if semi_env_var_is_flag "${env_var}"; then
      cli_arg="$(semi_flag_cli_arg_for_value "${env_var}" "${value}")"
      [[ -n "${cli_arg}" ]] && printf '%s\n' "${cli_arg}"
      continue
    fi
    printf '%s\n%s\n' "${cli_arg}" "${value}"
  done
}

semi_apply_trial_assignments() {
  local trial="$1"
  local assignment
  local env_var
  local value
  [[ -n "${trial}" ]] || return 0
  IFS=';' read -r -a _trial_assignments <<< "${trial}"
  for assignment in "${_trial_assignments[@]}"; do
    [[ -n "${assignment}" ]] || continue
    env_var="${assignment%%=*}"
    value="${assignment#*=}"
    printf -v "${env_var}" '%s' "${value}"
  done
}

semi_trial_tag() {
  local trial="$1"
  if [[ -z "${trial}" ]]; then
    printf 'default\n'
    return 0
  fi
  printf '%s\n' "${trial}" | tr ';=' '__' | sed 's/[^A-Za-z0-9_.-]/_/g'
}

semi_trial_display() {
  local trial="$1"
  if [[ -z "${trial}" ]]; then
    printf 'default\n'
    return 0
  fi
  printf '%s\n' "${trial}"
}

semi_export_string() {
  local env_var
  local value
  local joined=""
  while IFS= read -r env_var; do
    [[ -n "${env_var}" ]] || continue
    value="${!env_var:-}"
    [[ -n "${value}" ]] || continue
    if [[ -n "${joined}" ]]; then
      joined="${joined} "
    fi
    joined="${joined}${env_var}=${value}"
  done < <(semi_tunable_env_vars)
  printf '%s\n' "${joined}"
}

emit_semi_export_bundle() {
  local dataset="$1"
  local expid="$2"
  local export_string
  export_string="$(semi_export_string)"
  [[ -n "${export_string}" ]] || return 0
  log "[AUTOTUNE_EXPORT] semi ${dataset} exp${expid}: export ${export_string}"
}

log_pipeline_configuration() {
  log "Pipeline config: configs=${PIPELINE_CONFIGS} variants=${SEMI_VARIANTS} run_test=${RUN_TEST} skip_existing=${SKIP_EXISTING}"
  log "Output config: output_root=${OUTPUT_ROOT} checkpoint_root=${SEMI_CHECKPOINT_ROOT} result_root=${SEMI_RESULT_ROOT} summary_file=${TEST_SUMMARY_FILE}"
  log "GAN config: autotune=${GAN_AUTOTUNE} batch=${GAN_BATCH_SIZE} workers=${GAN_WORKERS} precision=${GAN_PRECISION} tf32=${GAN_TF32} niter=${GAN_NITER} expected_epoch=${GAN_EXPECTED_EPOCH} resume_from_latest=${GAN_RESUME_FROM_LATEST} lrD=${GAN_LR_D} lrG=${GAN_LR_G}"
  log "Semi config: autotune=${SEMI_AUTOTUNE} mode=${SEMI_AUTOTUNE_MODE} batch=${SEMI_BATCH_SIZE} workers=${SEMI_NUM_WORKERS} lr=${SEMI_LR} precision=${SEMI_PRECISION} tf32=${SEMI_TF32} eval_tta=${SEMI_EVAL_TTA:-<dataset-default>} eval_threshold=${SEMI_EVAL_THRESHOLD:-<dataset-default>} eval_min_area=${SEMI_EVAL_MIN_AREA:-<dataset-default>} eval_postprocess_mode=${SEMI_EVAL_POSTPROCESS_MODE:-<code-default>} auto_eval_postprocess=${SEMI_AUTO_TUNE_EVAL_POSTPROCESS:-<dataset-default>} save_reliability_maps=${SEMI_SAVE_RELIABILITY_MAPS} teacher_rampup=${SEMI_TEACHER_RAMPUP:-<consistency_rampup>} structure_rampup=${SEMI_STRUCTURE_RAMPUP:-<consistency_rampup>} scd_start_epoch=${SEMI_SCD_START_EPOCH:-<code-default>} scd_rampup=${SEMI_SCD_RAMPUP:-<consistency_rampup>}"
  log "Semi reliability: entropy_tau=${SEMI_ENTROPY_TAU:-<code-default>} floor=${SEMI_RELIABILITY_FLOOR:-<code-default>}"
  log "HC18 protocol: BiPCC transductive (official test_set may serve as unlabeled images; labels never used)"
  if [[ "${SEMI_AUTOTUNE_MODE}" == "signal" ]]; then
    log "Semi signal config: steps=${SEMI_SIGNAL_STEPS} warmup=${SEMI_SIGNAL_WARMUP_STEPS} seed=${SEMI_SIGNAL_SEED} lr_candidates=${SEMI_SIGNAL_LR_CANDIDATES:-<derived>} lr_multipliers=${SEMI_SIGNAL_LR_MULTIPLIERS}"
  elif [[ "${SEMI_AUTOTUNE_MODE}" == "result" ]]; then
    log "Semi result-tune config: metric=${SEMI_AUTOTUNE_METRIC_DISPLAY} epochs=${SEMI_RESULT_AUTOTUNE_EPOCHS} hp_spec=${SEMI_HP_TUNE_SPEC:-<none>}"
  else
    log "Semi throughput config: batch_candidates=${SEMI_AUTOTUNE_BATCH_CANDIDATES} worker_candidates=${SEMI_AUTOTUNE_WORKER_CANDIDATES}"
  fi
}

build_semi_result_tune_trials() {
  local spec_var_set="|"
  local -a dimensions=()
  local -a entries=()
  local -a trials=("")
  local -a next_trials=()
  local -a values=()
  local entry
  local normalized
  local values_raw
  local dimension
  local value
  local trial

  if [[ -n "${SEMI_HP_TUNE_SPEC}" ]]; then
    IFS=';' read -r -a entries <<< "${SEMI_HP_TUNE_SPEC}"
    for entry in "${entries[@]}"; do
      entry="$(printf '%s' "${entry}" | trim_whitespace)"
      [[ -n "${entry}" ]] || continue
      [[ "${entry}" == *=* ]] || die "invalid SEMI_HP_TUNE_SPEC entry '${entry}' (expected name=v1|v2)"
      normalized="$(normalize_semi_tune_var "${entry%%=*}")" || die "unsupported semi tune parameter '${entry%%=*}' in SEMI_HP_TUNE_SPEC"
      values_raw="${entry#*=}"
      values_raw="$(printf '%s' "${values_raw}" | tr ',' '|' | tr -d '[:space:]')"
      [[ -n "${values_raw}" ]] || die "empty candidate list for ${normalized} in SEMI_HP_TUNE_SPEC"
      dimensions+=("${normalized}=${values_raw}")
      spec_var_set="${spec_var_set}${normalized}|"
    done
  fi

  if [[ -z "${SEMI_BATCH_SIZE_WAS_SET}" && "${spec_var_set}" != *"|SEMI_BATCH_SIZE|"* ]]; then
    dimensions+=("SEMI_BATCH_SIZE=${SEMI_AUTOTUNE_BATCH_CANDIDATES// /}")
  fi
  if [[ -z "${SEMI_NUM_WORKERS_WAS_SET}" && "${spec_var_set}" != *"|SEMI_NUM_WORKERS|"* ]]; then
    dimensions+=("SEMI_NUM_WORKERS=${SEMI_AUTOTUNE_WORKER_CANDIDATES// /}")
  fi

  for dimension in "${dimensions[@]}"; do
    values_raw="${dimension#*=}"
    IFS='|' read -r -a values <<< "${values_raw}"
    next_trials=()
    for trial in "${trials[@]}"; do
      for value in "${values[@]}"; do
        value="$(printf '%s' "${value}" | trim_whitespace)"
        [[ -n "${value}" ]] || continue
        if [[ -n "${trial}" ]]; then
          next_trials+=("${trial};${dimension%%=*}=${value}")
        else
          next_trials+=("${dimension%%=*}=${value}")
        fi
      done
    done
    trials=("${next_trials[@]}")
  done

  printf '%s\n' "${trials[@]}"
}

count_csv_items() {
  local raw="${1// /}"
  if [[ -z "${raw}" ]]; then
    printf '0\n'
    return 0
  fi
  awk -F',' '{print NF}' <<< "${raw}"
}

relative_gpu_ids() {
  local raw="${1// /}"
  local -a items=()
  local -a result=()
  local idx

  if [[ -z "${raw}" ]]; then
    printf '\n'
    return 0
  fi

  IFS=',' read -r -a items <<< "${raw}"
  for idx in "${!items[@]}"; do
    [[ -n "${items[idx]}" ]] || continue
    result+=("${idx}")
  done

  local joined=""
  if [[ ${#result[@]} -gt 0 ]]; then
    joined="$(IFS=,; printf '%s' "${result[*]}")"
  fi
  printf '%s\n' "${joined}"
}

SEMI_GPUS="${SEMI_GPUS:-$(relative_gpu_ids "${SEMI_VISIBLE_DEVICES}")}"
SEMI_GPU_COUNT="$(count_csv_items "${SEMI_VISIBLE_DEVICES}")"

if [[ -z "${SEMI_AUTOTUNE_BATCH_CANDIDATES}" ]]; then
  if (( SEMI_GPU_COUNT >= 8 )); then
    SEMI_AUTOTUNE_BATCH_CANDIDATES="24,32"
  elif (( SEMI_GPU_COUNT >= 4 )); then
    SEMI_AUTOTUNE_BATCH_CANDIDATES="16,24,32"
  else
    SEMI_AUTOTUNE_BATCH_CANDIDATES="16,24"
  fi
fi

if [[ -z "${SEMI_AUTOTUNE_WORKER_CANDIDATES}" ]]; then
  if (( SEMI_GPU_COUNT >= 8 )); then
    SEMI_AUTOTUNE_WORKER_CANDIDATES="8,12"
  elif (( SEMI_GPU_COUNT >= 4 )); then
    SEMI_AUTOTUNE_WORKER_CANDIDATES="6,8,12"
  else
    SEMI_AUTOTUNE_WORKER_CANDIDATES="4,6,8"
  fi
fi

if [[ -z "${SEMI_BATCH_SIZE}" ]]; then
  if (( SEMI_GPU_COUNT >= 8 )); then
    SEMI_BATCH_SIZE="24"
  else
    SEMI_BATCH_SIZE="16"
  fi
fi

if [[ -z "${SEMI_NUM_WORKERS}" ]]; then
  if (( SEMI_GPU_COUNT >= 8 )); then
    SEMI_NUM_WORKERS="8"
  elif (( SEMI_GPU_COUNT >= 4 )); then
    SEMI_NUM_WORKERS="6"
  else
    SEMI_NUM_WORKERS="4"
  fi
fi

if [[ -z "${SEMI_EPOCHS}" ]]; then
  if (( SEMI_GPU_COUNT >= 8 )); then
    SEMI_EPOCHS="240"
  else
    SEMI_EPOCHS="200"
  fi
fi

if [[ -z "${SEMI_LR}" ]]; then
  if (( SEMI_GPU_COUNT >= 8 )); then
    SEMI_LR="1.25e-4"
  else
    SEMI_LR="1e-4"
  fi
fi

SEMI_BASE_BATCH_SIZE="${SEMI_BATCH_SIZE}"
SEMI_BASE_LR="${SEMI_LR}"

SEMI_AUTOTUNE_MODE="$(normalize_semi_autotune_mode "${SEMI_AUTOTUNE_MODE}")" || \
  die "unsupported SEMI_AUTOTUNE_MODE '${SEMI_AUTOTUNE_MODE}' (expected throughput, signal, or result)"
SEMI_AUTOTUNE_METRIC_KEY="$(normalize_semi_autotune_metric "${SEMI_AUTOTUNE_METRIC}")" || \
  die "unsupported SEMI_AUTOTUNE_METRIC '${SEMI_AUTOTUNE_METRIC}' (expected F1 or dice)"
SEMI_AUTOTUNE_METRIC_DISPLAY="${SEMI_AUTOTUNE_METRIC_KEY#best_}"

if [[ -z "${SEMI_RESULT_AUTOTUNE_EPOCHS}" ]]; then
  SEMI_RESULT_AUTOTUNE_EPOCHS="${SEMI_EPOCHS}"
fi

GAN_RUNTIME_BATCH_SIZE="${GAN_BATCH_SIZE}"
GAN_RUNTIME_WORKERS="${GAN_WORKERS}"
SEMI_RUNTIME_BATCH_SIZE="${SEMI_BATCH_SIZE}"
SEMI_RUNTIME_WORKERS="${SEMI_NUM_WORKERS}"
SEMI_RUNTIME_SETTINGS_READY="0"
SEMI_RUNTIME_BENCHMARK_VARIANT=""
SEMI_SELECTED_HP_ASSIGNMENTS=""

print_help() {
  cat <<EOF
Usage:
  bash scripts/run_training_pipeline.sh
  PIPELINE_CONFIGS="PAPER_ALL" bash scripts/run_training_pipeline.sh

Required layout:
  - Project repo: ${WORKSPACE_ROOT}/${PROJECT_NAME}
  - Data root: ${WORKSPACE_ROOT}/DATA
  - ResNet-34 weights: ${BACKBONE_PATH}

Outputs:
  - GAN artifacts: ${GAN_OUTPUT_ROOT}/{dataset}/exp{expID}/
  - Semi checkpoints: ${SEMI_CHECKPOINT_ROOT}/{ckpt_name}/
  - Semi visual results: ${SEMI_RESULT_ROOT}/{ckpt_name}/{manner}/
  - Test summary markdown: ${TEST_SUMMARY_FILE}
  - Logs: ${LOG_ROOT}/gan/ and ${LOG_ROOT}/semi/
  - Archived FBWA weights: ${SCD_ARCHIVE_ROOT}/{dataset}/exp{expID}/

Key environment variables:
  PIPELINE_CONFIGS     "PAPER_ALL" or space/comma separated list such as "BUSI TN3K:2 HC18:all"
  WORKSPACE_ROOT       Workspace root containing both the repo and DATA
  SEMI_VISIBLE_DEVICES Visible GPUs for semi, defaults to "0,1,2,3,4,5,6,7"
  GAN_BATCH_SIZE       Defaults to 32 for H200-class single-GPU GAN pretraining
  GAN_WORKERS          Defaults to 12 for most single-GPU GAN runs; HC18 falls back to 0 unless overridden because multi-worker loading can stall before the first step
  GAN_NITER            Defaults to 3001, expecting the final checkpoint at epoch 3000
  GAN_RESUME_FROM_LATEST
                       Defaults to 1; if the final GAN checkpoint is absent, resume from the newest paired netG/netD checkpoint
  GAN_AUTOTUNE         Defaults to 0; set to 1 to benchmark batch/workers before GAN training
  GAN_AUTOTUNE_STEPS   Measured GAN benchmark steps, default 6
  GAN_AUTOTUNE_BATCH_CANDIDATES   Comma-separated GAN batch candidates, default "32,48,64"
  GAN_AUTOTUNE_WORKER_CANDIDATES  Comma-separated GAN worker candidates, default "8,12,16"
  GAN_LAMBDA_ADV       Generator adversarial weight, default 0.1
  GAN_LAMBDA_EDGE      Generator edge loss weight, default 1.0
  GAN_LAMBDA_FM        Generator feature-matching weight, default 1.0
  GAN_LAMBDA_MASK      Generator mask reconstruction weight, default 2.0
  GAN_PRECISION        Defaults to fp32 because WGAN training is sensitive to bf16 critic quantization
  GAN_TF32            Set to 1 to enable TF32 kernels on Ampere/Hopper
  SEMI_BATCH_SIZE      Defaults to 24 on 8 GPUs, 16 otherwise
  SEMI_NUM_WORKERS     Defaults to 8 on 8 GPUs, 6 on 4 GPUs, 4 otherwise
  SEMI_EPOCHS          Defaults to 240 on 8 GPUs, 200 otherwise
  SEMI_LR              Defaults to 1.25e-4 on 8 GPUs, 1e-4 otherwise
  SEMI_AUTOTUNE        Set to 1 to tune batch/workers before semi training when they were not set explicitly
  SEMI_AUTOTUNE_MODE   "signal" (default), "throughput", or "result"
  signal mode first tunes batch/workers by throughput, then tunes learning rate by short warmup stability signals
  SEMI_AUTOTUNE_METRIC Validation metric for result mode, "F1" (default) or "dice"
  SEMI_AUTOTUNE_STEPS  Measured semi benchmark steps, default 6
  SEMI_RESULT_AUTOTUNE_EPOCHS  Training epochs used by semi result-mode autotune, default matches SEMI_EPOCHS
  SEMI_AUTOTUNE_BATCH_CANDIDATES  Defaults to "24,32" on 8 GPUs, smaller sets otherwise
  SEMI_AUTOTUNE_WORKER_CANDIDATES Defaults to "8,12" on 8 GPUs, smaller sets otherwise
  SEMI_HP_TUNE_SPEC    Result-mode grid, e.g. "lr=1e-4|2e-4;weight_decay=1e-5|5e-5;aux_supervision_weight=0.3|0.5"
  SEMI_SIGNAL_STEPS    Measured warmup steps for signal-mode LR tuning, default 120
  SEMI_SIGNAL_WARMUP_STEPS  Warmup steps ignored by signal-mode statistics, default 20
  SEMI_SIGNAL_SEED     Fixed seed used by signal-mode short runs, default 3407
  SEMI_SIGNAL_LR_CANDIDATES  Optional comma-separated LR candidates for signal mode
  SEMI_SIGNAL_LR_MULTIPLIERS Optional multipliers around the scaled base LR when explicit candidates are not provided, default "0.70,0.85,1.00,1.15"
  SEMI_VAL_INTERVAL    Validation cadence for semi training, default 5 epochs
  SEMI_DATALOADER_START_METHOD  DataLoader worker start method, default auto
  SEMI_EVAL_TTA        Set to 1 to force validation TTA, 0 to disable it; blank keeps dataset defaults (BUSI/TN3K on)
  SEMI_EVAL_THRESHOLD  Validation/test threshold for simple post-processing; blank keeps dataset defaults (BUSI 0.48, TN3K 0.50, others 0.50)
  SEMI_EVAL_MIN_AREA   Validation/test minimum connected-component area for simple post-processing; blank keeps dataset defaults (BUSI 100, TN3K 150, others 200)
  SEMI_EVAL_POSTPROCESS_MODE  "raw" (paper default), "simple", or "legacy" (older dataset-specific heuristics)
  SEMI_SAVE_BEST_TTA   Set to 1 to rerun best-checkpoint saving with TTA
  SEMI_SAVE_RELIABILITY_MAPS  Set to 1 to save reliability maps and reliability_stats.csv during save_best/test
  SEMI_SAVE_RELIABILITY_LIMIT Maximum number of reliability visualization cases to save per evaluation call
  SEMI_PRECISION       "auto" enables bf16 when supported
  SEMI_TF32            Set to 1 to enable TF32 kernels on Ampere/Hopper
  SEMI_VARIANTS        Space/comma separated list from {abrnet, abrnet_wo_boundary, abrnet_wo_anatomy, abrnet_wo_reliability, baseline}
  RUN_ABLATION         Set to 1 to run ABRNet ablation variants (default: 0, only runs "abrnet")
  TEST_SUMMARY_FILE    Markdown file that collects one row per completed test run
  GAN_VISIBLE_DEVICES  Visible GPU(s) for GAN, defaults to the first semi GPU
  CKPT_PREFIX          Prefix for semi checkpoint names
  SKIP_EXISTING        Defaults to 0, so existing outputs are retrained and overwritten
  SKIP_GAN             Set to 1 to skip the GAN/FBWA pretrain stage entirely (must already have
                       netD checkpoints in ${GAN_OUTPUT_ROOT}); FBWA weights still get handed off.
                       Independent from SKIP_EXISTING; useful for "only re-train semi" flows.
  RUN_TEST             Set to 0 to skip semi test after training
  GAN_EXTRA_ARGS       Extra args appended to GAN training
  SEMI_EXTRA_ARGS      Extra args appended to semi train/test

Examples:
  PIPELINE_CONFIGS="PAPER_ALL" SEMI_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" bash scripts/run_training_pipeline.sh
  PIPELINE_CONFIGS=\"HC18 PSFH BUSI TN3K\" SEMI_VARIANTS=\"abrnet abrnet_wo_boundary abrnet_wo_anatomy baseline\" bash scripts/run_training_pipeline.sh
  PIPELINE_CONFIGS="HC18:all PSFH:all" SEMI_VISIBLE_DEVICES="0" bash scripts/run_training_pipeline.sh
EOF
}

normalize_dataset() {
  case "$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')" in
    TN3K) printf 'TN3K\n' ;;
    BUSI) printf 'BUSI\n' ;;
    HC18) printf 'HC18\n' ;;
    PSFH) printf 'PSFH\n' ;;
    *) return 1 ;;
  esac
}

default_expids_for_dataset() {
  case "$1" in
    TN3K|BUSI) printf '1\n2\n3\n' ;;
    HC18|PSFH) printf '1\n2\n' ;;
    *) return 1 ;;
  esac
}

lower_dataset() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

gan_dataset_for() {
  case "$1" in
    TN3K) printf 'tn3k\n' ;;
    BUSI) printf 'busi\n' ;;
    HC18) printf 'hc18\n' ;;
    PSFH) printf 'psfh\n' ;;
    *) return 1 ;;
  esac
}

validate_expid() {
  local dataset="$1"
  local expid="$2"
  case "${dataset}" in
    TN3K|BUSI)
      [[ "${expid}" == "1" || "${expid}" == "2" || "${expid}" == "3" ]] || return 1
      ;;
    HC18|PSFH)
      [[ "${expid}" == "1" || "${expid}" == "2" ]] || return 1
      ;;
    *)
      return 1
      ;;
  esac
}

normalize_variant() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    baseline|base) printf 'baseline\n' ;;
    full|abrnet|abrl|abrpl|default) printf 'abrnet\n' ;;
    abrnet_wo_boundary|abrnet-without-boundary|wo_boundary|without_boundary|no_boundary|no-boundary) printf 'abrnet_wo_boundary\n' ;;
    abrnet_wo_anatomy|abrnet-without-anatomy|wo_anatomy|without_anatomy|no_anatomy|no-anatomy) printf 'abrnet_wo_anatomy\n' ;;
    abrnet_wo_reliability|abrnet-without-reliability|wo_reliability|without_reliability|no_reliability|no-reliability) printf 'abrnet_wo_reliability\n' ;;
    scra|scra_full|legacy_full) printf 'scra_full\n' ;;
    scd_only|scd-only|scdonly|scd) printf 'scd_only\n' ;;
    scd_sor|scd-sor|scdsor) printf 'scd_sor\n' ;;
    wo_scd|woscd|without_scd|no_scd|no-scd) printf 'wo_scd\n' ;;
    wo_sor|wosor|without_sor|no_sor|no-sor) printf 'wo_sor\n' ;;
    wo_ca|woca|without_ca|no_ca|no-ca) printf 'wo_ca\n' ;;
    sor_only|sor-only|soronly|sor) printf 'sor_only\n' ;;
    ca_only|ca-only|caonly|ca) printf 'ca_only\n' ;;
    *) return 1 ;;
  esac
}

variant_extra_args() {
  case "$1" in
    baseline)
      printf '%s\n' --no_scd --no_sor --no_ca
      ;;
    abrnet) ;;
    abrnet_wo_boundary)
      printf '%s\n' --boundary_weight 0 --calibration_weight 0
      ;;
    abrnet_wo_anatomy)
      printf '%s\n' --no_scd --adv_weight 0
      ;;
    abrnet_wo_reliability)
      printf '%s\n' --no_scd --calibration_weight 0
      ;;
    scra_full) ;;
    scd_only)
      printf '%s\n' --no_sor --no_ca
      ;;
    scd_sor|wo_ca)
      printf '%s\n' --no_ca
      ;;
    wo_scd)
      printf '%s\n' --no_scd
      ;;
    wo_sor)
      printf '%s\n' --no_sor
      ;;
    sor_only)
      printf '%s\n' --no_scd --no_ca
      ;;
    ca_only)
      printf '%s\n' --no_scd --no_sor
      ;;
    *)
      return 1
      ;;
  esac
}

variant_uses_scd() {
  case "$1" in
    abrnet|abrnet_wo_boundary|scra_full|scd_only|scd_sor|wo_sor|wo_ca)
      return 0
      ;;
    baseline|abrnet_wo_anatomy|abrnet_wo_reliability|wo_scd|sor_only|ca_only)
      return 1
      ;;
    *)
      return 1
      ;;
  esac
}

build_variant_entries() {
  local variants_raw="${SEMI_VARIANTS//,/ }"
  local -a raw_variants=()
  local raw_variant
  local normalized_variant

  read -r -a raw_variants <<< "${variants_raw}"
  [[ ${#raw_variants[@]} -gt 0 ]] || die "SEMI_VARIANTS is empty"

  for raw_variant in "${raw_variants[@]}"; do
    normalized_variant="$(normalize_variant "${raw_variant}")" || die "unsupported semi variant '${raw_variant}'"
    printf '%s\n' "${normalized_variant}"
  done
}

expand_pipeline_entry() {
  local raw_entry="$1"
  local normalized_raw
  local dataset_part
  local expid_part
  local dataset
  local expid

  normalized_raw="$(printf '%s' "${raw_entry}" | tr '[:lower:]' '[:upper:]')"
  if [[ "${normalized_raw}" == "PAPER_ALL" || "${normalized_raw}" == "ALL_PAPER" || "${normalized_raw}" == "PAPER" ]]; then
    for dataset in HC18 PSFH BUSI TN3K; do
      while IFS= read -r expid; do
        [[ -n "${expid}" ]] || continue
        printf '%s:%s\n' "${dataset}" "${expid}"
      done < <(default_expids_for_dataset "${dataset}")
    done
    return 0
  fi

  if [[ "${raw_entry}" == *:* ]]; then
    dataset_part="${raw_entry%%:*}"
    expid_part="${raw_entry##*:}"
  else
    dataset_part="${raw_entry}"
    expid_part="all"
  fi

  dataset="$(normalize_dataset "${dataset_part}")" || die "unsupported dataset '${dataset_part}'"
  if [[ "$(printf '%s' "${expid_part}" | tr '[:lower:]' '[:upper:]')" == "ALL" ]]; then
    while IFS= read -r expid; do
      [[ -n "${expid}" ]] || continue
      printf '%s:%s\n' "${dataset}" "${expid}"
    done < <(default_expids_for_dataset "${dataset}")
    return 0
  fi

  validate_expid "${dataset}" "${expid_part}" || die "unsupported expID '${expid_part}' for ${dataset}"
  printf '%s:%s\n' "${dataset}" "${expid_part}"
}

extract_autotune_value() {
  local line="$1"
  local key="$2"
  printf '%s\n' "${line}" | tr ' ' '\n' | sed -n "s/^${key}=//p" | head -n 1
}

is_numeric_value() {
  local value="$1"
  [[ "${value}" =~ ^[-+]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][-+]?[0-9]+)?$ ]]
}

score_is_better() {
  local candidate="$1"
  local best="$2"
  awk -v candidate="${candidate}" -v best="${best}" 'BEGIN { exit !(best == "" || candidate + 0 > best + 0) }'
}

select_semi_benchmark_variant() {
  local priority
  local variant
  for priority in abrnet abrnet_wo_boundary scra_full wo_ca wo_sor scd_sor scd_only ca_only sor_only abrnet_wo_anatomy wo_scd baseline; do
    for variant in "$@"; do
      if [[ "${variant}" == "${priority}" ]]; then
        printf '%s\n' "${variant}"
        return 0
      fi
    done
  done
  printf '%s\n' "$1"
}

derive_signal_lr_candidates() {
  local lr_value=""
  local multiplier=""
  local scaled_base_lr="${SEMI_LR}"

  if [[ -n "${SEMI_SIGNAL_LR_CANDIDATES}" ]]; then
    IFS=',' read -r -a _semi_signal_lr_candidates <<< "${SEMI_SIGNAL_LR_CANDIDATES// /}"
    for lr_value in "${_semi_signal_lr_candidates[@]}"; do
      [[ -n "${lr_value}" ]] || continue
      printf '%s\n' "${lr_value}"
    done
    return 0
  fi

  if [[ -n "${SEMI_BASE_BATCH_SIZE}" && -n "${SEMI_RUNTIME_BATCH_SIZE}" && "${SEMI_BASE_BATCH_SIZE}" != "${SEMI_RUNTIME_BATCH_SIZE}" ]]; then
    scaled_base_lr="$(float_mul "${SEMI_BASE_LR}" "$(float_div "${SEMI_RUNTIME_BATCH_SIZE}" "${SEMI_BASE_BATCH_SIZE}")")"
  fi

  IFS=',' read -r -a _semi_signal_lr_multipliers <<< "${SEMI_SIGNAL_LR_MULTIPLIERS// /}"
  for multiplier in "${_semi_signal_lr_multipliers[@]}"; do
    [[ -n "${multiplier}" ]] || continue
    printf '%s\n' "$(float_mul "${scaled_base_lr}" "${multiplier}")"
  done
}

benchmark_gan_candidate() {
  local gan_dataset="$1"
  local expid="$2"
  local batch_size="$3"
  local workers="$4"
  local log_file="$5"
  local bench_exp_dir="${GAN_OUTPUT_ROOT}/.autotune/${gan_dataset}/exp${expid}/b${batch_size}_w${workers}"
  local result_line
  local throughput
  local -a cmd=(
    "${PYTHON_BIN}" airs/GAN/main.py
    --dataset "${gan_dataset}"
    --root "${WORKSPACE_ROOT}"
    --expID "${expid}"
    --cuda
    --ngpu "${GAN_NGPU}"
    --batchSize "${batch_size}"
    --workers "${workers}"
    --niter "${GAN_NITER}"
    --lrD "${GAN_LR_D}"
    --lrG "${GAN_LR_G}"
    --lambda_adv "${GAN_LAMBDA_ADV}"
    --lambda_edge "${GAN_LAMBDA_EDGE}"
    --lambda_fm "${GAN_LAMBDA_FM}"
    --lambda_mask "${GAN_LAMBDA_MASK}"
    --precision "${GAN_PRECISION}"
    --warmup_gen_iterations "${GAN_WARMUP_GEN_ITERATIONS}"
    --warmup_diters "${GAN_WARMUP_DITERS}"
    --extra_diters_every "${GAN_EXTRA_DITERS_EVERY}"
    --save_every "${GAN_SAVE_EVERY}"
    --log_interval "${GAN_LOG_INTERVAL}"
    --benchmark_steps "${GAN_AUTOTUNE_STEPS}"
    --benchmark_warmup_steps "${GAN_AUTOTUNE_WARMUP_STEPS}"
    --experiment "${bench_exp_dir}"
  )

  if [[ "${GAN_TF32}" == "1" ]]; then
    cmd+=(--tf32)
  fi
  if [[ ${#GAN_EXTRA_ARRAY[@]} -gt 0 ]]; then
    cmd+=("${GAN_EXTRA_ARRAY[@]}")
  fi

  if (
    cd "${PROJECT_DIR}"
    CUDA_VISIBLE_DEVICES="${GAN_VISIBLE_DEVICES}" "${cmd[@]}"
  ) > "${log_file}" 2>&1; then
    result_line="$(grep '\[AUTOTUNE_RESULT\]' "${log_file}" | tail -n 1 || true)"
    throughput="$(extract_autotune_value "${result_line}" "samples_per_sec")"
    if is_numeric_value "${throughput}"; then
      log "GAN autotune candidate batch=${batch_size} workers=${workers} throughput=${throughput} samples/s"
      printf '%s\n' "${throughput}"
      return 0
    fi
    log "GAN autotune candidate batch=${batch_size} workers=${workers} finished without parsable throughput; see ${log_file}"
    return 1
  fi

  log "GAN autotune candidate batch=${batch_size} workers=${workers} failed; see ${log_file}"
  return 1
}

resolve_gan_runtime_settings() {
  local gan_dataset="$1"
  local expid="$2"
  local benchmark_dir="${LOG_ROOT}/autotune/gan"
  local selected_batch="${GAN_BATCH_SIZE}"
  local selected_workers="${GAN_WORKERS}"
  local best_batch="${selected_batch}"
  local best_workers="${selected_workers}"
  local best_score=""
  local score=""
  local candidate=""
  local log_file=""
  local skip_worker_autotune="0"

  if [[ -z "${GAN_WORKERS_WAS_SET}" && "${gan_dataset}" == "hc18" ]]; then
    if [[ "${selected_workers}" != "0" ]]; then
      log "Using GAN workers=0 for ${gan_dataset} by default; HC18 multi-worker loading can stall before the first training step."
    fi
    selected_workers="0"
    skip_worker_autotune="1"
  fi

  GAN_RUNTIME_BATCH_SIZE="${selected_batch}"
  GAN_RUNTIME_WORKERS="${selected_workers}"

  if [[ "${GAN_AUTOTUNE}" != "1" ]]; then
    return 0
  fi
  if [[ -n "${GAN_BATCH_SIZE_WAS_SET}" && -n "${GAN_WORKERS_WAS_SET}" ]]; then
    log "Skipping GAN autotune because batch size and workers were set explicitly"
    return 0
  fi

  mkdir -p "${benchmark_dir}"
  log "Auto-tuning GAN settings for ${gan_dataset} exp${expid}"

  if [[ -z "${GAN_BATCH_SIZE_WAS_SET}" ]]; then
    log_file="${benchmark_dir}/${gan_dataset}_exp${expid}_batch${selected_batch}_workers${selected_workers}.log"
    score="$(benchmark_gan_candidate "${gan_dataset}" "${expid}" "${selected_batch}" "${selected_workers}" "${log_file}" || true)"
    if is_numeric_value "${score}"; then
      best_score="${score}"
      best_batch="${selected_batch}"
    fi

    IFS=',' read -r -a _gan_batch_candidates <<< "${GAN_AUTOTUNE_BATCH_CANDIDATES// /}"
    for candidate in "${_gan_batch_candidates[@]}"; do
      [[ -n "${candidate}" ]] || continue
      [[ "${candidate}" == "${selected_batch}" ]] && continue
      log_file="${benchmark_dir}/${gan_dataset}_exp${expid}_batch${candidate}_workers${selected_workers}.log"
      score="$(benchmark_gan_candidate "${gan_dataset}" "${expid}" "${candidate}" "${selected_workers}" "${log_file}" || true)"
      if is_numeric_value "${score}" && score_is_better "${score}" "${best_score}"; then
        best_score="${score}"
        best_batch="${candidate}"
      fi
    done
    selected_batch="${best_batch}"
  fi

  if [[ -z "${GAN_WORKERS_WAS_SET}" ]]; then
    if [[ "${skip_worker_autotune}" == "1" ]]; then
      log "Skipping GAN worker autotune for ${gan_dataset}; keeping workers=${selected_workers}"
      GAN_RUNTIME_BATCH_SIZE="${selected_batch}"
      GAN_RUNTIME_WORKERS="${selected_workers}"
      log "Selected GAN settings: batch=${GAN_RUNTIME_BATCH_SIZE} workers=${GAN_RUNTIME_WORKERS}"
      emit_export_hint "gan" "${gan_dataset}" "${expid}" "GAN_BATCH_SIZE" "${GAN_RUNTIME_BATCH_SIZE}" "GAN_WORKERS" "${GAN_RUNTIME_WORKERS}"
      return 0
    fi

    best_score=""
    log_file="${benchmark_dir}/${gan_dataset}_exp${expid}_batch${selected_batch}_workers${selected_workers}.log"
    score="$(benchmark_gan_candidate "${gan_dataset}" "${expid}" "${selected_batch}" "${selected_workers}" "${log_file}" || true)"
    if is_numeric_value "${score}"; then
      best_score="${score}"
      best_workers="${selected_workers}"
    fi

    IFS=',' read -r -a _gan_worker_candidates <<< "${GAN_AUTOTUNE_WORKER_CANDIDATES// /}"
    for candidate in "${_gan_worker_candidates[@]}"; do
      [[ -n "${candidate}" ]] || continue
      [[ "${candidate}" == "${selected_workers}" ]] && continue
      log_file="${benchmark_dir}/${gan_dataset}_exp${expid}_batch${selected_batch}_workers${candidate}.log"
      score="$(benchmark_gan_candidate "${gan_dataset}" "${expid}" "${selected_batch}" "${candidate}" "${log_file}" || true)"
      if is_numeric_value "${score}" && score_is_better "${score}" "${best_score}"; then
        best_score="${score}"
        best_workers="${candidate}"
      fi
    done
    selected_workers="${best_workers}"
  fi

  GAN_RUNTIME_BATCH_SIZE="${selected_batch}"
  GAN_RUNTIME_WORKERS="${selected_workers}"
  log "Selected GAN settings: batch=${GAN_RUNTIME_BATCH_SIZE} workers=${GAN_RUNTIME_WORKERS}"
  emit_export_hint "gan" "${gan_dataset}" "${expid}" "GAN_BATCH_SIZE" "${GAN_RUNTIME_BATCH_SIZE}" "GAN_WORKERS" "${GAN_RUNTIME_WORKERS}"
}

benchmark_semi_candidate() {
  local semi_dataset="$1"
  local expid="$2"
  local variant="$3"
  local batch_size="$4"
  local workers="$5"
  local log_file="$6"
  local ckpt_name="autotune_$(lower_dataset "${semi_dataset}")_exp${expid}_${variant}_b${batch_size}_w${workers}"
  local result_line
  local throughput
  local -a cmd=(
    "${PYTHON_BIN}" airs/semi/code/main.py
    --manner semi
    --dataset "${semi_dataset}"
    --root "${WORKSPACE_ROOT}"
    --expID "${expid}"
    --GPUs "${SEMI_GPUS}"
    --batch_size "${batch_size}"
    --num_workers "${workers}"
    --nEpoch "${SEMI_EPOCHS}"
    --lr "${SEMI_LR}"
    --precision "${SEMI_PRECISION}"
    --ckpt_name "${ckpt_name}"
    --benchmark_steps "${SEMI_AUTOTUNE_STEPS}"
    --benchmark_warmup_steps "${SEMI_AUTOTUNE_WARMUP_STEPS}"
    --disable_tqdm
    --log_interval 0
  )

  if [[ "${SEMI_TF32}" == "1" ]]; then
    cmd+=(--tf32)
  fi
  while IFS= read -r variant_arg; do
    [[ -n "${variant_arg}" ]] || continue
    cmd+=("${variant_arg}")
  done < <(variant_extra_args "${variant}")
  while IFS= read -r override_arg; do
    [[ -n "${override_arg}" ]] || continue
    cmd+=("${override_arg}")
  done < <(semi_append_optional_override_args)
  if [[ ${#SEMI_EXTRA_ARRAY[@]} -gt 0 ]]; then
    cmd+=("${SEMI_EXTRA_ARRAY[@]}")
  fi

  if (
    cd "${PROJECT_DIR}"
    export AIRS_SEMI_CHECKPOINT_ROOT="${SEMI_CHECKPOINT_ROOT}"
    export AIRS_BACKBONE_PRETRAIN_PATH="${BACKBONE_PATH}"
    export AIRS_SCD_PRETRAIN_PATH="${SCD_ACTIVE_PATH}"
    export AIRS_SEMI_RESULT_DIR="${SEMI_RESULT_ROOT}/autotune/${ckpt_name}"
    export AIRS_TEST_SUMMARY_FILE="${TEST_SUMMARY_FILE}"
    CUDA_VISIBLE_DEVICES="${SEMI_VISIBLE_DEVICES}" "${cmd[@]}"
  ) > "${log_file}" 2>&1; then
    result_line="$(grep '\[AUTOTUNE_RESULT\]' "${log_file}" | tail -n 1 || true)"
    throughput="$(extract_autotune_value "${result_line}" "samples_per_sec")"
    if is_numeric_value "${throughput}"; then
      log "Semi autotune candidate variant=${variant} batch=${batch_size} workers=${workers} throughput=${throughput} samples/s"
      printf '%s\n' "${throughput}"
      return 0
    fi
    log "Semi autotune candidate variant=${variant} batch=${batch_size} workers=${workers} finished without parsable throughput; see ${log_file}"
    return 1
  fi

  log "Semi autotune candidate variant=${variant} batch=${batch_size} workers=${workers} failed; see ${log_file}"
  return 1
}

benchmark_semi_result_candidate() {
  local semi_dataset="$1"
  local expid="$2"
  local variant="$3"
  local trial="$4"
  local log_file="$5"
  local batch_size="${SEMI_BATCH_SIZE}"
  local workers="${SEMI_NUM_WORKERS}"
  local ckpt_name="autotune_result_$(lower_dataset "${semi_dataset}")_exp${expid}_${variant}_b${batch_size}_w${workers}"
  local result_line
  local score
  local autotune_checkpoint_root="${SEMI_CHECKPOINT_ROOT}/autotune"

  if semi_trial_value "${trial}" "SEMI_BATCH_SIZE" >/dev/null 2>&1; then
    batch_size="$(semi_trial_value "${trial}" "SEMI_BATCH_SIZE")"
  fi
  if semi_trial_value "${trial}" "SEMI_NUM_WORKERS" >/dev/null 2>&1; then
    workers="$(semi_trial_value "${trial}" "SEMI_NUM_WORKERS")"
  fi
  ckpt_name="autotune_result_$(lower_dataset "${semi_dataset}")_exp${expid}_${variant}_b${batch_size}_w${workers}"
  local -a cmd=(
    "${PYTHON_BIN}" airs/semi/code/main.py
    --manner semi
    --dataset "${semi_dataset}"
    --root "${WORKSPACE_ROOT}"
    --expID "${expid}"
    --GPUs "${SEMI_GPUS}"
    --batch_size "${batch_size}"
    --num_workers "${workers}"
    --nEpoch "${SEMI_RESULT_AUTOTUNE_EPOCHS}"
    --lr "${SEMI_LR}"
    --precision "${SEMI_PRECISION}"
    --ckpt_name "${ckpt_name}"
    --disable_tqdm
    --log_interval 0
  )

  if [[ "${SEMI_TF32}" == "1" ]]; then
    cmd+=(--tf32)
  fi
  while IFS= read -r variant_arg; do
    [[ -n "${variant_arg}" ]] || continue
    cmd+=("${variant_arg}")
  done < <(variant_extra_args "${variant}")
  while IFS= read -r override_arg; do
    [[ -n "${override_arg}" ]] || continue
    cmd+=("${override_arg}")
  done < <(semi_append_optional_override_args)
  while IFS= read -r trial_arg; do
    [[ -n "${trial_arg}" ]] || continue
    cmd+=("${trial_arg}")
  done < <(semi_trial_override_args "${trial}")
  if [[ ${#SEMI_EXTRA_ARRAY[@]} -gt 0 ]]; then
    cmd+=("${SEMI_EXTRA_ARRAY[@]}")
  fi

  if (
    cd "${PROJECT_DIR}"
    export AIRS_SEMI_CHECKPOINT_ROOT="${autotune_checkpoint_root}"
    export AIRS_BACKBONE_PRETRAIN_PATH="${BACKBONE_PATH}"
    export AIRS_SCD_PRETRAIN_PATH="${SCD_ACTIVE_PATH}"
    export AIRS_SEMI_RESULT_DIR="${SEMI_RESULT_ROOT}/autotune/${ckpt_name}"
    export AIRS_TEST_SUMMARY_FILE="${TEST_SUMMARY_FILE}"
    CUDA_VISIBLE_DEVICES="${SEMI_VISIBLE_DEVICES}" "${cmd[@]}"
  ) > "${log_file}" 2>&1; then
    result_line="$(grep '\[TRAIN_RESULT\]' "${log_file}" | tail -n 1 || true)"
    score="$(extract_autotune_value "${result_line}" "${SEMI_AUTOTUNE_METRIC_KEY}")"
    if is_numeric_value "${score}"; then
      log "Semi result-tune candidate variant=${variant} batch=${batch_size} workers=${workers} metric=${SEMI_AUTOTUNE_METRIC_DISPLAY} score=${score}"
      printf '%s\n' "${score}"
      return 0
    fi
    log "Semi result-tune candidate variant=${variant} batch=${batch_size} workers=${workers} finished without parsable ${SEMI_AUTOTUNE_METRIC_DISPLAY}; see ${log_file}"
    return 1
  fi

  log "Semi result-tune candidate variant=${variant} batch=${batch_size} workers=${workers} failed; see ${log_file}"
  return 1
}

benchmark_semi_signal_candidate() {
  local semi_dataset="$1"
  local expid="$2"
  local variant="$3"
  local lr="$4"
  local log_file="$5"
  local batch_size="${SEMI_RUNTIME_BATCH_SIZE}"
  local workers="${SEMI_RUNTIME_WORKERS}"
  local ckpt_name="autotune_signal_$(lower_dataset "${semi_dataset}")_exp${expid}_${variant}_lr$(printf '%s' "${lr}" | sed 's/[^A-Za-z0-9_.-]/_/g')"
  local result_line
  local score
  local val_f1
  local val_dice
  local autotune_checkpoint_root="${SEMI_CHECKPOINT_ROOT}/autotune_signal"
  local -a cmd=(
    "${PYTHON_BIN}" airs/semi/code/main.py
    --manner semi
    --dataset "${semi_dataset}"
    --root "${WORKSPACE_ROOT}"
    --expID "${expid}"
    --GPUs "${SEMI_GPUS}"
    --batch_size "${batch_size}"
    --num_workers "${workers}"
    --nEpoch "${SEMI_EPOCHS}"
    --lr "${lr}"
    --precision "${SEMI_PRECISION}"
    --ckpt_name "${ckpt_name}"
    --seed "${SEMI_SIGNAL_SEED}"
    --signal_steps "${SEMI_SIGNAL_STEPS}"
    --signal_warmup_steps "${SEMI_SIGNAL_WARMUP_STEPS}"
    --disable_tqdm
    --log_interval 0
  )

  if [[ "${SEMI_TF32}" == "1" ]]; then
    cmd+=(--tf32)
  fi
  while IFS= read -r variant_arg; do
    [[ -n "${variant_arg}" ]] || continue
    cmd+=("${variant_arg}")
  done < <(variant_extra_args "${variant}")
  while IFS= read -r override_arg; do
    [[ -n "${override_arg}" ]] || continue
    cmd+=("${override_arg}")
  done < <(semi_append_optional_override_args)
  if [[ ${#SEMI_EXTRA_ARRAY[@]} -gt 0 ]]; then
    cmd+=("${SEMI_EXTRA_ARRAY[@]}")
  fi

  if (
    cd "${PROJECT_DIR}"
    export AIRS_SEMI_CHECKPOINT_ROOT="${autotune_checkpoint_root}"
    export AIRS_BACKBONE_PRETRAIN_PATH="${BACKBONE_PATH}"
    export AIRS_SCD_PRETRAIN_PATH="${SCD_ACTIVE_PATH}"
    export AIRS_SEMI_RESULT_DIR="${SEMI_RESULT_ROOT}/autotune_signal/${ckpt_name}"
    export AIRS_TEST_SUMMARY_FILE="${TEST_SUMMARY_FILE}"
    CUDA_VISIBLE_DEVICES="${SEMI_VISIBLE_DEVICES}" "${cmd[@]}"
  ) > "${log_file}" 2>&1; then
    result_line="$(grep '\[SIGNAL_RESULT\]' "${log_file}" | tail -n 1 || true)"
    score="$(extract_autotune_value "${result_line}" "signal_score")"
    val_f1="$(extract_autotune_value "${result_line}" "val_f1")"
    val_dice="$(extract_autotune_value "${result_line}" "val_dice")"
    if is_numeric_value "${score}"; then
      log "Semi signal-tune candidate variant=${variant} batch=${batch_size} workers=${workers} lr=${lr} score=${score} val_f1=${val_f1:-NA} val_dice=${val_dice:-NA}"
      printf '%s\n' "${score}"
      return 0
    fi
    log "Semi signal-tune candidate variant=${variant} batch=${batch_size} workers=${workers} lr=${lr} finished without parsable signal score; see ${log_file}"
    return 1
  fi

  log "Semi signal-tune candidate variant=${variant} batch=${batch_size} workers=${workers} lr=${lr} failed; see ${log_file}"
  return 1
}

resolve_semi_runtime_settings_by_result() {
  local semi_dataset="$1"
  local expid="$2"
  local benchmark_variant="$3"
  local benchmark_dir="${LOG_ROOT}/autotune/semi_result"
  local selected_batch="${SEMI_BATCH_SIZE}"
  local selected_workers="${SEMI_NUM_WORKERS}"
  local best_batch="${selected_batch}"
  local best_workers="${selected_workers}"
  local best_score=""
  local best_trial=""
  local score=""
  local trial=""
  local log_file=""
  local trial_tag=""

  mkdir -p "${benchmark_dir}"
  log "Auto-tuning semi settings for ${semi_dataset} exp${expid} using variant ${benchmark_variant} by validation ${SEMI_AUTOTUNE_METRIC_DISPLAY} over ${SEMI_RESULT_AUTOTUNE_EPOCHS} epoch(s)"

  while IFS= read -r trial; do
    [[ -n "${trial}" ]] || true
    trial_tag="$(semi_trial_tag "${trial}")"
    log_file="${benchmark_dir}/${semi_dataset}_exp${expid}_${benchmark_variant}_${trial_tag}.log"
    score="$(benchmark_semi_result_candidate "${semi_dataset}" "${expid}" "${benchmark_variant}" "${trial}" "${log_file}" || true)"
    if is_numeric_value "${score}" && score_is_better "${score}" "${best_score}"; then
      best_score="${score}"
      best_trial="${trial}"
    fi
  done < <(build_semi_result_tune_trials)

  [[ -n "${best_score}" ]] || die "semi result autotune failed to produce a valid ${SEMI_AUTOTUNE_METRIC_DISPLAY} score"
  semi_apply_trial_assignments "${best_trial}"

  best_batch="${SEMI_BATCH_SIZE}"
  best_workers="${SEMI_NUM_WORKERS}"

  SEMI_RUNTIME_BATCH_SIZE="${best_batch}"
  SEMI_RUNTIME_WORKERS="${best_workers}"
  SEMI_BATCH_SIZE="${SEMI_RUNTIME_BATCH_SIZE}"
  SEMI_NUM_WORKERS="${SEMI_RUNTIME_WORKERS}"
  SEMI_SELECTED_HP_ASSIGNMENTS="${best_trial}"
  log "Selected semi settings: batch=${SEMI_RUNTIME_BATCH_SIZE} workers=${SEMI_RUNTIME_WORKERS} mode=result metric=${SEMI_AUTOTUNE_METRIC_DISPLAY} score=${best_score} trial=$(semi_trial_display "${best_trial}")"
  emit_semi_export_bundle "${semi_dataset}" "${expid}"
}

resolve_semi_runtime_settings_by_signal() {
  local semi_dataset="$1"
  local expid="$2"
  local benchmark_variant="$3"
  local benchmark_dir="${LOG_ROOT}/autotune/semi_signal"
  local candidate_lr=""
  local best_lr=""
  local best_score=""
  local score=""
  local log_file=""

  resolve_semi_runtime_settings_by_throughput "${semi_dataset}" "${expid}" "${benchmark_variant}"

  mkdir -p "${benchmark_dir}"
  log "Auto-tuning semi learning rate for ${semi_dataset} exp${expid} using variant ${benchmark_variant} by signal score (steps=${SEMI_SIGNAL_STEPS} warmup=${SEMI_SIGNAL_WARMUP_STEPS} seed=${SEMI_SIGNAL_SEED})"

  while IFS= read -r candidate_lr; do
    [[ -n "${candidate_lr}" ]] || continue
    log_file="${benchmark_dir}/${semi_dataset}_exp${expid}_${benchmark_variant}_lr$(printf '%s' "${candidate_lr}" | sed 's/[^A-Za-z0-9_.-]/_/g').log"
    score="$(benchmark_semi_signal_candidate "${semi_dataset}" "${expid}" "${benchmark_variant}" "${candidate_lr}" "${log_file}" || true)"
    if is_numeric_value "${score}" && score_is_better "${score}" "${best_score}"; then
      best_score="${score}"
      best_lr="${candidate_lr}"
    fi
  done < <(derive_signal_lr_candidates)

  [[ -n "${best_score}" && -n "${best_lr}" ]] || die "semi signal autotune failed to produce a valid signal score"

  SEMI_LR="${best_lr}"
  SEMI_SELECTED_HP_ASSIGNMENTS="SEMI_LR=${SEMI_LR}"
  log "Selected semi settings: batch=${SEMI_RUNTIME_BATCH_SIZE} workers=${SEMI_RUNTIME_WORKERS} lr=${SEMI_LR} mode=signal score=${best_score}"
  emit_semi_export_bundle "${semi_dataset}" "${expid}"
}

resolve_semi_runtime_settings_by_throughput() {
  local semi_dataset="$1"
  local expid="$2"
  local benchmark_variant="$3"
  local benchmark_dir="${LOG_ROOT}/autotune/semi"
  local selected_batch="${SEMI_BATCH_SIZE}"
  local selected_workers="${SEMI_NUM_WORKERS}"
  local best_batch="${selected_batch}"
  local best_workers="${selected_workers}"
  local best_score=""
  local score=""
  local candidate=""
  local log_file=""

  SEMI_RUNTIME_BATCH_SIZE="${selected_batch}"
  SEMI_RUNTIME_WORKERS="${selected_workers}"

  if [[ -n "${SEMI_BATCH_SIZE_WAS_SET}" && -n "${SEMI_NUM_WORKERS_WAS_SET}" && -z "${SEMI_HP_TUNE_SPEC}" ]]; then
    log "Skipping semi autotune because batch size and workers were set explicitly"
    return 0
  fi

  mkdir -p "${benchmark_dir}"
  log "Auto-tuning semi settings for ${semi_dataset} exp${expid} using variant ${benchmark_variant}"

  if [[ -z "${SEMI_BATCH_SIZE_WAS_SET}" ]]; then
    log_file="${benchmark_dir}/${semi_dataset}_exp${expid}_${benchmark_variant}_batch${selected_batch}_workers${selected_workers}.log"
    score="$(benchmark_semi_candidate "${semi_dataset}" "${expid}" "${benchmark_variant}" "${selected_batch}" "${selected_workers}" "${log_file}" || true)"
    if is_numeric_value "${score}"; then
      best_score="${score}"
      best_batch="${selected_batch}"
    fi

    IFS=',' read -r -a _semi_batch_candidates <<< "${SEMI_AUTOTUNE_BATCH_CANDIDATES// /}"
    for candidate in "${_semi_batch_candidates[@]}"; do
      [[ -n "${candidate}" ]] || continue
      [[ "${candidate}" == "${selected_batch}" ]] && continue
      log_file="${benchmark_dir}/${semi_dataset}_exp${expid}_${benchmark_variant}_batch${candidate}_workers${selected_workers}.log"
      score="$(benchmark_semi_candidate "${semi_dataset}" "${expid}" "${benchmark_variant}" "${candidate}" "${selected_workers}" "${log_file}" || true)"
      if is_numeric_value "${score}" && score_is_better "${score}" "${best_score}"; then
        best_score="${score}"
        best_batch="${candidate}"
      fi
    done
    selected_batch="${best_batch}"
  fi

  if [[ -z "${SEMI_NUM_WORKERS_WAS_SET}" ]]; then
    best_score=""
    log_file="${benchmark_dir}/${semi_dataset}_exp${expid}_${benchmark_variant}_batch${selected_batch}_workers${selected_workers}.log"
    score="$(benchmark_semi_candidate "${semi_dataset}" "${expid}" "${benchmark_variant}" "${selected_batch}" "${selected_workers}" "${log_file}" || true)"
    if is_numeric_value "${score}"; then
      best_score="${score}"
      best_workers="${selected_workers}"
    fi

    IFS=',' read -r -a _semi_worker_candidates <<< "${SEMI_AUTOTUNE_WORKER_CANDIDATES// /}"
    for candidate in "${_semi_worker_candidates[@]}"; do
      [[ -n "${candidate}" ]] || continue
      [[ "${candidate}" == "${selected_workers}" ]] && continue
      log_file="${benchmark_dir}/${semi_dataset}_exp${expid}_${benchmark_variant}_batch${selected_batch}_workers${candidate}.log"
      score="$(benchmark_semi_candidate "${semi_dataset}" "${expid}" "${benchmark_variant}" "${selected_batch}" "${candidate}" "${log_file}" || true)"
      if is_numeric_value "${score}" && score_is_better "${score}" "${best_score}"; then
        best_score="${score}"
        best_workers="${candidate}"
      fi
    done
    selected_workers="${best_workers}"
  fi

  SEMI_RUNTIME_BATCH_SIZE="${selected_batch}"
  SEMI_RUNTIME_WORKERS="${selected_workers}"
  SEMI_BATCH_SIZE="${SEMI_RUNTIME_BATCH_SIZE}"
  SEMI_NUM_WORKERS="${SEMI_RUNTIME_WORKERS}"
  SEMI_SELECTED_HP_ASSIGNMENTS=""
  log "Selected semi settings: batch=${SEMI_RUNTIME_BATCH_SIZE} workers=${SEMI_RUNTIME_WORKERS} mode=throughput"
  emit_semi_export_bundle "${semi_dataset}" "${expid}"
}

resolve_semi_runtime_settings() {
  local semi_dataset="$1"
  local expid="$2"
  local benchmark_variant="$3"

  SEMI_RUNTIME_BATCH_SIZE="${SEMI_BATCH_SIZE}"
  SEMI_RUNTIME_WORKERS="${SEMI_NUM_WORKERS}"

  if [[ "${SEMI_AUTOTUNE}" != "1" ]]; then
    return 0
  fi

  if [[ "${SEMI_AUTOTUNE_MODE}" == "result" ]]; then
    resolve_semi_runtime_settings_by_result "${semi_dataset}" "${expid}" "${benchmark_variant}"
    return 0
  fi

  if [[ -n "${SEMI_HP_TUNE_SPEC}" ]]; then
    if [[ "${SEMI_AUTOTUNE_MODE}" == "signal" ]]; then
      log "Ignoring SEMI_HP_TUNE_SPEC because SEMI_AUTOTUNE_MODE=signal; signal mode currently tunes learning rate only."
    else
      log "Ignoring SEMI_HP_TUNE_SPEC because SEMI_AUTOTUNE_MODE=${SEMI_AUTOTUNE_MODE}; use result mode to tune training hyperparameters"
    fi
  fi

  if [[ "${SEMI_AUTOTUNE_MODE}" == "signal" ]]; then
    resolve_semi_runtime_settings_by_signal "${semi_dataset}" "${expid}" "${benchmark_variant}"
    return 0
  fi

  resolve_semi_runtime_settings_by_throughput "${semi_dataset}" "${expid}" "${benchmark_variant}"
}

python_deps_ok() {
  "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import albumentations  # noqa: F401
import cv2  # noqa: F401
import numpy  # noqa: F401
import PIL  # noqa: F401
import scipy  # noqa: F401
import skimage  # noqa: F401
import torch  # noqa: F401
import torchvision  # noqa: F401
import tqdm  # noqa: F401
PY
}

ensure_python_deps() {
  [[ "${AUTO_INSTALL_DEPS}" == "1" ]] || return 0

  if [[ "${FORCE_INSTALL_DEPS}" == "1" ]] || ! python_deps_ok; then
    [[ -f "${REQUIREMENTS_FILE}" ]] || die "requirements file not found: ${REQUIREMENTS_FILE}"
    mkdir -p "${PIP_CACHE_DIR}"
    log "Installing Python dependencies from ${REQUIREMENTS_FILE}"
    "${PYTHON_BIN}" -m pip install --cache-dir "${PIP_CACHE_DIR}" -r "${REQUIREMENTS_FILE}"
  else
    log "Python dependencies already available"
  fi
}

ensure_layout() {
  [[ -d "${PROJECT_DIR}" ]] || die "project dir not found: ${PROJECT_DIR}"
  [[ -d "${WORKSPACE_ROOT}/DATA" ]] || die "data dir not found: ${WORKSPACE_ROOT}/DATA"
  [[ -f "${BACKBONE_PATH}" ]] || die "ResNet-34 weights not found: ${BACKBONE_PATH}"

  mkdir -p \
    "${GAN_OUTPUT_ROOT}" \
    "${SEMI_CHECKPOINT_ROOT}" \
    "${SEMI_RESULT_ROOT}" \
    "${SUMMARY_ROOT}" \
    "${LOG_ROOT}/gan" \
    "${LOG_ROOT}/semi" \
    "$(dirname "${BACKBONE_PATH}")" \
    "$(dirname "${SCD_ACTIVE_PATH}")" \
    "${SCD_ARCHIVE_ROOT}"
}

initialize_test_summary() {
  cat > "${TEST_SUMMARY_FILE}" <<'EOF'
# Test Results

| dataset | expid | ckpt_name | recall | specificity | precision | F1 | F2 | ACC_overall | IoU_poly | IoU_bg | IoU_mean | dice | DSC_all | Jacc_all | HD95_all | ASD_all | DSC_PS | Jacc_PS | HD95_PS | ASD_PS | DSC_FH | Jacc_FH | HD95_FH | ASD_FH |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
EOF
}

latest_paired_gan_epoch() {
  local exp_dir="$1"
  local expected_epoch="$2"
  local best_epoch=-1
  local path
  local base
  local epoch

  [[ -d "${exp_dir}" ]] || return 0
  shopt -s nullglob
  for path in "${exp_dir}"/netD_epoch_*.pth; do
    base="$(basename "${path}")"
    epoch="${base#netD_epoch_}"
    epoch="${epoch%.pth}"
    [[ "${epoch}" =~ ^[0-9]+$ ]] || continue
    (( epoch < expected_epoch )) || continue
    [[ -f "${exp_dir}/netG_epoch_${epoch}.pth" ]] || continue
    if (( epoch > best_epoch )); then
      best_epoch="${epoch}"
    fi
  done
  shopt -u nullglob

  if (( best_epoch >= 0 )); then
    printf '%s\n' "${best_epoch}"
  fi
}

clear_gan_checkpoints() {
  local exp_dir="$1"
  local path
  local removed=0

  [[ -d "${exp_dir}" ]] || return 0
  shopt -s nullglob
  for path in "${exp_dir}"/netG_epoch_*.pth "${exp_dir}"/netD_epoch_*.pth; do
    rm -f -- "${path}"
    removed=$((removed + 1))
  done
  shopt -u nullglob

  if (( removed > 0 )); then
    log "Cleared ${removed} GAN checkpoint file(s) in ${exp_dir}; GAN_RESUME_FROM_LATEST=${GAN_RESUME_FROM_LATEST} requests a fresh GAN run"
  fi
}

run_gan() {
  local semi_dataset="$1"
  local expid="$2"
  local gan_dataset="$3"
  local exp_dir="${GAN_OUTPUT_ROOT}/${gan_dataset}/exp${expid}"
  local expected_ckpt="${exp_dir}/netD_epoch_${GAN_EXPECTED_EPOCH}.pth"
  local log_file="${LOG_ROOT}/gan/${gan_dataset}_exp${expid}.log"
  local resume_epoch=""
  local resume_next_epoch=""
  local -a cmd=()

  mkdir -p "${exp_dir}"

  if [[ "${SKIP_GAN}" == "1" ]]; then
    [[ -f "${expected_ckpt}" ]] || die "SKIP_GAN=1 but GAN checkpoint missing: ${expected_ckpt}"
    log "SKIP_GAN=1: reusing existing GAN ckpt ${expected_ckpt} for ${gan_dataset} exp${expid}"
    return 0
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${expected_ckpt}" ]]; then
    log "Skipping GAN for ${gan_dataset} exp${expid}; found ${expected_ckpt}"
    return 0
  fi

  if [[ "${GAN_RESUME_FROM_LATEST}" != "1" ]]; then
    clear_gan_checkpoints "${exp_dir}"
  fi

  resolve_gan_runtime_settings "${gan_dataset}" "${expid}"

  if [[ "${GAN_RESUME_FROM_LATEST}" == "1" ]]; then
    resume_epoch="$(latest_paired_gan_epoch "${exp_dir}" "${GAN_EXPECTED_EPOCH}")"
    if [[ -n "${resume_epoch}" ]]; then
      resume_next_epoch=$((resume_epoch + 1))
      if (( resume_next_epoch < GAN_NITER )); then
        log "Resuming GAN for ${gan_dataset} exp${expid} from paired checkpoint epoch ${resume_epoch}; target epoch ${GAN_EXPECTED_EPOCH}"
      else
        log "Ignoring GAN resume checkpoint epoch ${resume_epoch}; next epoch ${resume_next_epoch} is outside niter=${GAN_NITER}"
        resume_epoch=""
      fi
    fi
  fi

  cmd=(
    "${PYTHON_BIN}" airs/GAN/main.py
    --dataset "${gan_dataset}"
    --root "${WORKSPACE_ROOT}"
    --expID "${expid}"
    --cuda
    --ngpu "${GAN_NGPU}"
    --batchSize "${GAN_RUNTIME_BATCH_SIZE}"
    --workers "${GAN_RUNTIME_WORKERS}"
    --niter "${GAN_NITER}"
    --lrD "${GAN_LR_D}"
    --lrG "${GAN_LR_G}"
    --lambda_adv "${GAN_LAMBDA_ADV}"
    --lambda_edge "${GAN_LAMBDA_EDGE}"
    --lambda_fm "${GAN_LAMBDA_FM}"
    --lambda_mask "${GAN_LAMBDA_MASK}"
    --precision "${GAN_PRECISION}"
    --warmup_gen_iterations "${GAN_WARMUP_GEN_ITERATIONS}"
    --warmup_diters "${GAN_WARMUP_DITERS}"
    --extra_diters_every "${GAN_EXTRA_DITERS_EVERY}"
    --save_every "${GAN_SAVE_EVERY}"
    --log_interval "${GAN_LOG_INTERVAL}"
    --experiment "${exp_dir}"
  )
  if [[ -n "${resume_epoch}" ]]; then
    cmd+=(
      --netG "${exp_dir}/netG_epoch_${resume_epoch}.pth"
      --netD "${exp_dir}/netD_epoch_${resume_epoch}.pth"
      --start_epoch "${resume_next_epoch}"
    )
  fi

  log "Running GAN pretraining for ${gan_dataset} exp${expid}"
  if [[ "${GAN_TF32}" == "1" ]]; then
    cmd+=(--tf32)
  fi
  if [[ ${#GAN_EXTRA_ARRAY[@]} -gt 0 ]]; then
    cmd+=("${GAN_EXTRA_ARRAY[@]}")
  fi
  (
    cd "${PROJECT_DIR}"
    log "GAN settings: gpus=${GAN_VISIBLE_DEVICES} batch=${GAN_RUNTIME_BATCH_SIZE} workers=${GAN_RUNTIME_WORKERS} niter=${GAN_NITER} lrD=${GAN_LR_D} lrG=${GAN_LR_G} lambda_adv=${GAN_LAMBDA_ADV} lambda_edge=${GAN_LAMBDA_EDGE} lambda_fm=${GAN_LAMBDA_FM} lambda_mask=${GAN_LAMBDA_MASK} precision=${GAN_PRECISION} tf32=${GAN_TF32}"
    CUDA_VISIBLE_DEVICES="${GAN_VISIBLE_DEVICES}" "${cmd[@]}"
  ) 2>&1 | tee "${log_file}"

  [[ -f "${expected_ckpt}" ]] || die "expected GAN checkpoint not found: ${expected_ckpt}"
}

handoff_scd_weight() {
  local semi_dataset="$1"
  local expid="$2"
  local gan_dataset="$3"
  local src_ckpt="${GAN_OUTPUT_ROOT}/${gan_dataset}/exp${expid}/netD_epoch_${GAN_EXPECTED_EPOCH}.pth"
  local archive_dir="${SCD_ARCHIVE_ROOT}/${semi_dataset}/exp${expid}"
  local archive_ckpt="${archive_dir}/netD_epoch_${GAN_EXPECTED_EPOCH}.pth"

  [[ -f "${src_ckpt}" ]] || die "GAN checkpoint not found for handoff: ${src_ckpt}"

  mkdir -p "${archive_dir}" "$(dirname "${SCD_ACTIVE_PATH}")"
  cp "${src_ckpt}" "${archive_ckpt}"
  cp "${src_ckpt}" "${SCD_ACTIVE_PATH}"
  log "Copied FBWA pretrained weight to ${archive_ckpt} and ${SCD_ACTIVE_PATH}"
}

run_semi() {
  local manner="$1"
  local semi_dataset="$2"
  local expid="$3"
  local ckpt_name="$4"
  local variant="$5"
  local log_file="${LOG_ROOT}/semi/${ckpt_name}_${manner}.log"
  local best_ckpt="${SEMI_CHECKPOINT_ROOT}/${ckpt_name}/best.pth"
  local result_dir="${SEMI_RESULT_ROOT}/${ckpt_name}/${manner}"
  local -a cmd=()

  if [[ "${manner}" == "semi" && "${SKIP_EXISTING}" == "1" && -f "${best_ckpt}" ]]; then
    log "Skipping semi training for ${ckpt_name}; found ${best_ckpt}"
    return 0
  fi

  if [[ "${manner}" == "semi" && "${SEMI_RUNTIME_SETTINGS_READY}" != "1" ]]; then
    resolve_semi_runtime_settings "${semi_dataset}" "${expid}" "${SEMI_RUNTIME_BENCHMARK_VARIANT}"
    SEMI_RUNTIME_SETTINGS_READY="1"
  fi

  cmd=(
    "${PYTHON_BIN}" airs/semi/code/main.py
    --manner "${manner}"
    --dataset "${semi_dataset}"
    --root "${WORKSPACE_ROOT}"
    --expID "${expid}"
    --GPUs "${SEMI_GPUS}"
    --batch_size "${SEMI_RUNTIME_BATCH_SIZE}"
    --num_workers "${SEMI_RUNTIME_WORKERS}"
    --nEpoch "${SEMI_EPOCHS}"
    --lr "${SEMI_LR}"
    --precision "${SEMI_PRECISION}"
    --ckpt_name "${ckpt_name}"
  )

  if [[ "${SEMI_TF32}" == "1" ]]; then
    cmd+=(--tf32)
  fi
  while IFS= read -r variant_arg; do
    [[ -n "${variant_arg}" ]] || continue
    cmd+=("${variant_arg}")
  done < <(variant_extra_args "${variant}")
  while IFS= read -r override_arg; do
    [[ -n "${override_arg}" ]] || continue
    cmd+=("${override_arg}")
  done < <(semi_append_optional_override_args)
  if [[ ${#SEMI_EXTRA_ARRAY[@]} -gt 0 ]]; then
    cmd+=("${SEMI_EXTRA_ARRAY[@]}")
  fi

  log "Running semi ${manner} for ${semi_dataset} exp${expid} -> ${ckpt_name} (gpus=${SEMI_VISIBLE_DEVICES} batch=${SEMI_RUNTIME_BATCH_SIZE} workers=${SEMI_RUNTIME_WORKERS} epochs=${SEMI_EPOCHS} lr=${SEMI_LR} precision=${SEMI_PRECISION} tf32=${SEMI_TF32})"
  log "Semi export bundle: export $(semi_export_string)"
  (
    cd "${PROJECT_DIR}"
    mkdir -p "${result_dir}"
    export AIRS_SEMI_CHECKPOINT_ROOT="${SEMI_CHECKPOINT_ROOT}"
    export AIRS_BACKBONE_PRETRAIN_PATH="${BACKBONE_PATH}"
    export AIRS_SCD_PRETRAIN_PATH="${SCD_ACTIVE_PATH}"
    export AIRS_SEMI_RESULT_DIR="${result_dir}"
    export AIRS_TEST_SUMMARY_FILE="${TEST_SUMMARY_FILE}"
    CUDA_VISIBLE_DEVICES="${SEMI_VISIBLE_DEVICES}" "${cmd[@]}"
  ) 2>&1 | tee "${log_file}"
}

run_pipeline_entry() {
  local raw_entry="$1"
  local dataset_part="${raw_entry%%:*}"
  local expid_part="${raw_entry##*:}"
  local semi_dataset
  local gan_dataset
  local ckpt_name
  local variant
  local benchmark_variant
  local need_scd="0"
  local -a variants=()

  [[ "${raw_entry}" == *:* ]] || die "invalid config entry '${raw_entry}'; expected DATASET:EXPID"

  semi_dataset="$(normalize_dataset "${dataset_part}")" || die "unsupported dataset '${dataset_part}'"
  validate_expid "${semi_dataset}" "${expid_part}" || die "unsupported expID '${expid_part}' for ${semi_dataset}"

  gan_dataset="$(gan_dataset_for "${semi_dataset}")"
  while IFS= read -r variant; do
    [[ -n "${variant}" ]] || continue
    variants+=("${variant}")
    if variant_uses_scd "${variant}"; then
      need_scd="1"
    fi
  done < <(build_variant_entries)

  if [[ ${#variants[@]} -eq 0 ]]; then
    die "no semi variants resolved for ${semi_dataset} exp${expid_part}"
  fi

  benchmark_variant="$(select_semi_benchmark_variant "${variants[@]}")"
  SEMI_RUNTIME_BATCH_SIZE="${SEMI_BATCH_SIZE}"
  SEMI_RUNTIME_WORKERS="${SEMI_NUM_WORKERS}"
  SEMI_RUNTIME_SETTINGS_READY="0"
  SEMI_RUNTIME_BENCHMARK_VARIANT="${benchmark_variant}"
  log "Semi variant plan for ${semi_dataset} exp${expid_part}: requested=${variants[*]} benchmark=${benchmark_variant}"

  if [[ "${need_scd}" == "1" ]]; then
    run_gan "${semi_dataset}" "${expid_part}" "${gan_dataset}"
    handoff_scd_weight "${semi_dataset}" "${expid_part}" "${gan_dataset}"
  else
    log "Skipping GAN for ${semi_dataset} exp${expid_part}; selected variants do not use FBWA"
  fi

  for variant in "${variants[@]}"; do
    ckpt_name="${CKPT_PREFIX}_$(lower_dataset "${semi_dataset}")_exp${expid_part}_${variant}"
    run_semi semi "${semi_dataset}" "${expid_part}" "${ckpt_name}" "${variant}"
    if [[ "${RUN_TEST}" == "1" ]]; then
      run_semi test "${semi_dataset}" "${expid_part}" "${ckpt_name}" "${variant}"
    fi
  done
}

run_pipeline() {
  local configs_raw="${PIPELINE_CONFIGS//,/ }"
  local -a raw_entries=()
  local -a entries=()
  local raw_entry
  local entry

  read -r -a raw_entries <<< "${configs_raw}"
  for raw_entry in "${raw_entries[@]}"; do
    while IFS= read -r entry; do
      [[ -n "${entry}" ]] || continue
      entries+=("${entry}")
    done < <(expand_pipeline_entry "${raw_entry}")
  done
  [[ ${#entries[@]} -gt 0 ]] || die "PIPELINE_CONFIGS is empty"

  ensure_python_deps
  ensure_layout
  log_repo_revision "${PROJECT_DIR}"
  log_pipeline_configuration
  if [[ "${RUN_TEST}" == "1" ]]; then
    initialize_test_summary
  fi

  for entry in "${entries[@]}"; do
    run_pipeline_entry "${entry}"
  done

  log "Pipeline completed"
  log "GAN outputs: ${GAN_OUTPUT_ROOT}"
  log "Semi checkpoints: ${SEMI_CHECKPOINT_ROOT}"
  log "Semi visual results: ${SEMI_RESULT_ROOT}"
  if [[ "${RUN_TEST}" == "1" ]]; then
    log "Test summary markdown: ${TEST_SUMMARY_FILE}"
    generate_comparison_report
  fi
  log "Logs: ${LOG_ROOT}"
}

generate_comparison_report() {
  local report_script="${SCRIPT_DIR}/generate_comparison_report.py"
  local report_file="${SUMMARY_ROOT}/comparison_report.md"
  if [[ ! -f "${report_script}" ]]; then
    log "Skipping comparison report: ${report_script} not found"
    return 0
  fi
  log "Generating comparison report -> ${report_file}"
  (
    cd "${PROJECT_DIR}"
    "${PYTHON_BIN}" "${report_script}" "${TEST_SUMMARY_FILE}" "${report_file}"
  ) 2>&1 | tee -a "${LOG_ROOT}/semi/comparison_report.log" || {
    log "WARNING: comparison report generation failed (non-fatal)"
  }
  if [[ -f "${report_file}" ]]; then
    log "Comparison report: ${report_file}"
  fi
}

case "${MODE}" in
  help|-h|--help)
    print_help
    ;;
  run)
    run_pipeline
    ;;
  deps)
    ensure_python_deps
    ;;
  *)
    die "unknown mode '${MODE}'"
    ;;
esac
