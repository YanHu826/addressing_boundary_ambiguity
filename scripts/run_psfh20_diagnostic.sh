#!/usr/bin/env bash
# PSFH 20% diagnostic sweep: 8 parallel single-GPU experiments testing each
# hypothesis for why Ours underperforms BiPCC at higher label ratio.
#
# Each variant changes ONE thing vs the Full Ours configuration so we can
# attribute the gain/loss cleanly. 400 epochs to be safe.
#
# Layout (8 H100):
#   GPU 0 -> control      (Full Ours, 400 epochs)             → tests #1 (needs more training)
#   GPU 1 -> no_strong    (no --use_strong_aug)                → tests #2 (FixMatch hurts 20%)
#   GPU 2 -> no_bcp       (no --use_bcp)                       → tests #3 (BCP hurts 20%)
#   GPU 3 -> fixed_wsdice (--wsdice_mode fixed)                → tests #4 (adaptive WSDice hurts)
#   GPU 4 -> no_wgan      (--adv_weight 0)                     → tests #5 (shape critic hurts 20%)
#   GPU 5 -> mt999        (--mt 0.999)                         → tests #6 (teacher EMA too fast)
#   GPU 6 -> lr2e4        (--lr 2e-4, doubled)                 → tests #7 (LR too small)
#   GPU 7 -> pure         (only adaptive, no strong/bcp/wgan)  → bonus: Less-is-More at 20%
#
# Wall clock: ~20h
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/outputs/logs/psfh20_diag"
mkdir -p "${LOG_DIR}"

launch_exp() {
  local prefix="$1"
  local gpu="$2"
  local extra_args="$3"
  local epochs="${4:-400}"
  local lr="${5:-1e-4}"
  local timestamp
  timestamp="$(date '+%Y%m%d_%H%M%S')"
  local log_file="${LOG_DIR}/${prefix}_${timestamp}.log"
  echo "[$(date '+%F %T')] Launching ${prefix} on GPU ${gpu}  epochs=${epochs} lr=${lr}"
  echo "                       extra=${extra_args}"
  (
    export SEMI_BATCH_SIZE=16
    export SEMI_NUM_WORKERS=8
    export SEMI_AUTOTUNE=0
    export SEMI_EPOCHS="${epochs}"
    export SEMI_LR="${lr}"
    export SEMI_VISIBLE_DEVICES="${gpu}"
    export SEMI_EXTRA_ARGS="${extra_args}"
    export PIPELINE_CONFIGS="PSFH:2"
    export CKPT_PREFIX="${prefix}"
    export SKIP_GAN=1   # reuse existing PSFH exp2 GAN ckpt
    bash scripts/run_training_pipeline.sh run \
      > "${log_file}" 2>&1
  ) &
  echo $! > "${LOG_DIR}/${prefix}.pid"
}

echo "================================================================"
echo "PSFH 20% Diagnostic Sweep  start=$(date '+%F %T')"
echo "================================================================"

# Control: same as failing config but 400 epochs
launch_exp "psfh20_control"  "0" "--use_strong_aug --use_bcp --wsdice_mode adaptive"

# Hypothesis 2: strong aug hurts at 20%
launch_exp "psfh20_no_strong"  "1" "--use_bcp --wsdice_mode adaptive"

# Hypothesis 3: BCP hurts at 20%
launch_exp "psfh20_no_bcp"  "2" "--use_strong_aug --wsdice_mode adaptive"

# Hypothesis 4: adaptive WSDice fails at 20%
launch_exp "psfh20_fixed"  "3" "--use_strong_aug --use_bcp --wsdice_mode fixed"

# Hypothesis 5: WGAN-GP DSR hurts at 20%
launch_exp "psfh20_no_wgan"  "4" "--use_strong_aug --use_bcp --wsdice_mode adaptive --adv_weight 0"

# Hypothesis 6: teacher EMA too fast
launch_exp "psfh20_mt999"  "5" "--use_strong_aug --use_bcp --wsdice_mode adaptive --mt 0.999"

# Hypothesis 7: LR too small
launch_exp "psfh20_lr2e4"  "6" "--use_strong_aug --use_bcp --wsdice_mode adaptive" 400 "2e-4"

# Bonus: Pure Ours (Less-is-More) at 20%
launch_exp "psfh20_pure"  "7" "--wsdice_mode adaptive --adv_weight 0"

echo
echo "All 8 diagnostic experiments launched. Wall clock target: ~20h"
echo "Tail any: tail -f ${LOG_DIR}/psfh20_<variant>_<timestamp>.log"
echo

wait
echo "[$(date '+%F %T')] All PSFH 20% diagnostics complete"
echo
echo "=== Final Best Dice per variant ==="
for prefix in psfh20_control psfh20_no_strong psfh20_no_bcp psfh20_fixed psfh20_no_wgan psfh20_mt999 psfh20_lr2e4 psfh20_pure; do
  latest=$(ls -t "${LOG_DIR}/${prefix}_"*.log 2>/dev/null | head -1)
  if [[ -n "$latest" ]]; then
    best=$(grep -oE "Best Dice:: [0-9.]+" "$latest" | tail -1)
    echo "${prefix}: ${best}"
  fi
done
