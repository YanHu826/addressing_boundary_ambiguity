#!/usr/bin/env bash
# PSFH 20% diagnostic sweep V2 — now with SGD/LR variants.
# 之前 V1 全部用 Adam + lr=1e-4，没法验证 optimizer 是否是根因。
# V2 加入 SGD + 大 LR 来直接对比 BiPCC/Shape-Prior 的优化器配置。
#
# Layout (8 H100):
#   GPU 0 -> control       (Adam + lr=1e-4, 现有配置)
#   GPU 1 -> sgd_shape     (SGD + lr=1e-3, Shape-Prior 配置)
#   GPU 2 -> sgd_bipcc     (SGD + lr=1e-2, BiPCC 配置)
#   GPU 3 -> adam_lr1e3    (Adam + lr=1e-3, 提高 Adam LR)
#   GPU 4 -> no_strong     (Adam + 当前 LR，关 strong aug)
#   GPU 5 -> no_bcp        (Adam + 当前 LR，关 BCP)
#   GPU 6 -> no_wgan       (Adam + 当前 LR，--adv_weight 0)
#   GPU 7 -> pure          (Adam + 当前 LR，只 adaptive WSDice)
#
# Wall clock: ~20h (PSFH 单 ratio 单 GPU ~12-15h)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/outputs/logs/psfh20_diag2"
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
    export SEMI_NUM_WORKERS=6
    export SEMI_AUTOTUNE=0
    export SEMI_EPOCHS="${epochs}"
    export SEMI_LR="${lr}"
    export SEMI_VISIBLE_DEVICES="${gpu}"
    export SEMI_EXTRA_ARGS="${extra_args}"
    export PIPELINE_CONFIGS="PSFH:2"
    export CKPT_PREFIX="${prefix}"
    export SKIP_GAN=1
    bash scripts/run_training_pipeline.sh run \
      > "${log_file}" 2>&1
  ) &
  echo $! > "${LOG_DIR}/${prefix}.pid"
}

echo "================================================================"
echo "PSFH 20% Diagnostic V2 (with SGD variants)  start=$(date '+%F %T')"
echo "================================================================"

# Control: 当前配置 (Adam + lr=1e-4) + 400 epochs
launch_exp "psfh20v2_control"   "0" "--use_strong_aug --use_bcp --wsdice_mode adaptive"

# 关键: SGD + Shape-Prior LR
launch_exp "psfh20v2_sgd_shape" "1" "--use_strong_aug --use_bcp --wsdice_mode adaptive --optim sgd --momentum 0.9" 400 "1e-3"

# 关键: SGD + BiPCC LR
launch_exp "psfh20v2_sgd_bipcc" "2" "--use_strong_aug --use_bcp --wsdice_mode adaptive --optim sgd --momentum 0.9" 400 "1e-2"

# Adam + 提高 LR
launch_exp "psfh20v2_adam_lr1e3" "3" "--use_strong_aug --use_bcp --wsdice_mode adaptive" 400 "1e-3"

# 关 strong aug
launch_exp "psfh20v2_no_strong"  "4" "--use_bcp --wsdice_mode adaptive"

# 关 BCP
launch_exp "psfh20v2_no_bcp"     "5" "--use_strong_aug --wsdice_mode adaptive"

# 关 WGAN-GP
launch_exp "psfh20v2_no_wgan"    "6" "--use_strong_aug --use_bcp --wsdice_mode adaptive --adv_weight 0"

# Pure Ours (Less-is-More)
launch_exp "psfh20v2_pure"       "7" "--wsdice_mode adaptive --adv_weight 0"

echo
echo "All 8 V2 diagnostic experiments launched. Wall clock target: ~20h"
echo

wait
echo "[$(date '+%F %T')] All PSFH 20% V2 diagnostics complete"
echo
echo "=== Final Best Dice per variant ==="
for prefix in psfh20v2_control psfh20v2_sgd_shape psfh20v2_sgd_bipcc psfh20v2_adam_lr1e3 \
              psfh20v2_no_strong psfh20v2_no_bcp psfh20v2_no_wgan psfh20v2_pure; do
  latest=$(ls -t "${LOG_DIR}/${prefix}_"*.log 2>/dev/null | head -1)
  if [[ -n "$latest" ]]; then
    best=$(grep -oE "Best Dice:: [0-9.]+" "$latest" | tail -1)
    echo "${prefix}: ${best}"
  fi
done
