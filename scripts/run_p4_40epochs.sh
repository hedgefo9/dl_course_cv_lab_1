#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="artifacts/logs/p4_${TIMESTAMP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/custom_models}"
PLOTS_ROOT="${PLOTS_ROOT:-artifacts/plots/custom_models}"
SPLIT_CSV="${SPLIT_CSV:-artifacts/baseline/splits.csv}"
MASTER_LOG="${LOG_DIR}/master.log"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT" "$PLOTS_ROOT"
touch "$MASTER_LOG"

run_and_log() {
  local run_name="$1"
  shift

  local log_file="${LOG_DIR}/${run_name}.log"
  local started_at
  started_at="$(date +"%Y-%m-%d %H:%M:%S")"

  {
    echo "============================================================"
    echo "RUN: ${run_name}"
    echo "START: ${started_at}"
    echo "LOG: ${log_file}"
    echo "CMD: TQDM_DISABLE=1 ${PYTHON_BIN} $*"
    echo "============================================================"
  } | tee -a "$MASTER_LOG"

  set +e
  (TQDM_DISABLE=1 "$PYTHON_BIN" "$@") 2>&1 | tee "$log_file"
  local cmd_exit=${PIPESTATUS[0]}
  set -e

  local finished_at
  finished_at="$(date +"%Y-%m-%d %H:%M:%S")"
  echo "END: ${finished_at} | EXIT_CODE: ${cmd_exit} | RUN: ${run_name}" | tee -a "$MASTER_LOG"

  if [[ $cmd_exit -ne 0 ]]; then
    echo "Run failed: ${run_name}. See ${log_file}" | tee -a "$MASTER_LOG" >&2
    exit $cmd_exit
  fi
}

echo "Logs directory: ${LOG_DIR}"
echo "Master log: ${MASTER_LOG}"

COMMON_ARGS=(
  --split-csv "$SPLIT_CSV"
  --output-root "$OUTPUT_ROOT"
  --plots-root "$PLOTS_ROOT"
  --epochs-cnn 40
  --epochs-vit 40
  --num-workers 0
)

BASELINE_VIT_ARGS=(
  --batch-size-vit 32
  --vit-dim 192
  --vit-depth 8
  --vit-heads 6
  --lr-vit-baseline 5e-4
  --vit-drop-path-baseline 0.10
  --vit-label-smoothing-baseline 0.05
  --vit-mixup-alpha-baseline 0.0
  --vit-warmup-epochs-baseline 3
  --vit-min-lr-ratio-baseline 0.15
)

IMPROVED_VIT_ARGS=(
  --batch-size-vit 32
  --vit-dim 256
  --vit-depth 10
  --vit-heads 8
  --lr-vit-improved 3e-4
  --vit-drop-path-improved 0.20
  --vit-label-smoothing-improved 0.08
  --vit-mixup-alpha-improved 0.0
  --vit-warmup-epochs-improved 3
  --vit-min-lr-ratio-improved 0.05
  --improved-random-erasing-prob 0.15
)

# 4a-e: custom models without techniques from improved baseline.
run_and_log "p4_baseline_long" \
  src/custom_models_experiments.py \
  --run-name baseline_long \
  --phase baseline \
  --models custom_cnn tiny_vit \
  "${COMMON_ARGS[@]}" \
  "${BASELINE_VIT_ARGS[@]}"

# 4f-j: add techniques from point 3c.
run_and_log "p4_improved_long" \
  src/custom_models_experiments.py \
  --run-name improved_long \
  --phase improved \
  --models custom_cnn tiny_vit \
  "${COMMON_ARGS[@]}" \
  "${IMPROVED_VIT_ARGS[@]}"

echo "P4 runs completed successfully."
echo "Read logs in: ${LOG_DIR}"
