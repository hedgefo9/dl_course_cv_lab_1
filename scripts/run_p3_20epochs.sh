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
LOG_DIR="artifacts/logs/p3_${TIMESTAMP}"
IMPROVED_OUT="${IMPROVED_OUT:-artifacts/improved_baseline}"
PLOTS_ROOT="${PLOTS_ROOT:-artifacts/plots}"
SPLIT_CSV="${SPLIT_CSV:-artifacts/baseline/splits.csv}"
MASTER_LOG="${LOG_DIR}/master.log"

if [[ ! -f "$SPLIT_CSV" ]]; then
  echo "Split file not found: $SPLIT_CSV" >&2
  echo "Run p2 first or provide SPLIT_CSV=/path/to/splits.csv" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$IMPROVED_OUT" "$PLOTS_ROOT"
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
  --output-root "$IMPROVED_OUT"
  --plots-root "$PLOTS_ROOT"
  --split-csv "$SPLIT_CSV"
  --epochs-resnet 20
  --epochs-vit 20
  --num-workers 0
)

run_and_log "p3_h0_resnet_frozen_weak" \
  src/improved_baseline_experiments.py \
  --run-name h0_resnet_frozen_weak \
  --models resnet18 \
  --augmentation weak \
  --label-smoothing 0.0 \
  "${COMMON_ARGS[@]}"

run_and_log "p3_h1_resnet_unfreeze_weak" \
  src/improved_baseline_experiments.py \
  --run-name h1_resnet_unfreeze_weak \
  --models resnet18 \
  --augmentation weak \
  --trainable-backbone \
  --label-smoothing 0.0 \
  "${COMMON_ARGS[@]}"

run_and_log "p3_h2_resnet_unfreeze_strong" \
  src/improved_baseline_experiments.py \
  --run-name h2_resnet_unfreeze_strong \
  --models resnet18 \
  --augmentation strong \
  --trainable-backbone \
  --label-smoothing 0.0 \
  "${COMMON_ARGS[@]}"

run_and_log "p3_h3_resnet_unfreeze_strong_weighted_ls" \
  src/improved_baseline_experiments.py \
  --run-name h3_resnet_unfreeze_strong_weighted_ls \
  --models resnet18 \
  --augmentation strong \
  --trainable-backbone \
  --weighted-loss \
  --label-smoothing 0.1 \
  "${COMMON_ARGS[@]}"

run_and_log "p3_final_improved_resnet_vit" \
  src/improved_baseline_experiments.py \
  --run-name final_improved_baseline \
  --models resnet18 vit_b_16 \
  --augmentation weak \
  --trainable-backbone \
  --label-smoothing 0.0 \
  "${COMMON_ARGS[@]}"

echo "P3 runs completed successfully."
echo "Read logs in: ${LOG_DIR}"
