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
LOG_DIR="artifacts/logs/p2_${TIMESTAMP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/baseline}"
PLOTS_ROOT="${PLOTS_ROOT:-artifacts/plots}"
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

run_and_log "p2_baseline_resnet_vit" \
  src/baseline_experiments.py \
  --output-root "$OUTPUT_ROOT" \
  --plots-root "$PLOTS_ROOT" \
  --epochs-resnet 20 \
  --epochs-vit 20 \
  --num-workers 0

echo "P2 run completed successfully."
echo "Read logs in: ${LOG_DIR}"
