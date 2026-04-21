#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export DEVICE_MODE="${DEVICE_MODE:-auto}" # auto | gpu | cpu
export EPOCHS="${EPOCHS:-}"               # optional: override config epochs
export LOG_FILE="${LOG_FILE:-}"           # optional: write output to file

if [[ "${DEVICE_MODE}" == "cpu" ]]; then
  # Force CPU path in train.py.
  export CUDA_VISIBLE_DEVICES=""
elif [[ "${DEVICE_MODE}" == "gpu" ]]; then
  unset CUDA_VISIBLE_DEVICES || true
elif [[ "${DEVICE_MODE}" != "auto" ]]; then
  echo "ERROR: DEVICE_MODE must be one of: auto, gpu, cpu" >&2
  exit 2
fi

if [[ -n "${LOG_FILE}" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  exec > >(tee -a "${LOG_FILE}") 2>&1
fi

echo "[stgnn/local] DEVICE_MODE=${DEVICE_MODE} EPOCHS=${EPOCHS:-<config>} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

EXTRA=()
if [[ -n "${EPOCHS}" ]]; then
  EXTRA+=(--epochs "${EPOCHS}")
fi

python3 train.py "${EXTRA[@]}"
