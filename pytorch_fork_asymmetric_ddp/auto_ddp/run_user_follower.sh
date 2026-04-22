#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <user_train.py> [script_args...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29623}"
export WORLD_SIZE="${WORLD_SIZE:-2}"
export RANK="${RANK:-1}"
export TORCH_DDP_INIT_TIMEOUT_SEC="${TORCH_DDP_INIT_TIMEOUT_SEC:-90}"
export TORCH_DDP_TRAINER_RANK="${TORCH_DDP_TRAINER_RANK:-0}"
export TORCH_DDP_AUTO_WRAP=1
export TORCH_DDP_ASYMMETRIC_MODE="${TORCH_DDP_ASYMMETRIC_MODE:-1}"
export TORCH_DDP_SKIP_ALLREDUCE="${TORCH_DDP_SKIP_ALLREDUCE:-1}"
export TORCH_DDP_HETERO_PARAM_SYNC="${TORCH_DDP_HETERO_PARAM_SYNC:-1}"
export TORCH_DDP_NON_TRAINER_FORWARD_ONLY="${TORCH_DDP_NON_TRAINER_FORWARD_ONLY:-1}"
export TORCH_DDP_NON_TRAINER_BACKWARD="${TORCH_DDP_NON_TRAINER_BACKWARD:-error}"
export TORCH_DDP_SYNC_INTERVAL="${TORCH_DDP_SYNC_INTERVAL:-1}"
export TORCH_DDP_AUTO_SKIP_FOLLOWER_FORWARD="${TORCH_DDP_AUTO_SKIP_FOLLOWER_FORWARD:-1}"
export PYTHONPATH="${ROOT_DIR}/auto_ddp:${ROOT_DIR}/pytorch:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""

if [[ "${WORLD_SIZE}" != "1" && "${MASTER_ADDR}" == "127.0.0.1" && "${ALLOW_LOOPBACK_MASTER:-0}" != "1" ]]; then
  echo "ERROR: MASTER_ADDR=127.0.0.1 is loopback and only works when all ranks are on the same host." >&2
  echo "Set MASTER_ADDR to trainer host IP (or export ALLOW_LOOPBACK_MASTER=1 for single-host local test)." >&2
  exit 2
fi

echo "[auto-ddp follower] master=${MASTER_ADDR}:${MASTER_PORT} rank=${RANK} trainer_rank=${TORCH_DDP_TRAINER_RANK} world_size=${WORLD_SIZE}"
python3 "$@"

