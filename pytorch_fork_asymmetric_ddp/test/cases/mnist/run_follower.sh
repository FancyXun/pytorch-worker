#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export MASTER_ADDR="${MASTER_ADDR:-10.60.82.27}"
export MASTER_PORT="${MASTER_PORT:-29621}"
export WORLD_SIZE="${WORLD_SIZE:-2}"
export RANK="${RANK:-1}"

export EPOCHS="${EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:-1000}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export DATA_DIR="${DATA_DIR:-/tmp/mnist_data}"
export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-200}"
export SAVE_DIR="${SAVE_DIR:-/tmp/ddp_hetero_mnist_ckpt}"

export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_DDP_ASYMMETRIC_MODE="${TORCH_DDP_ASYMMETRIC_MODE:-1}"
export TORCH_DDP_TRAINER_RANK="${TORCH_DDP_TRAINER_RANK:-0}"
export TORCH_DDP_SKIP_ALLREDUCE="${TORCH_DDP_SKIP_ALLREDUCE:-1}"
export TORCH_DDP_NON_TRAINER_FORWARD_ONLY="${TORCH_DDP_NON_TRAINER_FORWARD_ONLY:-1}"
export TORCH_DDP_NON_TRAINER_BACKWARD="${TORCH_DDP_NON_TRAINER_BACKWARD:-error}"
export TORCH_DDP_SYNC_INTERVAL="${TORCH_DDP_SYNC_INTERVAL:-1}"
export TORCH_DDP_HETERO_PARAM_SYNC="${TORCH_DDP_HETERO_PARAM_SYNC:-1}"

# Minimal images often lack `ip` (iproute2); with set -e a missing `ip` would exit before any echo.
if [[ -z "${GLOO_SOCKET_IFNAME:-}" ]] && command -v ip >/dev/null 2>&1; then
  GLOO_SOCKET_IFNAME="$(
    ip route get "$MASTER_ADDR" 2>/dev/null | awk '{for (i = 1; i <= NF; ++i) if ($i == "dev") {print $(i + 1); exit}}' || true
  )"
fi
if [[ -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
  GLOO_SOCKET_IFNAME="eth0"
  echo "WARN: set GLOO_SOCKET_IFNAME explicitly if needed; defaulting to eth0 (no usable iproute2 or no route match)." >&2
fi
export GLOO_SOCKET_IFNAME

echo "[mnist/follower] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT RANK=$RANK IFACE=$GLOO_SOCKET_IFNAME MAX_STEPS=$MAX_STEPS SAVE_DIR=$SAVE_DIR"

python3 ddp_hetero_mnist.py \
  --rank "$RANK" \
  --trainer-rank "$TORCH_DDP_TRAINER_RANK" \
  --world-size "$WORLD_SIZE" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  --epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --log-interval "$LOG_INTERVAL" \
  --data-dir "$DATA_DIR" \
  --save-every-steps "$SAVE_EVERY_STEPS" \
  --save-dir "$SAVE_DIR"

if [[ -d "$SAVE_DIR" ]]; then
  echo "[mnist/follower] saved_checkpoints:"
  ls -lh "$SAVE_DIR"/follower_mnist_step_*.pt 2>/dev/null || true
fi
