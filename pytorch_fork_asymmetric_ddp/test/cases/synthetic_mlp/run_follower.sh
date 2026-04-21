#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export MASTER_ADDR="${MASTER_ADDR:-10.60.82.27}"
export MASTER_PORT="${MASTER_PORT:-29621}"
export WORLD_SIZE="${WORLD_SIZE:-2}"
export RANK="${RANK:-1}"
export STEPS="${STEPS:-1000}"
export LOG_INTERVAL="${LOG_INTERVAL:-100}"
export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-200}"
export SAVE_DIR="${SAVE_DIR:-/tmp/ddp_hetero_ckpt}"

export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_DDP_ASYMMETRIC_MODE="${TORCH_DDP_ASYMMETRIC_MODE:-1}"
export TORCH_DDP_TRAINER_RANK="${TORCH_DDP_TRAINER_RANK:-0}"
export TORCH_DDP_SKIP_ALLREDUCE="${TORCH_DDP_SKIP_ALLREDUCE:-1}"
export TORCH_DDP_NON_TRAINER_FORWARD_ONLY="${TORCH_DDP_NON_TRAINER_FORWARD_ONLY:-1}"
export TORCH_DDP_NON_TRAINER_BACKWARD="${TORCH_DDP_NON_TRAINER_BACKWARD:-error}"
export TORCH_DDP_SYNC_INTERVAL="${TORCH_DDP_SYNC_INTERVAL:-1}"
export TORCH_DDP_HETERO_PARAM_SYNC="${TORCH_DDP_HETERO_PARAM_SYNC:-1}"

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

echo "[follower] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE RANK=$RANK IFACE=$GLOO_SOCKET_IFNAME STEPS=$STEPS LOG_INTERVAL=$LOG_INTERVAL SAVE_EVERY_STEPS=$SAVE_EVERY_STEPS SAVE_DIR=$SAVE_DIR"

python3 ddp_hetero_role.py \
  --rank "$RANK" \
  --trainer-rank "$TORCH_DDP_TRAINER_RANK" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  --world-size "$WORLD_SIZE" \
  --steps "$STEPS" \
  --log-interval "$LOG_INTERVAL" \
  --save-every-steps "$SAVE_EVERY_STEPS" \
  --save-dir "$SAVE_DIR"

if [[ -d "$SAVE_DIR" ]]; then
  echo "[follower] saved_checkpoints:"
  ls -lh "$SAVE_DIR"/follower_step_*.pt 2>/dev/null || true
fi