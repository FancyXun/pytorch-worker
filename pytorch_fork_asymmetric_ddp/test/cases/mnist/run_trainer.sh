#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export MASTER_ADDR="${MASTER_ADDR:-10.60.82.27}"
export MASTER_PORT="${MASTER_PORT:-29621}"
export WORLD_SIZE="${WORLD_SIZE:-2}"
export RANK="${RANK:-0}"

export EPOCHS="${EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:-1000}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-200}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export DATA_DIR="${DATA_DIR:-/tmp/mnist_data}"

export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_DDP_ASYMMETRIC_MODE="${TORCH_DDP_ASYMMETRIC_MODE:-1}"
export TORCH_DDP_TRAINER_RANK="${TORCH_DDP_TRAINER_RANK:-0}"
export TORCH_DDP_SKIP_ALLREDUCE="${TORCH_DDP_SKIP_ALLREDUCE:-1}"
export TORCH_DDP_NON_TRAINER_FORWARD_ONLY="${TORCH_DDP_NON_TRAINER_FORWARD_ONLY:-1}"
export TORCH_DDP_NON_TRAINER_BACKWARD="${TORCH_DDP_NON_TRAINER_BACKWARD:-error}"
export TORCH_DDP_SYNC_INTERVAL="${TORCH_DDP_SYNC_INTERVAL:-1}"
export TORCH_DDP_HETERO_PARAM_SYNC="${TORCH_DDP_HETERO_PARAM_SYNC:-1}"

if [[ -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
  GLOO_SOCKET_IFNAME="$(
    ip -o -4 addr show | awk -v master_ip="$MASTER_ADDR" '$0 ~ master_ip {print $2; exit}'
  )"
fi
export GLOO_SOCKET_IFNAME

echo "[mnist/trainer] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT RANK=$RANK IFACE=$GLOO_SOCKET_IFNAME MAX_STEPS=$MAX_STEPS"

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
  --eval-interval "$EVAL_INTERVAL" \
  --data-dir "$DATA_DIR"
