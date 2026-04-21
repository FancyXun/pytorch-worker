#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export MASTER_ADDR="${MASTER_ADDR:-10.60.82.27}"
export MASTER_PORT="${MASTER_PORT:-29623}"
export WORLD_SIZE="${WORLD_SIZE:-2}"
export RANK="${RANK:-1}"

export EPOCHS="${EPOCHS:-}"
export LOG_INTERVAL="${LOG_INTERVAL:-0}"
export SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-10}"
export SAVE_DIR="${SAVE_DIR:-/tmp/stgnn_hetero_ckpt}"
export INIT_TIMEOUT_SEC="${INIT_TIMEOUT_SEC:-90}"

export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_DDP_ASYMMETRIC_MODE="${TORCH_DDP_ASYMMETRIC_MODE:-1}"
export TORCH_DDP_TRAINER_RANK="${TORCH_DDP_TRAINER_RANK:-0}"
export TORCH_DDP_SKIP_ALLREDUCE="${TORCH_DDP_SKIP_ALLREDUCE:-1}"
export TORCH_DDP_NON_TRAINER_FORWARD_ONLY="${TORCH_DDP_NON_TRAINER_FORWARD_ONLY:-1}"
export TORCH_DDP_NON_TRAINER_BACKWARD="${TORCH_DDP_NON_TRAINER_BACKWARD:-error}"
export TORCH_DDP_SYNC_INTERVAL="${TORCH_DDP_SYNC_INTERVAL:-1}"
export TORCH_DDP_HETERO_PARAM_SYNC="${TORCH_DDP_HETERO_PARAM_SYNC:-1}"
export SKIP_FOLLOWER_FORWARD="${SKIP_FOLLOWER_FORWARD:-1}"

if [[ -z "${GLOO_SOCKET_IFNAME:-}" ]] && command -v ip >/dev/null 2>&1; then
  GLOO_SOCKET_IFNAME="$(
    ip route get "$MASTER_ADDR" 2>/dev/null | awk '{for (i = 1; i <= NF; ++i) if ($i == "dev") {print $(i + 1); exit}}' || true
  )"
fi
if [[ -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
  if [[ -d /sys/class/net/eth0 ]]; then
    GLOO_SOCKET_IFNAME="eth0"
  else
    GLOO_SOCKET_IFNAME="$(
      ls /sys/class/net 2>/dev/null | awk '$0 != "lo" {print; exit}' || true
    )"
  fi
  if [[ -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
    echo "ERROR: unable to infer GLOO_SOCKET_IFNAME; please export it explicitly." >&2
    exit 2
  fi
  echo "WARN: iproute2 missing or no route match; inferred GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME" >&2
fi
export GLOO_SOCKET_IFNAME

echo "[stgnn/follower] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT RANK=$RANK TRAINER_RANK=$TORCH_DDP_TRAINER_RANK IFACE=$GLOO_SOCKET_IFNAME INIT_TIMEOUT_SEC=$INIT_TIMEOUT_SEC SAVE_DIR=$SAVE_DIR"

EXTRA=()
if [[ -n "${EPOCHS}" ]]; then
  EXTRA+=(--epochs "${EPOCHS}")
fi

python3 ddp_train.py \
  --rank "$RANK" \
  --trainer-rank "$TORCH_DDP_TRAINER_RANK" \
  --world-size "$WORLD_SIZE" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  --init-timeout-sec "$INIT_TIMEOUT_SEC" \
  --log-interval "$LOG_INTERVAL" \
  --save-every-epochs "$SAVE_EVERY_EPOCHS" \
  --save-dir "$SAVE_DIR" \
  "${EXTRA[@]}"

if [[ -d "$SAVE_DIR" ]] && [[ "${SAVE_EVERY_EPOCHS}" != "0" ]]; then
  echo "[stgnn/follower] saved_checkpoints:"
  ls -lh "$SAVE_DIR"/follower_epoch_*.pt 2>/dev/null || true
fi
