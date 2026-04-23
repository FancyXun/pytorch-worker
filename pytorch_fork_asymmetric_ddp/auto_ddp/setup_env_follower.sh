#!/usr/bin/env bash
# Source this file, then run the user command unchanged.
# Example:
#   source /path/to/auto_ddp/setup_env_follower.sh && python3 train.py

_AUTO_DDP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_AUTO_DDP_ROOT="$(cd "${_AUTO_DDP_DIR}/.." && pwd)"

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
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export PYTHONPATH="${_AUTO_DDP_DIR}:${_AUTO_DDP_ROOT}/pytorch:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""

if [[ "${WORLD_SIZE}" != "1" && "${MASTER_ADDR}" == "127.0.0.1" && "${ALLOW_LOOPBACK_MASTER:-0}" != "1" ]]; then
  echo "ERROR: MASTER_ADDR=127.0.0.1 is loopback and only works when all ranks are on the same host." >&2
  echo "Set MASTER_ADDR to trainer host IP (or export ALLOW_LOOPBACK_MASTER=1 for single-host local test)." >&2
  return 2 2>/dev/null || exit 2
fi

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
fi
if [[ -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
  echo "ERROR: unable to infer GLOO_SOCKET_IFNAME; please export it explicitly." >&2
  return 2 2>/dev/null || exit 2
fi
export GLOO_SOCKET_IFNAME

echo "[auto-ddp env follower] master=${MASTER_ADDR}:${MASTER_PORT} rank=${RANK} trainer_rank=${TORCH_DDP_TRAINER_RANK} world_size=${WORLD_SIZE} iface=${GLOO_SOCKET_IFNAME}"
