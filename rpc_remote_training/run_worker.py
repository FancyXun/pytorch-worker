#!/usr/bin/env python3
"""
GPU / worker side — must start BEFORE the driver.

From repo root (torch-worker):

  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT=29500
  python -m rpc_remote_training.run_worker

This process uses CUDA (install GPU PyTorch). Blocks until the driver calls rpc.shutdown().
"""
from __future__ import annotations

import argparse
import os
import sys

# Before torch: so you always see something even if import hangs (CUDA/driver issues).
print("[rpc_remote_training worker] process started; loading PyTorch (first import can take 10–60s)…", flush=True)

import torch.distributed.rpc as rpc
from torch.distributed.rpc import TensorPipeRpcBackendOptions

# Ensure RPC can resolve worker_ops.* on this process
import rpc_remote_training.worker_ops  # noqa: F401

print("[rpc_remote_training worker] imports done.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage1 RPC worker (rank=1, GPU)")
    parser.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1"))
    parser.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT", "29500")))
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--name", default="gpu_worker")
    parser.add_argument("--rpc-timeout", type=int, default=600)
    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)

    print(
        f"[rpc_remote_training worker] rendezvous MASTER_ADDR={args.master_addr!r} MASTER_PORT={args.master_port}",
        flush=True,
    )
    if args.master_addr in ("127.0.0.1", "localhost"):
        print(
            "[rpc_remote_training worker] NOTE: driver in Docker must use the host LAN IP for MASTER_ADDR, "
            "and the worker on the host must use the SAME value (not 127.0.0.1).",
            flush=True,
        )

    opts = TensorPipeRpcBackendOptions(
        num_worker_threads=8,
        rpc_timeout=args.rpc_timeout,
    )
    print(
        "[rpc_remote_training worker] calling rpc.init_rpc — this BLOCKS here until the driver (rank=0) "
        "also calls init_rpc on the same MASTER_ADDR/PORT. Start the container driver next.",
        flush=True,
    )
    sys.stdout.flush()
    rpc.init_rpc(
        args.name,
        rank=args.rank,
        world_size=args.world_size,
        rpc_backend_options=opts,
    )
    print(
        f"[rpc_remote_training worker] name={args.name!r} rank={args.rank} world_size={args.world_size} "
        f"master={args.master_addr}:{args.master_port} — RPC ready; shutdown() waits for driver to finish.",
        flush=True,
    )
    rpc.shutdown()


if __name__ == "__main__":
    main()
