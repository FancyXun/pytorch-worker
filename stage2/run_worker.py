#!/usr/bin/env python3
"""
Stage2 GPU/CPU worker — start BEFORE driver.

  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT=29500
  python -m stage2.run_worker

Uses ``stage2.worker_ops`` (not stage1).
"""
from __future__ import annotations

import argparse
import os
import sys

print("[stage2 worker] process started; loading PyTorch (may take 10–60s)…", flush=True)

import torch.distributed.rpc as rpc
from torch.distributed.rpc import TensorPipeRpcBackendOptions

import stage2.worker_ops  # noqa: F401

print("[stage2 worker] imports done.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage2 RPC worker (rank=1)")
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
        f"[stage2 worker] rendezvous MASTER_ADDR={args.master_addr!r} MASTER_PORT={args.master_port}",
        flush=True,
    )
    if args.master_addr in ("127.0.0.1", "localhost"):
        print(
            "[stage2 worker] NOTE: Docker driver on bridge network should use host LAN IP "
            "as MASTER_ADDR on both sides, or use --network host.",
            flush=True,
        )

    opts = TensorPipeRpcBackendOptions(
        num_worker_threads=8,
        rpc_timeout=args.rpc_timeout,
    )
    print(
        "[stage2 worker] calling rpc.init_rpc — blocks until driver (rank=0) joins.",
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
        f"[stage2 worker] name={args.name!r} rank={args.rank} RPC ready; shutdown waits for driver.",
        flush=True,
    )
    rpc.shutdown()


if __name__ == "__main__":
    main()
