#!/usr/bin/env python3
"""Follower role (rank=1, CPU) for heterogeneous asymmetric DDP demo.

Run this in terminal B after trainer starts:
  python ddp_hetero_follower.py --master-addr 127.0.0.1 --master-port 29621
"""

from __future__ import annotations

import argparse

from ddp_hetero_common import run_hetero_role


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hetero asymmetric DDP follower role")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29621)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--target-seconds", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--in-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--out-dim", type=int, default=1024)
    parser.add_argument("--log-interval", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_hetero_role(
        rank=1,
        world_size=args.world_size,
        trainer_rank=0,
        master_addr=args.master_addr,
        master_port=args.master_port,
        steps=args.steps,
        target_seconds=args.target_seconds,
        batch_size=args.batch_size,
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()

