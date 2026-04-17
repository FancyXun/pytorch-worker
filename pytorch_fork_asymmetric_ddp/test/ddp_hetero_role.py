#!/usr/bin/env python3
"""Unified role entry for heterogeneous asymmetric DDP demo.

Usage:
  # GPU trainer (rank=0)
  python ddp_hetero_role.py --rank 0 --master-addr 10.60.82.27 --master-port 29621 --steps 3

  # CPU follower (rank=1)
  python ddp_hetero_role.py --rank 1 --master-addr 10.60.82.27 --master-port 29621 --steps 3
"""

from __future__ import annotations

import argparse

from ddp_hetero_common import run_hetero_role


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hetero asymmetric DDP unified role")
    parser.add_argument("--rank", type=int, required=True, choices=(0, 1))
    parser.add_argument("--trainer-rank", type=int, default=0, choices=(0, 1))
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29621)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--in-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--out-dim", type=int, default=1024)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--save-dir", default="/tmp/ddp_hetero_ckpt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_hetero_role(
        rank=args.rank,
        world_size=args.world_size,
        trainer_rank=args.trainer_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        steps=args.steps,
        batch_size=args.batch_size,
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        log_interval=args.log_interval,
        save_every_steps=args.save_every_steps,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()

