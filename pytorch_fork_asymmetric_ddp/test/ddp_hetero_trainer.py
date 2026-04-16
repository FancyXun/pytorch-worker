#!/usr/bin/env python3
"""Trainer role (rank=0, GPU) for heterogeneous asymmetric DDP demo.

Run this in terminal A first:
  python ddp_hetero_trainer.py --master-addr 127.0.0.1 --master-port 29621
"""

from __future__ import annotations

import argparse

from ddp_hetero_common import run_hetero_role


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hetero asymmetric DDP trainer role")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29621)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_hetero_role(
        rank=0,
        world_size=args.world_size,
        trainer_rank=0,
        master_addr=args.master_addr,
        master_port=args.master_port,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()

