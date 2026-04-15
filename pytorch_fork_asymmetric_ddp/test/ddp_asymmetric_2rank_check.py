#!/usr/bin/env python3
"""Two-rank functional check for asymmetric DDP modifications.

Run with:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 test/ddp_asymmetric_2rank_check.py

Optional policy check:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    test/ddp_asymmetric_2rank_check.py --non-trainer-backward error --expect-non-trainer-error
"""

from __future__ import annotations

import argparse
import os
from typing import List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Asymmetric DDP 2-rank check")
    parser.add_argument("--trainer-rank", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--sync-interval", type=int, default=1)
    parser.add_argument(
        "--non-trainer-backward",
        choices=("allow", "warn", "error"),
        default="allow",
    )
    parser.add_argument(
        "--expect-non-trainer-error",
        action="store_true",
        help="Expect non-trainer backward to raise when policy=error.",
    )
    return parser.parse_args()


def assert_close(values: List[float], tol: float, name: str) -> None:
    vmax = max(values)
    vmin = min(values)
    if abs(vmax - vmin) > tol:
        raise RuntimeError(f"{name} mismatch across ranks: values={values}, tol={tol}")


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != 2:
        raise RuntimeError(f"Expected world_size=2, got {world_size}")
    if args.trainer_rank not in (0, 1):
        raise RuntimeError(f"trainer-rank must be 0/1, got {args.trainer_rank}")

    os.environ["TORCH_DDP_ASYMMETRIC_MODE"] = "1"
    os.environ["TORCH_DDP_TRAINER_RANK"] = str(args.trainer_rank)
    os.environ["TORCH_DDP_SYNC_INTERVAL"] = str(args.sync_interval)
    os.environ["TORCH_DDP_NON_TRAINER_BACKWARD"] = args.non_trainer_backward

    torch.manual_seed(2026 + rank)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )
    ddp = DDP(model)
    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.1)
    is_trainer = rank == args.trainer_rank

    if rank == 0:
        print("Asymmetric config rank0:", ddp.get_asymmetric_mode_config(), flush=True)

    dist.barrier()

    for step in range(args.steps):
        x = torch.randn(32, 8)
        y = torch.randn(32, 4)

        optimizer.zero_grad(set_to_none=True)
        out = ddp(x)
        loss = torch.nn.functional.mse_loss(out, y)

        non_trainer_got_error = False
        try:
            loss.backward()
        except RuntimeError:
            if (
                (not is_trainer)
                and args.non_trainer_backward == "error"
                and args.expect_non_trainer_error
            ):
                non_trainer_got_error = True
            else:
                raise

        if (
            (not is_trainer)
            and args.non_trainer_backward == "error"
            and args.expect_non_trainer_error
            and (not non_trainer_got_error)
        ):
            raise RuntimeError(
                "Expected non-trainer backward error, but backward succeeded."
            )

        ddp.trainer_step(optimizer)

        # After trainer_step, all ranks should have identical parameters.
        param_sum = sum(float(p.detach().float().sum().item()) for p in ddp.parameters())
        gathered_param_sum = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_param_sum, param_sum)
        assert_close(gathered_param_sum, tol=1e-5, name=f"param_sum_step{step}")

        # Non-trainer grads should be cleared by trainer_step according to the patch.
        grad_count = sum(0 if p.grad is None else 1 for p in ddp.parameters())
        gathered_grad_count = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_grad_count, grad_count)
        if gathered_grad_count[1 - args.trainer_rank] != 0:
            raise RuntimeError(
                "Expected non-trainer grad_count=0 after trainer_step, "
                f"got {gathered_grad_count}"
            )

        if rank == 0:
            print(
                f"step={step} loss={loss.item():.6f} "
                f"param_sum={gathered_param_sum} grad_count={gathered_grad_count}",
                flush=True,
            )

        dist.barrier()

    if rank == 0:
        print("2-rank asymmetric DDP check: PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

