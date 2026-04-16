#!/usr/bin/env python3
"""Common runtime for asymmetric DDP heterogeneous demo (GPU trainer + CPU follower)."""

from __future__ import annotations

import os
from typing import List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def _gather_scalar_float(value: float, world_size: int) -> List[float]:
    local = torch.tensor([value], dtype=torch.float64)
    gathered = [torch.zeros(1, dtype=torch.float64) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return [float(t.item()) for t in gathered]


def _gather_scalar_int(value: int, world_size: int) -> List[int]:
    local = torch.tensor([value], dtype=torch.int64)
    gathered = [torch.zeros(1, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return [int(t.item()) for t in gathered]


def _configure_asymmetric_env(trainer_rank: int) -> None:
    os.environ["TORCH_DDP_ASYMMETRIC_MODE"] = "1"
    os.environ["TORCH_DDP_TRAINER_RANK"] = str(trainer_rank)
    os.environ["TORCH_DDP_SKIP_ALLREDUCE"] = "1"
    os.environ["TORCH_DDP_NON_TRAINER_FORWARD_ONLY"] = "1"
    os.environ["TORCH_DDP_NON_TRAINER_BACKWARD"] = "allow"
    os.environ["TORCH_DDP_SYNC_INTERVAL"] = "1"
    os.environ["TORCH_DDP_HETERO_PARAM_SYNC"] = "1"


def run_hetero_role(
    *,
    rank: int,
    world_size: int,
    trainer_rank: int,
    master_addr: str,
    master_port: int,
    steps: int,
) -> None:
    _configure_asymmetric_env(trainer_rank)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="gloo", init_method="env://")
    is_trainer = rank == trainer_rank

    if is_trainer:
        if not torch.cuda.is_available():
            raise RuntimeError("Trainer rank requires CUDA, but torch.cuda.is_available()=False")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Keep init deterministic so both ranks start from identical parameters.
    torch.manual_seed(2026)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    ).to(device)

    if device.type == "cuda":
        ddp = DDP(model, device_ids=[0], output_device=0)
    else:
        ddp = DDP(model)

    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.1) if is_trainer else None

    cfg = ddp.get_asymmetric_mode_config()
    print(
        f"[rank{rank}] device={device} is_trainer={is_trainer} cfg={cfg}",
        flush=True,
    )
    dist.barrier()

    for step in range(steps):
        x = torch.randn(32, 8, device=device)
        y = torch.randn(32, 4, device=device)

        if is_trainer:
            optimizer.zero_grad(set_to_none=True)

        out = ddp(x)
        loss = torch.nn.functional.mse_loss(out, y)

        # Key behavior: only trainer does backward/optimizer.step().
        if is_trainer:
            loss.backward()

        ddp.trainer_step(optimizer if is_trainer else None)

        param_sum = sum(float(p.detach().float().sum().item()) for p in ddp.parameters())
        gathered_param_sum = _gather_scalar_float(param_sum, world_size)
        grad_count = sum(0 if p.grad is None else 1 for p in ddp.parameters())
        gathered_grad_count = _gather_scalar_int(grad_count, world_size)

        if rank == trainer_rank:
            print(
                f"step={step} loss={float(loss.item()):.6f} "
                f"param_sum={gathered_param_sum} grad_count={gathered_grad_count}",
                flush=True,
            )

        dist.barrier()

    if rank == trainer_rank:
        print("Hetero asymmetric DDP demo: PASS", flush=True)
    dist.destroy_process_group()

