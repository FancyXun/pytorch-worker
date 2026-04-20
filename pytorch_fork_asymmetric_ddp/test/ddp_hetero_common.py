#!/usr/bin/env python3
"""Common runtime for asymmetric DDP heterogeneous demo (GPU trainer + CPU follower)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
    batch_size: int,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    log_interval: int,
    save_every_steps: int,
    save_dir: str,
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
        torch.nn.Linear(in_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, out_dim),
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

    start = time.time()
    ckpt_dir = Path(save_dir) if save_every_steps > 0 else None
    if ckpt_dir is not None and (not is_trainer):
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    while step < steps:
        x = torch.randn(batch_size, in_dim, device=device)
        y = torch.randn(batch_size, out_dim, device=device)

        if is_trainer:
            optimizer.zero_grad(set_to_none=True)

        out = ddp(x)
        loss = torch.nn.functional.mse_loss(out, y)

        # Key behavior: only trainer does backward/optimizer.step().
        if is_trainer:
            loss.backward()

        ddp.trainer_step(optimizer if is_trainer else None)

        if rank == trainer_rank and (step % max(1, log_interval) == 0):
            elapsed = time.time() - start
            print(
                f"step={step} trainer_loss={float(loss.item()):.6f} elapsed={elapsed:.1f}s",
                flush=True,
            )
        elif (not is_trainer) and (step % max(1, log_interval) == 0):
            elapsed = time.time() - start
            print(
                f"step={step} follower_loss={float(loss.item()):.6f} "
                f"follower_sync_ok elapsed={elapsed:.1f}s",
                flush=True,
            )

        if (not is_trainer) and ckpt_dir is not None and ((step + 1) % save_every_steps == 0):
            ckpt_path = ckpt_dir / f"follower_step_{step + 1}.pt"
            torch.save(
                {
                    "model": ddp.module.state_dict(),
                    "step": step + 1,
                    "saved_at": time.time(),
                },
                ckpt_path,
            )
            print(f"[rank{rank}] checkpoint_saved={ckpt_path}", flush=True)

        dist.barrier()
        step += 1

    if rank == trainer_rank:
        print(
            f"Hetero asymmetric DDP demo: PASS "
            f"(steps={step}, elapsed={time.time() - start:.1f}s)",
            flush=True,
        )
    dist.destroy_process_group()

