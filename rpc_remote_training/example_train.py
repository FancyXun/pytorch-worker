#!/usr/bin/env python3
"""
Minimal example: driver uses CPU tensors only; worker runs the full train step.

Prerequisite: start worker first on the GPU host (same MASTER_ADDR/MASTER_PORT):

  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT=29500
  python -m rpc_remote_training.run_worker

Then from repo root (can be a different machine if MASTER_ADDR points to worker host):

  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT=29500
  python -m rpc_remote_training.example_train
"""
from __future__ import annotations

import os
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rpc_remote_training.driver_api import Stage1Trainer


class TinyMLP(nn.Module):
    def __init__(self, d_in: int = 32, d_h: int = 64, d_out: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Linear(d_h, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    torch.manual_seed(0)
    device_cpu = torch.device("cpu")

    n, d_in, d_out = 256, 32, 4
    x = torch.randn(n, d_in)
    y = torch.randn(n, d_out)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = TinyMLP(d_in=d_in, d_out=d_out).to(device_cpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = Stage1Trainer(
        master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        master_port=int(os.environ.get("MASTER_PORT", "29500")),
    )
    trainer.start_rpc()
    trainer.attach(model, optimizer, loss="mse")

    try:
        for epoch in range(2):
            losses: list[float] = []
            for xb, yb in loader:
                loss_val = trainer.step(xb, yb)
                losses.append(loss_val)
            print(f"epoch {epoch + 1} mean_loss={sum(losses) / len(losses):.6f}", flush=True)
        fd, ckpt_path = tempfile.mkstemp(suffix="_rpc_remote_training_example.pt", prefix="torch_worker_")
        os.close(fd)
        try:
            trainer.save_checkpoint(ckpt_path, extra={"epochs_ran": 2})
            print(f"checkpoint saved to {ckpt_path}", flush=True)
        finally:
            os.remove(ckpt_path)
    finally:
        trainer.shutdown()


if __name__ == "__main__":
    main()
