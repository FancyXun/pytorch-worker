#!/usr/bin/env python3
"""
Two-argument forward model(x, adj), last arg to step() is still the target y.

  python -m stage1.run_worker   # first  python -m stage1.example_two_input
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stage1.driver_api import Stage1Trainer


class ToyTwoInput(nn.Module):
    def __init__(self, n: int = 8, f: int = 4, out: int = 2):
        super().__init__()
        self.lin = nn.Linear(f, out)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F), adj: (N, N) — trivial mix for demo
        b, n, f = x.shape
        agg = torch.einsum("ij,bjf->bif", adj, x)
        return self.lin(agg.mean(dim=1))


def main() -> None:
    torch.manual_seed(0)
    bsz, n_node, f, out = 16, 8, 4, 2
    adj = torch.softmax(torch.randn(n_node, n_node), dim=-1)
    x = torch.randn(128, n_node, f)
    y = torch.randn(128, out)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=bsz, shuffle=True)

    model = ToyTwoInput(n_node, f, out)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    trainer = Stage1Trainer(
        master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        master_port=int(os.environ.get("MASTER_PORT", "29500")),
    )
    trainer.start_rpc()
    trainer.attach(model, optimizer, loss="mse")

    adj_cpu = adj.cpu()
    try:
        for epoch in range(2):
            losses: list[float] = []
            for xb, yb in loader:
                loss_val = trainer.step(xb, adj_cpu, yb)
                losses.append(loss_val)
            print(f"epoch {epoch + 1} mean_loss={sum(losses) / len(losses):.6f}", flush=True)
    finally:
        trainer.shutdown()


if __name__ == "__main__":
    main()
