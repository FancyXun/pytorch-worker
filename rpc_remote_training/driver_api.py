"""
CPU-side driver: data + loop + rpc.rpc_sync to worker (official torch.distributed.rpc).
"""
from __future__ import annotations

import copy
import os
from typing import Any

import cloudpickle
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch.distributed.rpc import TensorPipeRpcBackendOptions

import rpc_remote_training.worker_ops as worker_ops


class Stage1Trainer:
    """
    Product-shaped driver: attach(model, optimizer) once, then step(*tensors) per batch.

    Convention for step():
      step(x, y)                 if model(x)
      step(x, adj, y)            if model(x, adj)
 Last tensor is always the supervised target for loss; all preceding tensors are model.forward args.

    Requires: CPU-only PyTorch on driver is fine. Worker must have CUDA + matching major torch.
    """

    def __init__(
        self,
        *,
        worker_name: str = "gpu_worker",
        driver_name: str = "driver",
        master_addr: str | None = None,
        master_port: int | None = None,
        world_size: int = 2,
        driver_rank: int = 0,
        rpc_timeout: int = 600,
    ) -> None:
        self._worker_name = worker_name
        self._driver_name = driver_name
        self._world_size = world_size
        self._driver_rank = driver_rank
        self._master_addr = master_addr or os.environ.get("MASTER_ADDR", "127.0.0.1")
        self._master_port = int(master_port or os.environ.get("MASTER_PORT", "29500"))
        self._rpc_timeout = rpc_timeout
        self._started = False

    def start_rpc(self) -> None:
        if self._started:
            return
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)

        print(
            f"[rpc_remote_training driver] rendezvous MASTER_ADDR={self._master_addr!r} MASTER_PORT={self._master_port}",
            flush=True,
        )

        opts = TensorPipeRpcBackendOptions(
            num_worker_threads=8,
            rpc_timeout=self._rpc_timeout,
        )
        rpc.init_rpc(
            self._driver_name,
            rank=self._driver_rank,
            world_size=self._world_size,
            rpc_backend_options=opts,
        )
        self._started = True

    def attach(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: str = "mse",
    ) -> None:
        if not self._started:
            self.start_rpc()
        model_cpu = copy.deepcopy(model).cpu()
        blob = cloudpickle.dumps(model_cpu)
        pack = worker_ops.pack_optimizer(optimizer)
        ret = rpc.rpc_sync(
            self._worker_name,
            worker_ops.setup_training,
            args=(blob, pack, loss),
        )
        if ret != "ok":
            raise RuntimeError(f"worker setup_training failed: {ret!r}")

    def step(self, *tensor_args: torch.Tensor) -> float:
        if not self._started:
            raise RuntimeError("call start_rpc() and attach() before step()")
        cpu_args = tuple(t.detach().cpu() for t in tensor_args)
        return rpc.rpc_sync(self._worker_name, worker_ops.train_step, args=cpu_args)

    def infer(self, *tensor_args: torch.Tensor) -> torch.Tensor:
        """Forward-only on worker; returns CPU tensor."""
        if not self._started:
            raise RuntimeError("call start_rpc() and attach() before infer()")
        cpu_args = tuple(t.detach().cpu() for t in tensor_args)
        return rpc.rpc_sync(self._worker_name, worker_ops.infer_step, args=cpu_args)

    def fetch_model_state_dict(self) -> dict[str, torch.Tensor]:
        """Pull current model weights from the worker (CPU tensors)."""
        if not self._started:
            raise RuntimeError("call start_rpc() and attach() first")
        return rpc.rpc_sync(self._worker_name, worker_ops.get_model_state_dict, args=())

    def fetch_optimizer_state_dict(self) -> dict[str, Any]:
        """Pull optimizer state from the worker (e.g. for resume training)."""
        if not self._started:
            raise RuntimeError("call start_rpc() and attach() first")
        return rpc.rpc_sync(self._worker_name, worker_ops.get_optimizer_state_dict, args=())

    def fetch_checkpoint(self) -> dict[str, Any]:
        """One RPC: ``{"model": state_dict, "optimizer": state_dict}`` on CPU."""
        if not self._started:
            raise RuntimeError("call start_rpc() and attach() first")
        return rpc.rpc_sync(self._worker_name, worker_ops.get_training_checkpoint, args=())

    def save_checkpoint(
        self,
        path: str | os.PathLike[str],
        *,
        include_optimizer: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """
        Pull state from worker and ``torch.save`` on the driver.
        Default payload: ``{"model": ..., "optimizer": ...}`` plus optional ``extra``.
        """
        ckpt: dict[str, Any]
        if include_optimizer:
            ckpt = self.fetch_checkpoint()
        else:
            ckpt = {"model": self.fetch_model_state_dict()}
        if extra:
            ckpt = {**ckpt, **extra}
        torch.save(ckpt, path)

    def sync_local_model(self, model: nn.Module, *, strict: bool = True) -> nn.Module:
        """``load_state_dict`` on a driver-side module from the worker (e.g. for local export)."""
        model.load_state_dict(self.fetch_model_state_dict(), strict=strict)
        return model

    def shutdown(self) -> None:
        if self._started:
            rpc.shutdown()
            self._started = False

    def __enter__(self) -> Stage1Trainer:
        self.start_rpc()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()
