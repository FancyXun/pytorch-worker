"""
Stage2 driver: does not modify stage1; uses ``stage2.worker_ops`` RPC targets.
"""
from __future__ import annotations

import copy
import inspect
import os
from typing import Any

import cloudpickle
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch.distributed.rpc import TensorPipeRpcBackendOptions

import stage2.worker_ops as worker_ops


class Stage2Trainer:
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
            f"[stage2 driver] rendezvous MASTER_ADDR={self._master_addr!r} MASTER_PORT={self._master_port}",
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
        *,
        use_amp: bool = False,
        allow_cpu_worker: bool = False,
        loss_module: nn.Module | None = None,
        resume_from: str | os.PathLike[str] | None = None,
        resume_strict: bool = True,
    ) -> None:
        if not self._started:
            self.start_rpc()
        spec: dict[str, Any] = {
            "model_blob": cloudpickle.dumps(copy.deepcopy(model).cpu()),
            "optim_pack": worker_ops.pack_optimizer(optimizer),
            "loss_kind": loss,
            "use_amp": use_amp,
            "allow_cpu_worker": allow_cpu_worker,
        }
        if loss_module is not None:
            spec["loss_blob"] = cloudpickle.dumps(loss_module)
        ret = rpc.rpc_sync(self._worker_name, worker_ops.setup_training_spec, args=(spec,))
        if ret != "ok":
            raise RuntimeError(f"setup_training_spec failed: {ret!r}")
        if resume_from is not None:
            self.resume_worker_from_file(resume_from, strict=resume_strict)

    def step(
        self,
        *tensor_args: torch.Tensor,
        forward_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> float:
        if not self._started:
            raise RuntimeError("call start_rpc() and attach() before step()")
        if forward_kwargs is None:
            cpu_args = tuple(t.detach().cpu() for t in tensor_args)
            return rpc.rpc_sync(self._worker_name, worker_ops.train_step, args=cpu_args)
        if len(tensor_args) < 2:
            raise ValueError("step(..., forward_kwargs=) needs *inputs, target")
        *inputs, target = tensor_args
        spec = {
            "forward_args": [t.detach().cpu() for t in inputs],
            "forward_kwargs": {k: v.detach().cpu() for k, v in forward_kwargs.items()},
            "target": target.detach().cpu(),
        }
        return rpc.rpc_sync(self._worker_name, worker_ops.train_step_ex, args=(spec,))

    def infer(
        self,
        *tensor_args: torch.Tensor,
        forward_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not self._started:
            raise RuntimeError("call start_rpc() and attach() before infer()")
        if forward_kwargs is None:
            cpu_args = tuple(t.detach().cpu() for t in tensor_args)
            return rpc.rpc_sync(self._worker_name, worker_ops.infer_step, args=cpu_args)
        spec = {
            "forward_args": [t.detach().cpu() for t in tensor_args],
            "forward_kwargs": {k: v.detach().cpu() for k, v in forward_kwargs.items()},
        }
        return rpc.rpc_sync(self._worker_name, worker_ops.infer_step_ex, args=(spec,))

    def fetch_model_state_dict(self) -> dict[str, torch.Tensor]:
        if not self._started:
            raise RuntimeError("call start_rpc() and attach() first")
        return rpc.rpc_sync(self._worker_name, worker_ops.get_model_state_dict, args=())

    def fetch_optimizer_state_dict(self) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("call start_rpc() and attach() first")
        return rpc.rpc_sync(self._worker_name, worker_ops.get_optimizer_state_dict, args=())

    def fetch_checkpoint(self) -> dict[str, Any]:
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
        if include_optimizer:
            ckpt = self.fetch_checkpoint()
        else:
            ckpt = {"model": self.fetch_model_state_dict()}
        if extra:
            ckpt = {**ckpt, **extra}
        torch.save(ckpt, path)

    def sync_local_model(self, model: nn.Module, *, strict: bool = True) -> nn.Module:
        model.load_state_dict(self.fetch_model_state_dict(), strict=strict)
        return model

    def load_checkpoint_to_worker(self, ckpt: dict[str, Any], *, strict: bool = True) -> None:
        if not self._started:
            raise RuntimeError("call start_rpc() and attach() first")
        ret = rpc.rpc_sync(
            self._worker_name,
            worker_ops.load_training_checkpoint,
            args=(ckpt, strict),
        )
        if ret != "ok":
            raise RuntimeError(f"load_training_checkpoint failed: {ret!r}")

    def resume_worker_from_file(
        self,
        path: str | os.PathLike[str],
        *,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> None:
        load_kw: dict[str, Any] = {"map_location": map_location}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kw["weights_only"] = False
        ckpt = torch.load(path, **load_kw)
        if not isinstance(ckpt, dict) or "model" not in ckpt or "optimizer" not in ckpt:
            raise ValueError("checkpoint must contain 'model' and 'optimizer'")
        self.load_checkpoint_to_worker(ckpt, strict=strict)

    def shutdown(self) -> None:
        if self._started:
            rpc.shutdown()
            self._started = False

    def __enter__(self) -> Stage2Trainer:
        self.start_rpc()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()
