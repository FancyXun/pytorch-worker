"""
Runs on the GPU worker process only. Exposed to the driver via torch.distributed.rpc.
Training step: forward + loss + backward + optimizer.step (closed on device).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

try:
    import cloudpickle
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install cloudpickle") from e

_model: nn.Module | None = None
_optimizer: torch.optim.Optimizer | None = None
_loss_fn: nn.Module | None = None
_device: torch.device = torch.device("cpu")


def _move_group_tensors(meta: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if k == "num_params":
            out[k] = v
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def rebuild_optimizer(
    model: nn.Module,
    pack: dict[str, Any],
    device: torch.device,
) -> torch.optim.Optimizer:
    opt_cls = cloudpickle.loads(pack["opt_class_blob"])
    all_params = list(model.parameters())
    idx = 0
    new_groups: list[dict[str, Any]] = []
    for g in pack["param_groups"]:
        n = int(g["num_params"])
        if idx + n > len(all_params):
            raise ValueError("optimizer param_groups do not match model.parameters() count")
        meta = _move_group_tensors(dict(g), device)
        meta.pop("num_params", None)
        meta["params"] = all_params[idx : idx + n]
        new_groups.append(meta)
        idx += n
    if idx != len(all_params):
        raise ValueError("optimizer param_groups cover fewer parameters than the model has")

    optim: torch.optim.Optimizer = opt_cls(new_groups)
    optim.load_state_dict(pack["state_dict"])
    return optim


def pack_optimizer(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Call on driver before RPC setup; same pack format worker expects."""

    def _serialize_group(group: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in group.items():
            if k == "params":
                continue
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().cpu()
            else:
                out[k] = v
        out["num_params"] = len(group["params"])
        return out

    return {
        "opt_class_blob": cloudpickle.dumps(optimizer.__class__),
        "param_groups": [_serialize_group(g) for g in optimizer.param_groups],
        "state_dict": optimizer.state_dict(),
    }


def _make_loss(name: str) -> nn.Module:
    n = name.lower().strip()
    if n in ("mse", "mSELoss", "nn.mseloss"):
        return nn.MSELoss()
    if n in ("l1", "mae", "l1loss"):
        return nn.L1Loss()
    if n in ("ce", "cross_entropy", "crossentropy"):
        return nn.CrossEntropyLoss()
    raise ValueError(f"unknown loss kind: {name!r} (use mse, l1, cross_entropy)")


def setup_training(model_blob: bytes, optim_pack: dict[str, Any], loss_kind: str) -> str:
    """
    Load cloudpickled model (CPU state in blob), move to CUDA, rebuild optimizer, set loss.
    """
    global _model, _optimizer, _loss_fn, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _device.type != "cuda":
        raise RuntimeError("GPU worker expected CUDA; set CUDA_VISIBLE_DEVICES and use a GPU host.")

    m = cloudpickle.loads(model_blob)
    if not isinstance(m, nn.Module):
        raise TypeError("model_blob did not deserialize to nn.Module")
    _model = m.to(_device)
    _optimizer = rebuild_optimizer(_model, optim_pack, _device)
    _loss_fn = _make_loss(loss_kind)
    return "ok"


def train_step(*tensor_args: torch.Tensor) -> float:
    """
    One training step. All leading tensors are forwarded to model(*inputs); the last tensor is the target for loss.

    Examples:
      model(x) -> train_step(x, y)
      model(x, adj)      -> train_step(x, adj, y)
    """
    if _model is None or _optimizer is None or _loss_fn is None:
        raise RuntimeError("setup_training was not called on this worker")

    if len(tensor_args) < 2:
        raise ValueError("train_step needs at least (input..., target); got fewer than 2 tensors")

    *inputs, target = tensor_args
    if not inputs:
        raise ValueError("need at least one model input tensor before target")

    tensors_in = [t.to(_device, non_blocking=True) for t in inputs]
    target_d = target.to(_device, non_blocking=True)

    _model.train()
    _optimizer.zero_grad(set_to_none=True)
    pred = _model(*tensors_in)
    loss = _loss_fn(pred, target_d)
    loss.backward()
    _optimizer.step()
    return float(loss.detach().cpu())


def infer_step(*tensor_args: torch.Tensor) -> torch.Tensor:
    """Optional: forward only (no grad). Same *args convention as train_step but no trailing target required."""
    if _model is None:
        raise RuntimeError("setup_training was not called on this worker")
    if not tensor_args:
        raise ValueError("infer_step needs at least one input tensor")
    _model.eval()
    with torch.no_grad():
        tensors_in = [t.to(_device, non_blocking=True) for t in tensor_args]
        out = _model(*tensors_in)
    return out.detach().cpu()


def _to_cpu_structure(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_structure(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_structure(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_structure(v) for v in obj)
    return obj


def get_model_state_dict() -> dict[str, torch.Tensor]:
    """Return model weights on CPU for checkpointing."""
    if _model is None:
        raise RuntimeError("setup_training was not called on this worker")
    return {k: v.detach().cpu() for k, v in _model.state_dict().items()}


def get_optimizer_state_dict() -> dict[str, Any]:
    """Return optimizer state on CPU (Adam moments, etc.)."""
    if _optimizer is None:
        raise RuntimeError("setup_training was not called on this worker")
    return _to_cpu_structure(_optimizer.state_dict())


def get_training_checkpoint() -> dict[str, Any]:
    """Single RPC: model + optimizer state_dicts, all tensors on CPU."""
    return {
        "model": get_model_state_dict(),
        "optimizer": get_optimizer_state_dict(),
    }
