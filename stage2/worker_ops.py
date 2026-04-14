"""
Stage2 worker RPC ops — standalone from stage1.

Adds: optional CPU worker, CUDA AMP, forward **kwargs, custom loss module blob,
checkpoint restore on worker.
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
_scaler: torch.cuda.amp.GradScaler | None = None
_use_amp: bool = False


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
    if n in ("mse", "mseloss"):
        return nn.MSELoss()
    if n in ("l1", "mae", "l1loss"):
        return nn.L1Loss()
    if n in ("ce", "cross_entropy", "crossentropy"):
        return nn.CrossEntropyLoss()
    if n in ("smooth_l1", "huber"):
        return nn.SmoothL1Loss()
    raise ValueError(f"unknown loss kind: {name!r}")


def setup_training_spec(spec: dict[str, Any]) -> str:
    """
    spec keys:
      - model_blob: bytes (cloudpickle nn.Module), required
      - optim_pack: dict from pack_optimizer, required
      - loss_kind: str if loss_blob absent
      - loss_blob: optional bytes, cloudpickle nn.Module (custom loss)
      - use_amp: bool, CUDA autocast + GradScaler (CUDA only)
      - allow_cpu_worker: bool, do not require CUDA
    """
    global _model, _optimizer, _loss_fn, _device, _scaler, _use_amp

    allow_cpu = bool(spec.get("allow_cpu_worker", False))
    _use_amp = bool(spec.get("use_amp", False)) and torch.cuda.is_available()

    if not torch.cuda.is_available():
        if not allow_cpu:
            raise RuntimeError("CUDA not available; set allow_cpu_worker=True for CPU worker (slow).")
        _device = torch.device("cpu")
        _use_amp = False
    else:
        _device = torch.device("cuda")

    model_blob = spec["model_blob"]
    optim_pack = spec["optim_pack"]
    m = cloudpickle.loads(model_blob)
    if not isinstance(m, nn.Module):
        raise TypeError("model_blob must deserialize to nn.Module")
    _model = m.to(_device)
    _optimizer = rebuild_optimizer(_model, optim_pack, _device)

    if spec.get("loss_blob") is not None:
        lf = cloudpickle.loads(spec["loss_blob"])
        if not isinstance(lf, nn.Module):
            raise TypeError("loss_blob must deserialize to nn.Module")
        _loss_fn = lf.to(_device)
    else:
        _loss_fn = _make_loss(str(spec.get("loss_kind", "mse")))

    _scaler = torch.cuda.amp.GradScaler() if _use_amp else None
    return "ok"


def _step_core(
    inputs: list[torch.Tensor],
    forward_kw: dict[str, torch.Tensor],
    target: torch.Tensor,
) -> float:
    if _model is None or _optimizer is None or _loss_fn is None:
        raise RuntimeError("setup_training_spec was not called")

    tensors_in = [t.to(_device, non_blocking=True) for t in inputs]
    kw = {k: v.to(_device, non_blocking=True) for k, v in forward_kw.items()}
    target_d = target.to(_device, non_blocking=True)

    _model.train()
    _optimizer.zero_grad(set_to_none=True)

    if _use_amp and _scaler is not None:
        with torch.cuda.amp.autocast():
            pred = _model(*tensors_in, **kw)
            loss = _loss_fn(pred, target_d)
        _scaler.scale(loss).backward()
        _scaler.step(_optimizer)
        _scaler.update()
    else:
        pred = _model(*tensors_in, **kw)
        loss = _loss_fn(pred, target_d)
        loss.backward()
        _optimizer.step()

    return float(loss.detach().cpu())


def train_step(*tensor_args: torch.Tensor) -> float:
    """Positional forward only: model(*inputs), last arg is target."""
    if len(tensor_args) < 2:
        raise ValueError("need input tensors + target")
    *inputs, target = tensor_args
    if not inputs:
        raise ValueError("need at least one forward input")
    return _step_core(list(inputs), {}, target)


def train_step_ex(spec: dict[str, Any]) -> float:
    """
    spec:
      - forward_args: list[Tensor]
      - forward_kwargs: dict[str, Tensor] (optional)
      - target: Tensor
    """
    inputs = spec["forward_args"]
    kw = spec.get("forward_kwargs") or {}
    target = spec["target"]
    if not isinstance(inputs, list):
        raise TypeError("forward_args must be a list")
    if not isinstance(kw, dict):
        raise TypeError("forward_kwargs must be a dict")
    return _step_core(inputs, kw, target)


def infer_step(*tensor_args: torch.Tensor) -> torch.Tensor:
    if _model is None:
        raise RuntimeError("setup_training_spec was not called")
    _model.eval()
    with torch.no_grad():
        tensors_in = [t.to(_device, non_blocking=True) for t in tensor_args]
        out = _model(*tensors_in)
    return out.detach().cpu()


def infer_step_ex(spec: dict[str, Any]) -> torch.Tensor:
    inputs = spec["forward_args"]
    kw = spec.get("forward_kwargs") or {}
    if _model is None:
        raise RuntimeError("setup_training_spec was not called")
    _model.eval()
    with torch.no_grad():
        tensors_in = [t.to(_device, non_blocking=True) for t in inputs]
        kw_d = {k: v.to(_device, non_blocking=True) for k, v in kw.items()}
        out = _model(*tensors_in, **kw_d)
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


def _to_device_structure(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device_structure(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device_structure(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device_structure(v, device) for v in obj)
    return obj


def get_model_state_dict() -> dict[str, torch.Tensor]:
    if _model is None:
        raise RuntimeError("setup_training_spec was not called")
    return {k: v.detach().cpu() for k, v in _model.state_dict().items()}


def get_optimizer_state_dict() -> dict[str, Any]:
    if _optimizer is None:
        raise RuntimeError("setup_training_spec was not called")
    return _to_cpu_structure(_optimizer.state_dict())


def get_training_checkpoint() -> dict[str, Any]:
    return {
        "model": get_model_state_dict(),
        "optimizer": get_optimizer_state_dict(),
    }


def load_training_checkpoint(ckpt: dict[str, Any], strict: bool = True) -> str:
    global _model, _optimizer
    if _model is None or _optimizer is None:
        raise RuntimeError("setup_training_spec was not called")
    if "model" not in ckpt or "optimizer" not in ckpt:
        raise ValueError("ckpt must contain model and optimizer")
    sd_m = _to_device_structure(ckpt["model"], _device)
    _model.load_state_dict(sd_m, strict=strict)
    sd_o = _to_device_structure(ckpt["optimizer"], _device)
    _optimizer.load_state_dict(sd_o)
    return "ok"
