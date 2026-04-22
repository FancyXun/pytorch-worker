from __future__ import annotations

import os
import weakref
from typing import Iterable, List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

_ENABLED = False
_RANK = -1
_TRAINER_RANK = 0
_SYNC_INTERVAL = 1
_AUTO_SKIP_FOLLOWER_FORWARD = False

_MODULES: "weakref.WeakSet[torch.nn.Module]" = weakref.WeakSet()
_OPTIMIZER_DDP: "weakref.WeakKeyDictionary[torch.optim.Optimizer, DDP]" = (
    weakref.WeakKeyDictionary()
)
_OPTIMIZER_STEP_COUNT: "weakref.WeakKeyDictionary[torch.optim.Optimizer, int]" = (
    weakref.WeakKeyDictionary()
)

_ORIG_MODULE_INIT = None
_ORIG_MODULE_CALL = None
_ORIG_OPTIMIZER_INIT = None
_ORIG_OPTIMIZER_STEP = None
_ORIG_TENSOR_BACKWARD = None
_ORIG_AUTO_BACKWARD = None
_LAST_DDP_FOR_LOSS: Optional[DDP] = None


class _SkippedForwardToken:
    __slots__ = ("ddp",)

    def __init__(self, ddp: DDP):
        self.ddp = ddp


def _set_default_env() -> None:
    os.environ.setdefault("TORCH_DDP_ASYMMETRIC_MODE", "1")
    os.environ.setdefault("TORCH_DDP_TRAINER_RANK", "0")
    os.environ.setdefault("TORCH_DDP_SKIP_ALLREDUCE", "1")
    os.environ.setdefault("TORCH_DDP_HETERO_PARAM_SYNC", "1")
    os.environ.setdefault("TORCH_DDP_NON_TRAINER_FORWARD_ONLY", "1")
    os.environ.setdefault("TORCH_DDP_NON_TRAINER_BACKWARD", "error")
    os.environ.setdefault("TORCH_DDP_SYNC_INTERVAL", "1")
    os.environ.setdefault("TORCH_DDP_AUTO_SKIP_FOLLOWER_FORWARD", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")


def _is_dist_enabled() -> bool:
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except Exception:
        return False


def _init_pg_if_needed() -> None:
    if not _is_dist_enabled():
        return
    if dist.is_initialized():
        return
    if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
        raise RuntimeError(
            "TORCH_DDP_AUTO_WRAP=1 requires MASTER_ADDR/MASTER_PORT when WORLD_SIZE>1"
        )
    dist.init_process_group(backend="gloo", init_method="env://")


def _flatten_params(params: Iterable) -> List[torch.nn.Parameter]:
    out: List[torch.nn.Parameter] = []
    for obj in params:
        if isinstance(obj, torch.nn.Parameter):
            out.append(obj)
        elif isinstance(obj, dict):
            for p in obj.get("params", []):
                if isinstance(p, torch.nn.Parameter):
                    out.append(p)
    return out


def _find_owner_module(params: List[torch.nn.Parameter]) -> Optional[torch.nn.Module]:
    if not params:
        return None
    wanted = {id(p) for p in params}
    best = None
    best_size = -1
    for m in list(_MODULES):
        try:
            all_params = list(m.parameters())
        except Exception:
            continue
        if not all_params:
            continue
        all_ids = {id(p) for p in all_params}
        if wanted.issubset(all_ids):
            if len(all_params) > best_size:
                best = m
                best_size = len(all_params)
    return best


def _wrap_module_if_needed(module: torch.nn.Module) -> DDP:
    wrapper = getattr(module, "_asym_ddp_wrapper", None)
    if wrapper is not None:
        return wrapper

    if _RANK == _TRAINER_RANK:
        if not torch.cuda.is_available():
            raise RuntimeError("trainer rank requires CUDA for auto-wrap mode")
        if next(module.parameters(), None) is not None:
            module = module.to(torch.device("cuda:0"))
        ddp = DDP(module, device_ids=[0], output_device=0, broadcast_buffers=False)
    else:
        module = module.to(torch.device("cpu"))
        ddp = DDP(module, broadcast_buffers=False)

    setattr(module, "_asym_ddp_wrapper", ddp)
    return ddp


def _patch_module_init() -> None:
    global _ORIG_MODULE_INIT
    _ORIG_MODULE_INIT = torch.nn.Module.__init__

    def _wrapped_init(self, *args, **kwargs):
        _ORIG_MODULE_INIT(self, *args, **kwargs)
        _MODULES.add(self)

    torch.nn.Module.__init__ = _wrapped_init


def _patch_module_call() -> None:
    global _ORIG_MODULE_CALL
    _ORIG_MODULE_CALL = torch.nn.Module.__call__

    def _wrapped_call(self, *args, **kwargs):
        global _LAST_DDP_FOR_LOSS
        if (
            _is_dist_enabled()
            and isinstance(self, torch.nn.modules.loss._Loss)
            and _LAST_DDP_FOR_LOSS is not None
        ):
            # Follower path: loss receives a skipped-forward token and gets
            # trainer scalar via broadcast.
            if _AUTO_SKIP_FOLLOWER_FORWARD and args and isinstance(args[0], _SkippedForwardToken):
                scalar = _LAST_DDP_FOR_LOSS.sync_scalar_from_trainer(None)
                return torch.tensor(float(scalar), dtype=torch.float32, device="cpu")

            # Trainer path: compute local scalar then broadcast to followers.
            out = _ORIG_MODULE_CALL(self, *args, **kwargs)
            if isinstance(out, torch.Tensor) and out.numel() == 1:
                _LAST_DDP_FOR_LOSS.sync_scalar_from_trainer(float(out.detach().item()))
            return out

        ddp = getattr(self, "_asym_ddp_wrapper", None)
        if ddp is not None:
            _LAST_DDP_FOR_LOSS = ddp
            if _AUTO_SKIP_FOLLOWER_FORWARD and _RANK != _TRAINER_RANK and _is_dist_enabled():
                return _SkippedForwardToken(ddp)
            return ddp(*args, **kwargs)
        return _ORIG_MODULE_CALL(self, *args, **kwargs)

    torch.nn.Module.__call__ = _wrapped_call


def _patch_optimizer_init() -> None:
    global _ORIG_OPTIMIZER_INIT
    _ORIG_OPTIMIZER_INIT = torch.optim.Optimizer.__init__

    def _wrapped_opt_init(self, params, defaults):
        params_list = list(params)
        _ORIG_OPTIMIZER_INIT(self, params_list, defaults)
        flat = _flatten_params(params_list)
        owner = _find_owner_module(flat)
        if owner is None:
            return
        ddp = _wrap_module_if_needed(owner)
        _OPTIMIZER_DDP[self] = ddp
        _OPTIMIZER_STEP_COUNT[self] = 0

    torch.optim.Optimizer.__init__ = _wrapped_opt_init


def _patch_optimizer_step() -> None:
    global _ORIG_OPTIMIZER_STEP
    _ORIG_OPTIMIZER_STEP = torch.optim.Optimizer.step

    def _wrapped_step(self, *args, **kwargs):
        ddp = _OPTIMIZER_DDP.get(self, None)
        if ddp is None:
            return _ORIG_OPTIMIZER_STEP(self, *args, **kwargs)

        is_trainer = bool(ddp.is_trainer_rank())
        result = _ORIG_OPTIMIZER_STEP(self, *args, **kwargs) if is_trainer else None

        step = _OPTIMIZER_STEP_COUNT.get(self, 0) + 1
        _OPTIMIZER_STEP_COUNT[self] = step
        if step % _SYNC_INTERVAL == 0:
            ddp.sync_params_from_trainer()
        return result

    torch.optim.Optimizer.step = _wrapped_step


def _patch_backward_for_follower() -> None:
    global _ORIG_TENSOR_BACKWARD, _ORIG_AUTO_BACKWARD
    _ORIG_TENSOR_BACKWARD = torch.Tensor.backward
    _ORIG_AUTO_BACKWARD = torch.autograd.backward

    def _wrapped_tensor_backward(self, *args, **kwargs):
        if _RANK != _TRAINER_RANK and _is_dist_enabled():
            return None
        return _ORIG_TENSOR_BACKWARD(self, *args, **kwargs)

    def _wrapped_autograd_backward(*args, **kwargs):
        if _RANK != _TRAINER_RANK and _is_dist_enabled():
            return None
        return _ORIG_AUTO_BACKWARD(*args, **kwargs)

    torch.Tensor.backward = _wrapped_tensor_backward
    torch.autograd.backward = _wrapped_autograd_backward


def enable_from_env() -> None:
    global _ENABLED, _RANK, _TRAINER_RANK, _SYNC_INTERVAL, _AUTO_SKIP_FOLLOWER_FORWARD
    if _ENABLED:
        return
    if os.environ.get("TORCH_DDP_AUTO_WRAP", "0") != "1":
        return

    _set_default_env()
    _RANK = int(os.environ.get("RANK", "0"))
    _TRAINER_RANK = int(os.environ.get("TORCH_DDP_TRAINER_RANK", "0"))
    _SYNC_INTERVAL = max(1, int(os.environ.get("TORCH_DDP_SYNC_INTERVAL", "1")))
    _AUTO_SKIP_FOLLOWER_FORWARD = (
        os.environ.get("TORCH_DDP_AUTO_SKIP_FOLLOWER_FORWARD", "0") == "1"
    )

    _init_pg_if_needed()
    _patch_module_init()
    _patch_module_call()
    _patch_optimizer_init()
    _patch_optimizer_step()
    _patch_backward_for_follower()
    _ENABLED = True

