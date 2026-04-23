from __future__ import annotations

import os
import time
import weakref
from datetime import timedelta
from typing import Iterable, List, Optional
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

_ENABLED = False
_DEBUG = False
_RANK = -1
_TRAINER_RANK = 0
_SYNC_INTERVAL = 1
_AUTO_SKIP_FOLLOWER_FORWARD = False

_MODULES: "weakref.WeakSet[torch.nn.Module]" = weakref.WeakSet()
_MODULE_DDP: "weakref.WeakKeyDictionary[torch.nn.Module, DDP]" = weakref.WeakKeyDictionary()
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
_IN_REDIRECTED_DDP_FORWARD = False


def _asymmetric_debug_enabled() -> bool:
    return os.environ.get("TORCH_DDP_ASYMMETRIC_DEBUG", "0") == "1"


def _debug_log(event: str, **fields) -> None:
    if not _DEBUG:
        return
    parts = [f"[auto-ddp-debug rank={_RANK}] {event}"]
    for k, v in fields.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}ms" if "ms" in k or k.endswith("_ms") else f"{k}={v:.6f}")
        else:
            parts.append(f"{k}={v}")
    print(" | ".join(parts), flush=True)


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
    os.environ.setdefault("TORCH_DDP_ASYMMETRIC_DEBUG", "0")
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
    timeout_sec = int(os.environ.get("TORCH_DDP_INIT_TIMEOUT_SEC", "90"))
    rank = int(os.environ.get("RANK", "0"))
    if _asymmetric_debug_enabled():
        print(
            f"[auto-ddp-debug rank={rank}] init_pg_start backend=gloo "
            f"master={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')} "
            f"world_size={os.environ.get('WORLD_SIZE')} timeout_sec={timeout_sec}",
            flush=True,
        )
    t0 = time.perf_counter()
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        timeout=timedelta(seconds=timeout_sec),
    )
    if _asymmetric_debug_enabled():
        _debug_log("init_process_group", wall_ms=(time.perf_counter() - t0) * 1000.0)


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
    wrapper = _MODULE_DDP.get(module, None)
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

    _MODULE_DDP[module] = ddp
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

    @contextmanager
    def _ddp_forward_guard():
        global _IN_REDIRECTED_DDP_FORWARD
        prev = _IN_REDIRECTED_DDP_FORWARD
        _IN_REDIRECTED_DDP_FORWARD = True
        try:
            yield
        finally:
            _IN_REDIRECTED_DDP_FORWARD = prev

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
                t0 = time.perf_counter()
                scalar = _LAST_DDP_FOR_LOSS.sync_scalar_from_trainer(None)
                _debug_log(
                    "loss_follower_sync_scalar",
                    wall_ms=(time.perf_counter() - t0) * 1000.0,
                )
                return torch.tensor(float(scalar), dtype=torch.float32, device="cpu")

            # Trainer path: compute local scalar then broadcast to followers.
            t_loss = time.perf_counter()
            out = _ORIG_MODULE_CALL(self, *args, **kwargs)
            if isinstance(out, torch.Tensor) and out.numel() == 1:
                _LAST_DDP_FOR_LOSS.sync_scalar_from_trainer(float(out.detach().item()))
            _debug_log(
                "loss_trainer_forward_plus_scalar_broadcast",
                wall_ms=(time.perf_counter() - t_loss) * 1000.0,
            )
            return out

        ddp = _MODULE_DDP.get(self, None)
        if ddp is not None:
            # Prevent recursive bounce: DDP.forward internally calls self.module(...).
            if _IN_REDIRECTED_DDP_FORWARD:
                return _ORIG_MODULE_CALL(self, *args, **kwargs)
            _LAST_DDP_FOR_LOSS = ddp
            if _AUTO_SKIP_FOLLOWER_FORWARD and _RANK != _TRAINER_RANK and _is_dist_enabled():
                _debug_log("module_forward_skipped_follower", wall_ms=0.0)
                return _SkippedForwardToken(ddp)
            with _ddp_forward_guard():
                _first_param = next(ddp.parameters(), None)
                use_cuda = (
                    _RANK == _TRAINER_RANK
                    and torch.cuda.is_available()
                    and _first_param is not None
                    and _first_param.is_cuda
                )
                t_wall = time.perf_counter()
                gpu_ms = None
                if use_cuda:
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    ev0.record()
                out = ddp(*args, **kwargs)
                if use_cuda:
                    ev1.record()
                    torch.cuda.synchronize()
                    gpu_ms = float(ev0.elapsed_time(ev1))
                wall_ms = (time.perf_counter() - t_wall) * 1000.0
                role = "trainer" if _RANK == _TRAINER_RANK else "follower"
                if use_cuda:
                    _debug_log(
                        f"module_forward_{role}",
                        wall_ms=wall_ms,
                        gpu_ms=gpu_ms if gpu_ms is not None else -1.0,
                    )
                else:
                    _debug_log(f"module_forward_{role}", wall_ms=wall_ms)
                return out
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
        t_step = time.perf_counter()
        result = _ORIG_OPTIMIZER_STEP(self, *args, **kwargs) if is_trainer else None
        if is_trainer:
            _debug_log(
                "optimizer_step_local_trainer",
                wall_ms=(time.perf_counter() - t_step) * 1000.0,
            )
        else:
            _debug_log("optimizer_step_skipped_follower", wall_ms=0.0)

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
    global _ENABLED, _DEBUG, _RANK, _TRAINER_RANK, _SYNC_INTERVAL, _AUTO_SKIP_FOLLOWER_FORWARD
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
    _DEBUG = _asymmetric_debug_enabled()

    _init_pg_if_needed()
    _patch_module_init()
    _patch_module_call()
    _patch_optimizer_init()
    _patch_optimizer_step()
    _patch_backward_for_follower()
    _ENABLED = True

