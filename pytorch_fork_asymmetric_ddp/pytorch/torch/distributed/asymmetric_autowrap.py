from __future__ import annotations

import os
import time
import types
import weakref
from datetime import timedelta
from typing import Iterable, List, Optional
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

_ENABLED = False
_DEBUG = False
_DEBUG_EVERY_N = 1
_DEBUG_EVENTS = set()
_SUMMARY_ENABLED = False
_SUMMARY_EVERY_STEPS = 0
_SUMMARY_STEPS_PER_EPOCH = 0
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
_ORIG_TENSOR_BACKWARD = None
_ORIG_AUTO_BACKWARD = None
_LAST_DDP_FOR_LOSS: Optional[DDP] = None
_IN_REDIRECTED_DDP_FORWARD = False
_DEBUG_COUNTERS = {}
_SUMMARY_ACC = {}
_SUMMARY_STEP_TOTAL = 0
_SUMMARY_LAST_FLUSH_STEP = 0
_SUMMARY_EPOCH_IDX = 0


def _asymmetric_debug_enabled() -> bool:
    return os.environ.get("TORCH_DDP_ASYMMETRIC_DEBUG", "0") == "1"


def _debug_log(event: str, **fields) -> None:
    if not _DEBUG:
        return
    if _DEBUG_EVENTS and event not in _DEBUG_EVENTS:
        return
    cnt = int(_DEBUG_COUNTERS.get(event, 0)) + 1
    _DEBUG_COUNTERS[event] = cnt
    if _DEBUG_EVERY_N > 1 and (cnt % _DEBUG_EVERY_N) != 0:
        return
    parts = [f"[auto-ddp-debug rank={_RANK}] {event}"]
    for k, v in fields.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}ms" if "ms" in k or k.endswith("_ms") else f"{k}={v:.6f}")
        else:
            parts.append(f"{k}={v}")
    print(" | ".join(parts), flush=True)


def _summary_add(metric: str, ms: float) -> None:
    if not _SUMMARY_ENABLED:
        return
    _SUMMARY_ACC[metric] = float(_SUMMARY_ACC.get(metric, 0.0)) + float(ms)


def _summary_flush_if_needed(is_trainer: bool) -> None:
    global _SUMMARY_LAST_FLUSH_STEP, _SUMMARY_EPOCH_IDX
    if not _SUMMARY_ENABLED:
        return
    step = _SUMMARY_STEP_TOTAL
    if step <= 0:
        return
    window = None
    if _SUMMARY_STEPS_PER_EPOCH > 0 and (step % _SUMMARY_STEPS_PER_EPOCH) == 0:
        _SUMMARY_EPOCH_IDX += 1
        window = f"epoch={_SUMMARY_EPOCH_IDX}"
    elif _SUMMARY_EVERY_STEPS > 0 and (step % _SUMMARY_EVERY_STEPS) == 0:
        window = f"step_window_end={step}"
    if window is None:
        return

    window_steps = max(1, step - _SUMMARY_LAST_FLUSH_STEP)
    total_ms = sum(float(v) for v in _SUMMARY_ACC.values())
    parts = [
        f"[auto-ddp-summary rank={_RANK}] {window}",
        f"role={'trainer' if is_trainer else 'follower'}",
        f"steps={window_steps}",
        f"total_ms={total_ms:.3f}",
        f"avg_step_ms={total_ms / window_steps:.3f}",
    ]
    for k in sorted(_SUMMARY_ACC.keys()):
        parts.append(f"{k}_ms={float(_SUMMARY_ACC[k]):.3f}")
    print(" | ".join(parts), flush=True)
    _SUMMARY_ACC.clear()
    _SUMMARY_LAST_FLUSH_STEP = step


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
    os.environ.setdefault("TORCH_DDP_ASYMMETRIC_DEBUG_EVERY_N", "1")
    os.environ.setdefault("TORCH_DDP_ASYMMETRIC_DEBUG_EVENTS", "")
    os.environ.setdefault("TORCH_DDP_ASYMMETRIC_SUMMARY", "0")
    os.environ.setdefault("TORCH_DDP_ASYMMETRIC_SUMMARY_EVERY_STEPS", "0")
    os.environ.setdefault("TORCH_DDP_ASYMMETRIC_STEPS_PER_EPOCH", "0")
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
                wall_ms = (time.perf_counter() - t0) * 1000.0
                _debug_log(
                    "loss_follower_sync_scalar",
                    wall_ms=wall_ms,
                )
                _summary_add("loss_sync_scalar", wall_ms)
                return torch.tensor(float(scalar), dtype=torch.float32, device="cpu")

            # Trainer path: compute local scalar then broadcast to followers.
            t_loss = time.perf_counter()
            out = _ORIG_MODULE_CALL(self, *args, **kwargs)
            if isinstance(out, torch.Tensor) and out.numel() == 1:
                _LAST_DDP_FOR_LOSS.sync_scalar_from_trainer(float(out.detach().item()))
            wall_ms = (time.perf_counter() - t_loss) * 1000.0
            _debug_log(
                "loss_trainer_forward_plus_scalar_broadcast",
                wall_ms=wall_ms,
            )
            _summary_add("loss_forward_plus_scalar", wall_ms)
            return out

        ddp = _MODULE_DDP.get(self, None)
        if ddp is not None:
            # Prevent recursive bounce: DDP.forward internally calls self.module(...).
            if _IN_REDIRECTED_DDP_FORWARD:
                return _ORIG_MODULE_CALL(self, *args, **kwargs)
            _LAST_DDP_FOR_LOSS = ddp
            if _AUTO_SKIP_FOLLOWER_FORWARD and _RANK != _TRAINER_RANK and _is_dist_enabled():
                _debug_log("module_forward_skipped_follower", wall_ms=0.0)
                _summary_add("module_forward_skipped", 0.0)
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
                _summary_add(f"module_forward_{role}", wall_ms)
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
        if getattr(self, "_asym_step_wrapped", False):
            return

        # Wrap each optimizer instance directly so subclass overrides
        # (e.g., Adam.step) are always intercepted.
        self._asym_orig_step = self.step

        def _wrapped_instance_step(opt_self, *args, **kwargs):
            global _SUMMARY_STEP_TOTAL
            _ddp = _OPTIMIZER_DDP.get(opt_self, None)
            if _ddp is None:
                return opt_self._asym_orig_step(*args, **kwargs)

            is_trainer = bool(_ddp.is_trainer_rank())
            t_step = time.perf_counter()
            result = opt_self._asym_orig_step(*args, **kwargs) if is_trainer else None
            if is_trainer:
                step_ms = (time.perf_counter() - t_step) * 1000.0
                _debug_log(
                    "optimizer_step_local_trainer",
                    wall_ms=step_ms,
                )
                _summary_add("optimizer_step_local", step_ms)
            else:
                _debug_log("optimizer_step_skipped_follower", wall_ms=0.0)
                _summary_add("optimizer_step_skipped", 0.0)

            step = _OPTIMIZER_STEP_COUNT.get(opt_self, 0) + 1
            _OPTIMIZER_STEP_COUNT[opt_self] = step
            _SUMMARY_STEP_TOTAL = max(_SUMMARY_STEP_TOTAL, step)
            if step % _SYNC_INTERVAL == 0:
                t_sync = time.perf_counter()
                _ddp.sync_params_from_trainer()
                _summary_add("sync_params", (time.perf_counter() - t_sync) * 1000.0)
            _summary_flush_if_needed(is_trainer=is_trainer)
            return result

        self.step = types.MethodType(_wrapped_instance_step, self)
        self._asym_step_wrapped = True

    torch.optim.Optimizer.__init__ = _wrapped_opt_init


def _patch_optimizer_step() -> None:
    # No-op: step interception is done per optimizer instance in __init__
    # to cover subclass overrides like Adam.step.
    return None


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
    global _ENABLED, _DEBUG, _DEBUG_EVERY_N, _DEBUG_EVENTS
    global _SUMMARY_ENABLED, _SUMMARY_EVERY_STEPS, _SUMMARY_STEPS_PER_EPOCH
    global _RANK, _TRAINER_RANK, _SYNC_INTERVAL, _AUTO_SKIP_FOLLOWER_FORWARD
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
    _DEBUG_EVERY_N = max(1, int(os.environ.get("TORCH_DDP_ASYMMETRIC_DEBUG_EVERY_N", "1")))
    _DEBUG_EVENTS = {
        e.strip()
        for e in os.environ.get("TORCH_DDP_ASYMMETRIC_DEBUG_EVENTS", "").split(",")
        if e.strip()
    }
    _SUMMARY_ENABLED = os.environ.get("TORCH_DDP_ASYMMETRIC_SUMMARY", "0") == "1"
    _SUMMARY_EVERY_STEPS = max(
        0, int(os.environ.get("TORCH_DDP_ASYMMETRIC_SUMMARY_EVERY_STEPS", "0"))
    )
    _SUMMARY_STEPS_PER_EPOCH = max(
        0, int(os.environ.get("TORCH_DDP_ASYMMETRIC_STEPS_PER_EPOCH", "0"))
    )

    _init_pg_if_needed()
    _patch_module_init()
    _patch_module_call()
    _patch_optimizer_init()
    _patch_optimizer_step()
    _patch_backward_for_follower()
    _ENABLED = True

