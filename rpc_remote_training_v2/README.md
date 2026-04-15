# Extended RPC remote training (`rpc_remote_training_v2`)

`rpc_remote_training_v2` is a **separate** evolution of the baseline RPC split: **do not edit** [`rpc_remote_training`](../rpc_remote_training/README.md) when changing this package.

Worker entry:

```bash
python -m rpc_remote_training_v2.run_worker
```

Driver API: `from rpc_remote_training_v2.driver_api import Stage2Trainer`

---

## What v2 adds vs baseline (`rpc_remote_training`)

| Area | v2 change |
|------|-----------|
| Initial model | Still **cloudpickle** of `nn.Module` (same Python/PyTorch alignment rules apply). |
| CPU worker | `attach(..., allow_cpu_worker=True)` allows worker without CUDA (debug / CI). |
| AMP | `attach(..., use_amp=True)` uses `autocast` + `GradScaler` on **CUDA** only. |
| `forward(**kwargs)` | `trainer.step(x, y, forward_kwargs={"mask": m})` → `train_step_ex` on worker. |
| Custom loss | `attach(..., loss_module=nn.SomeLoss(...))` sends a cloudpickled loss module (trusted path). |
| Extra built-in losses | `smooth_l1` / `huber` string alias in worker. |
| Resume | `attach(..., resume_from="ckpt.pt")` or `load_checkpoint_to_worker` / `resume_worker_from_file`. |
| NumPy warning | `requirements.txt` includes `numpy` to silence common torch import warnings. |

---

## What is still not “unlimited”

- **cloudpickle for full model** remains: exotic modules / C extensions may still fail; **driver and worker Python should match**.
- **LBFGS / `optimizer.step(closure=...)`** not supported (same fundamental RPC step model).
- **world_size=2** (one driver + one worker); multi-GPU → use **DDP inside the worker process** (not implemented here).
- **Security**: `loss_blob` and `model_blob` are still **trusted serialization**; do not accept arbitrary user pickles in production without governance.

---

## Run (same pattern as baseline)

**Worker (GPU host):**

```bash
cd /path/to/torch-worker
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
PYTHONUNBUFFERED=1 python -m rpc_remote_training_v2.run_worker
```

**Driver:**

```bash
PYTHONUNBUFFERED=1 python -m rpc_remote_training_v2.example_train
```

Use the **same** `MASTER_ADDR` / `MASTER_PORT` on both processes.

---

## API sketch

```python
trainer = Stage2Trainer(master_addr="10.x.x.x", master_port=29500)
trainer.start_rpc()
trainer.attach(
    model,
    optimizer,
    loss="mse",
    use_amp=True,
    allow_cpu_worker=False,
    loss_module=None,
    resume_from=None,
)

loss_val = trainer.step(xb, yb)
loss_val = trainer.step(xb, yb, forward_kwargs={"mask": mask})

trainer.save_checkpoint("/tmp/ckpt.pt")
trainer.shutdown()
```

---

## “Fake DDP with zero driver weights”

That does **not** reproduce “only GPU trains” under stock DDP allreduce. This package stays on the **RPC remote-step** line; for asymmetric DDP across ranks, see [`pytorch_fork_asymmetric_ddp`](../pytorch_fork_asymmetric_ddp/README.md).

---

## See also

- [Repository overview](../README.md)
- [Baseline RPC (`rpc_remote_training`)](../rpc_remote_training/README.md)
- [PyTorch fork: asymmetric DDP (`pytorch_fork_asymmetric_ddp`)](../pytorch_fork_asymmetric_ddp/README.md)
