# Stage1: RPC Remote Training (Worker Closes Step)

`stage1` implements a product-style training split using **official PyTorch RPC**:

- **Worker (GPU host)** runs `forward + loss + backward + optimizer.step` in one process.
- **Driver (CPU side, can be in Docker)** only does data loading, orchestration, logging, and checkpoint save.

This matches the Stage1 strategy: keep model compute on trusted GPU worker and keep user-side code mostly standard PyTorch.

---

## Architecture

- Transport/runtime: `torch.distributed.rpc` with `TensorPipeRpcBackendOptions`
- Worker entry: `stage1.run_worker`
- Driver API: `stage1.driver_api.Stage1Trainer`
- Core worker ops: `stage1.worker_ops`

Training step convention:

- `trainer.step(x, y)` for `model(x)`
- `trainer.step(x, adj, y)` for `model(x, adj)`

Rule: **last tensor is target**, previous tensors are `model.forward(*inputs)`.

---

## File Overview

- `run_worker.py`: GPU-side RPC process (rank=1).
- `driver_api.py`: driver-side API (`Stage1Trainer`).
- `worker_ops.py`: worker global session + train/infer/checkpoint RPC functions.
- `example_train.py`: single-input example.
- `example_two_input.py`: two-input example (`model(x, adj)`).

---

## Prerequisites

1. Worker host has NVIDIA GPU and CUDA driver ready (`nvidia-smi` works).
2. Worker and driver can both reach the same `MASTER_ADDR:MASTER_PORT`.
3. Use the same code version on both sides.
4. Start from the directory that contains the `stage1/` folder (important for `python -m stage1.xxx`).

---

## Environment Setup

### A) Worker host (GPU)

```bash
cd /path/to/torch-worker
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install GPU PyTorch (pick your CUDA wheel from pytorch.org), then:

```bash
pip install "cloudpickle>=3.0.0"
```

Quick check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### B) Driver side (CPU, inside Docker)

Start a long-running container first:

```bash
cd /path/to/torch-worker

docker run -itd \
  --name stage1-driver \
  -e MASTER_ADDR=10.60.82.27 \
  -e MASTER_PORT=29500 \
  -v "$(pwd):/workspace/torch-worker" \
  -w /workspace/torch-worker \
  python:3.11-bookworm \
  sleep infinity
```

Enter container and install CPU dependencies:

```bash
docker exec -it stage1-driver bash
cd /workspace/torch-worker
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install "cloudpickle>=3.0.0"
```

---

## Start and Run (Step-by-Step)

Assume:

- `MASTER_ADDR=10.60.82.27`
- `MASTER_PORT=29500`

### 1) Start worker first (GPU host)

```bash
cd /path/to/torch-worker
source .venv/bin/activate
export MASTER_ADDR=10.60.82.27
export MASTER_PORT=29500
PYTHONUNBUFFERED=1 python -m stage1.run_worker
```

Expected logs include:

- `process started; loading PyTorch...`
- `imports done.`
- `rendezvous MASTER_ADDR='10.60.82.27' MASTER_PORT=29500`
- `calling rpc.init_rpc — this BLOCKS ...`

`init_rpc` blocking here is normal until driver starts.

### 2) Run driver example (inside container)

```bash
docker exec -it stage1-driver bash
cd /workspace/torch-worker
export MASTER_ADDR=10.60.82.27
export MASTER_PORT=29500
PYTHONUNBUFFERED=1 python -m stage1.example_train
```

Or two-input model:

```bash
python -m stage1.example_two_input
```

When driver finishes and calls `shutdown`, worker exits `rpc.shutdown()` too.

---

## Using `Stage1Trainer` in Your Script

```python
from stage1.driver_api import Stage1Trainer

trainer = Stage1Trainer(master_addr="10.60.82.27", master_port=29500)
trainer.start_rpc()
trainer.attach(model, optimizer, loss="mse")  # loss: "mse" | "l1" | "cross_entropy"

for xb, yb in loader:
    loss_val = trainer.step(xb, yb)

# for model(x, adj):
# loss_val = trainer.step(xb, adj_tensor, yb)

trainer.save_checkpoint("/tmp/ckpt.pt", extra={"epoch": 1})
trainer.shutdown()
```

---

## Checkpoint APIs

Driver-side methods in `Stage1Trainer`:

- `fetch_model_state_dict()`
- `fetch_optimizer_state_dict()`
- `fetch_checkpoint()` -> `{"model": ..., "optimizer": ...}`
- `save_checkpoint(path, include_optimizer=True, extra=None)`
- `sync_local_model(model, strict=True)`

Note: training weights live on worker. Use these APIs to pull state to driver for saving/export.

---

## Common Pitfalls / Troubleshooting

1. **`ModuleNotFoundError: No module named 'stage1'`**
   - Run from project root (the parent directory containing `stage1/`), not inside `stage1/`.

2. **Worker shows no progress**
   - `rpc.init_rpc` waits for driver rank=0. Start driver after worker.

3. **Worker and driver cannot rendezvous**
   - `MASTER_ADDR` and `MASTER_PORT` must be exactly the same on both sides.
   - For host+docker split, avoid `127.0.0.1` unless both are in same network namespace.

4. **`hostname of the client socket cannot be retrieved. err=-3`**
   - Usually warning-only. If training proceeds, can ignore.
   - If hangs, verify routing/firewall and try setting interface:
     - `export GLOO_SOCKET_IFNAME=eth0`

5. **Docker image pull timeout**
   - Use mirror registry image name (example: `docker.m.daocloud.io/library/python:3.11-bookworm`) or configure registry mirrors.

---

## Stop / Cleanup

```bash
docker rm -f stage1-driver
```

Worker exits after driver shutdown; otherwise stop manually with `Ctrl+C`.
