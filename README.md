# torch-worker

Monorepo for experiments in **splitting PyTorch training** across a **CPU-side client** (e.g. container) and a **GPU-side host**, with three complementary tracks. Each track has its own README with prerequisites and run commands.

| Directory | Purpose |
|-----------|---------|
| [`rpc_remote_training/`](rpc_remote_training/README.md) | **Baseline remote training** using official `torch.distributed.rpc` (TensorPipe): GPU **worker** runs forward, loss, backward, and `optimizer.step`; CPU **driver** orchestrates and checkpoints. |
| [`rpc_remote_training_v2/`](rpc_remote_training_v2/README.md) | **Extended RPC prototype** (evolve separately from baseline): AMP, `forward(**kwargs)`, custom loss module, resume, optional CPU worker—still the same RPC “closed step on worker” model. |
| [`pytorch_fork_asymmetric_ddp/`](pytorch_fork_asymmetric_ddp/README.md) | **PyTorch fork workspace** (Scheme B): modify **DDP / Reducer** for **asymmetric** ranks (trainer vs followers), parameter sync from trainer, env-driven behavior. Includes `pytorch/` source tree and an example [`STGNN-for-Covid-in-Korea-master/`](pytorch_fork_asymmetric_ddp/STGNN-for-Covid-in-Korea-master/) baseline. |

---

## Quick links

- **Run baseline RPC (driver + worker):** [rpc_remote_training → Start and Run](rpc_remote_training/README.md#start-and-run-step-by-step)
- **Run extended RPC v2:** [rpc_remote_training_v2 → Run](rpc_remote_training_v2/README.md#run-same-pattern-as-baseline)
- **Clone / build forked PyTorch and asymmetric DDP notes:** [pytorch_fork_asymmetric_ddp](pytorch_fork_asymmetric_ddp/README.md)

---

## Historical names

These directories were previously named `stage1`, `stage2`, and `stage3`. Documentation and `python -m` entry points now use the names above.

---

## Layout (top level)

```text
torch-worker/
  README.md                      # this file
  rpc_remote_training/           # baseline RPC package + examples
  rpc_remote_training_v2/        # extended RPC package + examples
  pytorch_fork_asymmetric_ddp/     # PyTorch fork + optional STGNN sample
```
