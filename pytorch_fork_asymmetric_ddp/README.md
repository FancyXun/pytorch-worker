# PyTorch fork: asymmetric DDP (`pytorch_fork_asymmetric_ddp`)

This directory is the **in-tree workspace** for a **fork of PyTorch 2.5** that adds **asymmetric distributed data parallel (DDP)** behavior: a designated trainer rank runs the full optimization step, while other ranks can skip gradient all-reduce and instead synchronize parameters from the trainer. Changes live in `Reducer`, Python `DistributedDataParallel`, and related bindings—not in the Python-only RPC prototypes under [`rpc_remote_training`](../rpc_remote_training/README.md) / [`rpc_remote_training_v2`](../rpc_remote_training_v2/README.md).

Upstream PyTorch is expected under `pytorch/` (large tree; may be a full git clone).

---

## Layout

```text
pytorch_fork_asymmetric_ddp/
  README.md                 # this file
  pytorch/                  # PyTorch source (you clone or submodule here)
  STGNN-for-Covid-in-Korea-master/   # example customer model / local CPU training baseline
```

After you populate `pytorch/`, the tree should match the official repo layout (`setup.py`, `torch/`, `torch/csrc/distributed/`, `third_party/`, …).

---

## Getting PyTorch 2.5 here

From this directory, clone into `pytorch/`. If you use `pytorch/.gitkeep` to reserve the folder, remove it first so `pytorch/` is empty, then clone (or clone elsewhere and move the tree into `pytorch/`).

```bash
cd pytorch_fork_asymmetric_ddp
rm -f pytorch/.gitkeep
git clone --recursive https://github.com/pytorch/pytorch.git pytorch
cd pytorch
git checkout v2.5.1   # or another 2.5.x tag your team standardizes on
git submodule sync && git submodule update --init --recursive
```

You can instead attach `pytorch/` as a **git submodule** from the `torch-worker` root if you want a pinned commit recorded in this repo.

---

## Build prerequisites (common failure)

`python setup.py develop` / `pip install -e .` expects **`cmake` or `cmake3` ≥ 3.18** on `PATH`. If you see `RuntimeError: no cmake or cmake3 with version >= 3.18.0 found`:

- **Debian / Ubuntu:** `apt-get update && apt-get install -y cmake ninja-build build-essential`, then `cmake --version`.
- **CMake still too old:** `python3 -m pip install "cmake>=3.18"` and put `$(python3 -m site --user-base)/bin` on `PATH`.

**Python packages before configure:** CMake runs `torchgen` with the same interpreter as `python3 setup.py`. If you see `ModuleNotFoundError: No module named 'typing_extensions'`, install build deps for that interpreter:

```bash
cd pytorch_fork_asymmetric_ddp/pytorch
python3 -m pip install typing_extensions pyyaml
python3 -m pip install -r requirements.txt
```

Use the **same** venv/conda env for `pip` and `setup.py`.

Full dependency list remains the [official from-source guide](https://github.com/pytorch/pytorch#from-source).

---

## Build (reference)

Follow [PyTorch from source](https://github.com/pytorch/pytorch#from-source) for your platform. Typical pattern:

```bash
cd pytorch_fork_asymmetric_ddp/pytorch
# install build deps per PyTorch docs, then e.g.:
export CMAKE_PREFIX_PATH="$(python -c 'import sys; print(sys.prefix)')"
python setup.py develop
```

Adjust for CUDA version and toolchain. You will usually build **two** install artifacts if CPU-only clients and GPU hosts must match protocol: e.g. a CPU build for the container and a CUDA build for the host, both from this fork.

---

## What was changed (Scheme B preview)

Edits target distributed training internals, for example:

- `torch/csrc/distributed/c10d/reducer.hpp` / `reducer.cpp` — trainer rank, optional skip of all-reduce.
- `torch/csrc/distributed/c10d/init.cpp` — Python bindings for the new `Reducer` options.
- `torch/nn/parallel/distributed.py` — env-driven asymmetric mode, `sync_params_from_trainer()`, `trainer_step()`, policies for non-trainer backward, logging.

Enable and tune via environment variables (see the top of `distributed.py` in the fork for the authoritative list), including:

- `TORCH_DDP_ASYMMETRIC_MODE`
- `TORCH_DDP_TRAINER_RANK`
- `TORCH_DDP_SKIP_ALLREDUCE`
- `TORCH_DDP_SYNC_INTERVAL`
- `TORCH_DDP_NON_TRAINER_BACKWARD`

---

## Example model repo (local baseline)

[`STGNN-for-Covid-in-Korea-master/`](STGNN-for-Covid-in-Korea-master/) holds a sample STGNN project used for local CPU training experiments (`train.py`). It does **not** depend on this fork until you install the modified PyTorch and write a multi-process / multi-rank script that uses `DistributedDataParallel` with the env vars above.

---

## Relation to the RPC packages

- **[`rpc_remote_training`](../rpc_remote_training/README.md)** / **[`rpc_remote_training_v2`](../rpc_remote_training_v2/README.md)**: frozen-style **Python RPC** prototypes (`torch.distributed.rpc`); good for experiments, different tradeoffs (e.g. cloudpickle, explicit remote step).
- **`pytorch_fork_asymmetric_ddp`**: long-lived **fork** of PyTorch; expect merge/rebase cost on each upstream release.

---

## Navigation

- [Back to repository overview](../README.md)
