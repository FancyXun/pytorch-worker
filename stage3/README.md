# Stage3: PyTorch fork workspace (路线 2)

`stage3` is the **in-tree workspace** for modifying **PyTorch 2.5** to support **非对称 / 自定义同步** 的训练语义（动 `DistributedDataParallel` / `Reducer` / `ProcessGroup` 一侧），而不是 Stage1/2 的 Python 侧编排。

Upstream source is **not** vendored in this repo by default: you place it under `stage3/pytorch/`.

---

## Layout

```text
stage3/
  README.md # this file
  pytorch/           # PyTorch 2.5 source tree (you add: clone or submodule)
```

After you populate `pytorch/`, the tree should look like the official repo (e.g. `setup.py`, `torch/`, `torch/csrc/distributed/`, `third_party/`, …).

---

## Getting PyTorch 2.5 here

From `stage3/`, clone into `pytorch/`. This repo ships `pytorch/.gitkeep` so Git keeps the folder; delete that file so `pytorch/` is empty, then clone (or clone elsewhere and move the tree into `pytorch/`).

```bash
cd stage3
rm -f pytorch/.gitkeep
git clone --recursive https://github.com/pytorch/pytorch.git pytorch
cd pytorch
git checkout v2.5.1   # or another 2.5.x tag you standardize on
git submodule sync && git submodule update --init --recursive
```

Alternatively use a **git submodule** from the `torch-worker` root if you want a pinned commit recorded in this repo.

---

## Build (reference only)

Build steps follow [official PyTorch from source](https://github.com/pytorch/pytorch#from-source) for your platform. Typical pattern:

```bash
cd stage3/pytorch
# install build deps per PyTorch docs, then e.g.:
export CMAKE_PREFIX_PATH="$(python -c 'import sys; print(sys.prefix)')"
python setup.py develop
```

Adjust for CUDA version and your toolchain; details belong in your team runbook once the tree is present.

---

## What we will change (路线 2, preview)

Edits target **distributed training internals** (custom sync instead of classic allreduce semantics where appropriate), for example under:

- `torch/csrc/distributed/c10d/`
- `torch/csrc/distributed/reducer.*`
- `torch/nn/parallel/distributed.py` (Python)

Exact patch plan is decided **after** the 2.5 tree is in place and a baseline build passes.

---

## Relation to Stage1 / Stage2

- **Stage1 / Stage2**: frozen RPC-style prototypes in this repo; Stage3 does **not** replace them for product experiments.
- **Stage3**: long-lived **fork** of PyTorch; merge/rebase cost is expected each upstream release.
