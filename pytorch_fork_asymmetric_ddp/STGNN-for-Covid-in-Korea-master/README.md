# STGNN sample (local baseline)

This tree is an **example customer project** bundled under [`pytorch_fork_asymmetric_ddp`](../README.md) for local experiments. It is **not** required to build the PyTorch fork.

## Run (CPU-friendly baseline)

From this directory, with PyTorch installed:

```bash
python train.py
```

Configuration and data paths are under `tests/configs/` and `stgraph_trainer/` per the original project layout.

## Relation to asymmetric DDP

Training here is **single-process** unless you add a multi-rank launcher and `DistributedDataParallel` using the **modified** PyTorch from `../pytorch/`. See the parent [README](../README.md) for fork setup and environment variables.

- [Back to `pytorch_fork_asymmetric_ddp`](../README.md)
- [Back to repository root](../../README.md)
