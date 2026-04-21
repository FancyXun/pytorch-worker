# Case 2: MNIST (cross-entropy + accuracy)

## Files

| File | Role |
|------|------|
| `ddp_hetero_mnist.py` | Single-file demo: dataset, CNN, broadcast batch from trainer, `trainer_step`, **per-rank** `metric ...` prints (train loss, test CE loss, accuracy), follower checkpoints. |
| `run_trainer.sh` | Host GPU launcher. |
| `run_follower.sh` | Container CPU launcher + lists saved `.pt` files. |

This is a **second**, standalone case: it does **not** import `ddp_hetero_common.py`.

## What a customer would change vs single-machine training

Same runtime and asymmetric `backward` / `trainer_step` split as Case 1, plus:

1. **Batch alignment:** this demo uses `dist.broadcast` so follower sees the same batch as trainer; a real app might share indices or a dataloader protocol instead.
2. **Metrics:** print whatever they use (MSE, accuracy, etc.) on **each** rank if they want to diff logs manually—same idea as the `metric rank=...` lines here.
3. **Data:** MNIST here; swap for their dataset. If torchvision conflicts with a custom torch build, this script falls back to raw MNIST IDX download.

## Run

```bash
cd test/cases/mnist
# host
./run_trainer.sh
# container
./run_follower.sh
```

Use the same `MASTER_PORT` on both sides (default `29621`).
