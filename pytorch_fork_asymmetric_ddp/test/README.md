# Heterogeneous asymmetric DDP demos

Two **separate** examples live under `cases/`. They share the same forked PyTorch runtime but differ in model/data and what you copy for customer integration.

| Case | Directory | Purpose |
|------|-----------|---------|
| Synthetic MLP + MSE | `cases/synthetic_mlp/` | Minimal loop: random data, MSE, follower checkpoints. Shows **what a user must change** in a tiny training loop. |
| MNIST + CE + metrics | `cases/mnist/` | Public MNIST, CNN, trainer/follower both print `metric ...` lines for manual log comparison; optional raw IDX download if torchvision is unusable. |

Diagrams (`*.drawio`) stay in this `test/` folder; they are not tied to a single case.

**Run:** always `cd` into the case directory first, then `./run_trainer.sh` / `./run_follower.sh` (see each case’s README).
