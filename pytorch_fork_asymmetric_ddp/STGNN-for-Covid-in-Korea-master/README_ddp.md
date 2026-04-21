# STGNN heterogeneous asymmetric DDP

This mirrors `train.py` (same configs under `tests/configs/`, same data and `STGNN` model) but runs **two processes**:

| Role | Typical host | Script |
|------|----------------|--------|
| Trainer | GPU machine | `./run_trainer.sh` |
| Follower | CPU container | `./run_follower.sh` |

## Files

- `ddp_train.py` — distributed entry (argparse + `trainer_step` loop).
- `run_trainer.sh` / `run_follower.sh` — env vars + launch (same style as `test/cases/*/run_*.sh`).

## Run

```bash
cd STGNN-for-Covid-in-Korea-master
# host
./run_trainer.sh
# container (same MASTER_ADDR / MASTER_PORT)
./run_follower.sh
```

Default rendezvous port is **29623** (different from MNIST `29621` if you run both).

Smoke test (override long `epochs` in `stgnn_config.json`):

```bash
EPOCHS=2 ./run_trainer.sh
EPOCHS=2 ./run_follower.sh
```

## Compared to `train.py`

| `train.py` | `ddp_train.py` |
|------------|----------------|
| Single process | `init_process_group`, two ranks |
| `loss.backward(); optimizer.step()` | Trainer: backward + `ddp.trainer_step(optimizer)`; follower: `ddp.trainer_step(None)` |
| One device | Trainer CUDA, follower CPU |

Install the **forked** PyTorch with asymmetric DDP on **both** processes before running.

## Benchmarking (same config, compare wall time)

Both scripts log one line per epoch with the same fields so you can `grep '[bench]'`:

- **Local:** `python train.py` (optional `--epochs N` or `EPOCHS=N`). Uses `tests/configs/` like `ddp_train.py`.
- **Hetero:** `EPOCHS=N ./run_trainer.sh` and matching follower. Each rank prints `epoch_sec` for that epoch (they should be similar because steps synchronize).

Example:

```bash
# same epoch count
EPOCHS=5 python train.py 2>&1 | tee local.log
EPOCHS=5 ./run_trainer.sh 2>&1 | tee trainer.log   # plus follower in another terminal
grep '\[bench\]' local.log trainer.log
```

### Why local CPU can look faster than GPU hetero

Hetero training is **not** “GPU only”: every step still needs the **follower forward**, **Gloo** collectives, and **parameter sync** over the network. If the model and batch are small, that **communication and coordination** dominates; a single-process CPU run avoids all of that. Wall-clock `epoch_sec` reflects the **slowest path** (often sync-bound), not raw GPU FLOPs.
