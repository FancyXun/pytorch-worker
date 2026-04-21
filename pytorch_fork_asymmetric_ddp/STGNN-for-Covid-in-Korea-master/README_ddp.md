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
