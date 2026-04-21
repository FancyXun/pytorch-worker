# STGNN heterogeneous asymmetric DDP

This mirrors `train.py` (same configs under `tests/configs/`, same data and `STGNN` model) but runs **two processes**:

| Role | Typical host | Script |
|------|----------------|--------|
| Trainer | GPU machine | `./run_trainer.sh` |
| Follower | CPU container | `./run_follower.sh` |

Default role mapping in scripts:

- `rank=0` is trainer (GPU)
- `rank=1` is follower (CPU)

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
- Follower always skips local forward and only joins `trainer_step()` sync each step.

Example:

```bash
# same epoch count
EPOCHS=5 python train.py 2>&1 | tee local.log
EPOCHS=5 ./run_trainer.sh 2>&1 | tee trainer.log   # plus follower in another terminal
grep '\[bench\]' local.log trainer.log
```

## Four-log benchmark workflow

Collect four logs for performance and metric checks:

1) Local GPU baseline (`train.py`)
2) Local CPU baseline (`train.py`)
3) Hetero trainer (GPU rank)
4) Hetero follower (CPU rank)

Commands:

```bash
cd STGNN-for-Covid-in-Korea-master
mkdir -p logs

# local gpu
DEVICE_MODE=gpu EPOCHS=5 LOG_FILE=logs/local_gpu.log ./run_local.sh

# local cpu
DEVICE_MODE=cpu EPOCHS=5 LOG_FILE=logs/local_cpu.log ./run_local.sh

# hetero (run on separate hosts/containers with same MASTER_ADDR/MASTER_PORT)
EPOCHS=5 LOG_FILE=logs/hetero_trainer.log ./run_trainer.sh
EPOCHS=5 LOG_FILE=logs/hetero_follower.log ./run_follower.sh
```

Then analyze:

```bash
python3 analyze_bench_logs.py \
  --local-gpu-log logs/local_gpu.log \
  --local-cpu-log logs/local_cpu.log \
  --hetero-trainer-log logs/hetero_trainer.log \
  --hetero-follower-log logs/hetero_follower.log \
  --warmup-epochs 1 \
  --mse-tol 1e-6 \
  --json-out logs/bench_report.json
```

Generate documentation-ready plots:

```bash
python3 plot_bench_logs.py \
  --local-gpu-log logs/local_gpu.log \
  --local-cpu-log logs/local_cpu.log \
  --hetero-trainer-log logs/hetero_trainer.log \
  --hetero-follower-log logs/hetero_follower.log \
  --warmup-epochs 1 \
  --title-prefix "STGNN 100 epochs" \
  --out-dir logs/plots
```

Output files:

- `logs/plots/epoch_time_comparison.png`
- `logs/plots/epoch_mse_comparison.png`
- `logs/plots/mean_epoch_time_bar.png`

### Why local CPU can look faster than GPU hetero

Hetero training is still not “GPU only”: each step must coordinate across ranks with **Gloo** and **parameter sync** over the network. Even with follower forward skipped, wall-clock `epoch_sec` is set by the slowest distributed path (sync + CPU/network overhead), not raw GPU FLOPs.
