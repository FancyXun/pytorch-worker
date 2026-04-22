# Auto DDP Wrap (Zero user-script edits)

This launcher mode allows a user to run an existing single-process PyTorch
training script without editing Python code for DDP role branching.

## How it works

1. `sitecustomize.py` auto-runs at interpreter startup.
2. It enables `torch.distributed.asymmetric_autowrap.enable_from_env()`.
3. Runtime patches:
   - auto-init process group from env (`MASTER_ADDR/PORT`, `RANK`, `WORLD_SIZE`)
   - auto-wrap detected model in asymmetric `DDP`
   - on follower rank: `backward()` becomes no-op
   - `optimizer.step()` is intercepted:
     - trainer rank executes real step
     - all ranks perform param sync from trainer at configured interval
   - optional auto skip follower forward (enabled by launcher):
     - follower model call returns a token instead of real forward compute
     - scalar loss is broadcast from trainer during criterion call

## Launch

On trainer host:

```bash
MASTER_ADDR=10.60.82.27 MASTER_PORT=29623 \
./auto_ddp/run_user_trainer.sh /path/to/user_train.py --arg1 xxx
```

On follower host:

```bash
MASTER_ADDR=10.60.82.27 MASTER_PORT=29623 \
./auto_ddp/run_user_follower.sh /path/to/user_train.py --arg1 xxx
```

## Notes

- This mode targets common "single model + single optimizer + scalar criterion" training style.
- It removes explicit `is_trainer` branching from user code.
- Follower forward skip is enabled by default via
  `TORCH_DDP_AUTO_SKIP_FOLLOWER_FORWARD=1` in launcher scripts.
- For uncommon training loops (closure/multi-model/custom loss control flow),
  disable skip via `TORCH_DDP_AUTO_SKIP_FOLLOWER_FORWARD=0`.

