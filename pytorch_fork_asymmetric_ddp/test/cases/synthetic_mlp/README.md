# Case 1: Synthetic MLP (MSE)

## Files

| File | Role |
|------|------|
| `ddp_hetero_common.py` | Shared training loop: `init_process_group`, DDP, `trainer_step`, logging, follower `torch.save`. |
| `ddp_hetero_role.py` | CLI entry: parses args and calls `run_hetero_role`. |
| `run_trainer.sh` | Host GPU: env vars + `python3 ddp_hetero_role.py --rank 0 ...` |
| `run_follower.sh` | Container CPU: same + optional checkpoint interval. |

This is **one** runnable case: two processes, same code path, different `--rank`.

## What a customer would change vs single-machine training

1. **Runtime:** install your team’s forked `torch` (asymmetric DDP), set `TORCH_DDP_*` env vars (or keep defaults from the shell scripts).
2. **Process layout:** two processes (or one binary, two invocations): trainer on GPU, follower on CPU; matching `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`.
3. **Training loop (inside something like `run_hetero_role`):** only trainer calls `loss.backward()` and passes `optimizer` into `ddp.trainer_step(optimizer)`; follower calls `ddp.trainer_step(None)`. Inputs are replicated with a per-step CPU RNG seed then `.to(device)` so both ranks see the same batch (like a shared dataloader). No extra `broadcast` / `barrier` in the step loop—rely on collectives inside `trainer_step()`.
4. **Model / data / loss:** replace the `Sequential` + `mse_loss` block with the customer’s model and objective; keep the asymmetric step pattern above.

No MNIST, no `metric` prefix logging—this case is intentionally minimal.
