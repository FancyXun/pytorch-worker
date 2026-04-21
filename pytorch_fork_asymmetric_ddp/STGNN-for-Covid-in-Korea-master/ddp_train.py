#!/usr/bin/env python3
"""Heterogeneous asymmetric DDP training for STGNN (GPU trainer + CPU follower).

Mirrors train.py data/model setup; only the optimization loop is split.
Default roles in this script are:
  - rank 0: trainer (GPU)
  - rank 1: follower (CPU)

Optimization loop:
  - trainer: forward + backward + ddp.trainer_step(optimizer)
  - follower: ddp.trainer_step(None), and can skip forward for speed experiments

Requires the team's forked PyTorch with asymmetric DDP (TORCH_DDP_* env vars).
Each rank loads the same DataLoader order (shuffle=False) like native multi-process training.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

STGNN_ROOT = os.path.dirname(os.path.abspath(__file__))
if STGNN_ROOT not in sys.path:
    sys.path.insert(0, STGNN_ROOT)

from stgraph_trainer.datasets import (  # noqa: E402
    load_province_coordinates,
    load_province_temporal_data,
    preprocess_data_for_stgnn,
)
from stgraph_trainer.models.stgnn import STGNN  # noqa: E402
from stgraph_trainer.utils.config import get_config_from_json  # noqa: E402
from stgraph_trainer.utils.utils import (  # noqa: E402
    PairDataset,
    get_adjacency_matrix,
    get_distance_in_km_between_earth_coordinates,
    get_normalized_adj,
)


def _configure_asymmetric_env(trainer_rank: int) -> None:
    os.environ.setdefault("TORCH_DDP_ASYMMETRIC_MODE", "1")
    os.environ.setdefault("TORCH_DDP_TRAINER_RANK", str(trainer_rank))
    os.environ.setdefault("TORCH_DDP_SKIP_ALLREDUCE", "1")
    os.environ.setdefault("TORCH_DDP_NON_TRAINER_FORWARD_ONLY", "1")
    os.environ.setdefault("TORCH_DDP_NON_TRAINER_BACKWARD", "error")
    os.environ.setdefault("TORCH_DDP_SYNC_INTERVAL", "1")
    os.environ.setdefault("TORCH_DDP_HETERO_PARAM_SYNC", "1")


def _load_cfg():
    cfg_dir = os.path.join(STGNN_ROOT, "tests", "configs")
    data_cfg = get_config_from_json(os.path.join(cfg_dir, "data_config.json"))
    model_cfg = get_config_from_json(os.path.join(cfg_dir, "stgnn_config.json"))
    return data_cfg, model_cfg


def _build_adj() -> torch.Tensor:
    province_coords = load_province_coordinates().values[:, 1:]
    dist_km = []
    for c1 in province_coords:
        dist_km.append(
            [get_distance_in_km_between_earth_coordinates(c1, c2) for c2 in province_coords]
        )
    dist_mx = np.array(dist_km)
    adj_mx = get_adjacency_matrix(dist_mx).astype(np.float32)
    adj_mx = get_normalized_adj(adj_mx)
    return torch.tensor(adj_mx)


def _expand_node_data(
    x_train: np.ndarray, y_train: np.ndarray, target_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    base_nodes = x_train.shape[1]
    if target_nodes <= base_nodes:
        return x_train[:, :target_nodes, :], y_train[:, :target_nodes]
    reps = int(np.ceil(target_nodes / base_nodes))
    x_big = np.concatenate([x_train] * reps, axis=1)[:, :target_nodes, :]
    y_big = np.concatenate([y_train] * reps, axis=1)[:, :target_nodes]
    return x_big, y_big


def _expand_adj(adj_mx: np.ndarray, target_nodes: int) -> np.ndarray:
    base_nodes = adj_mx.shape[0]
    if target_nodes <= base_nodes:
        return adj_mx[:target_nodes, :target_nodes]
    reps = int(np.ceil(target_nodes / base_nodes))
    big = np.kron(np.eye(reps, dtype=np.float32), adj_mx)
    return big[:target_nodes, :target_nodes]


def _param_sum(model: torch.nn.Module) -> float:
    total = 0.0
    with torch.no_grad():
        for p in model.parameters():
            total += float(p.detach().float().sum().item())
    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STGNN hetero asymmetric DDP")
    p.add_argument("--rank", type=int, required=True)
    p.add_argument(
        "--trainer-rank",
        type=int,
        default=0,
        help="Default trainer rank is 0 (rank0 trainer, rank1 follower).",
    )
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--master-addr", default="127.0.0.1")
    p.add_argument("--master-port", type=int, default=29623)
    p.add_argument(
        "--init-timeout-sec",
        type=int,
        default=int(os.environ.get("INIT_TIMEOUT_SEC", "90")),
        help="Timeout (seconds) for dist.init_process_group (default: 90).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override stgnn_config.json epochs (default: use config).",
    )
    p.add_argument(
        "--log-interval",
        type=int,
        default=0,
        help="If >0, also print a metric line every N batches within an epoch.",
    )
    p.add_argument("--save-every-epochs", type=int, default=0)
    p.add_argument("--save-dir", default="/tmp/stgnn_hetero_ckpt")
    p.add_argument(
        "--skip-follower-forward",
        action="store_true",
        default=(os.environ.get("SKIP_FOLLOWER_FORWARD", "1") == "1"),
        help=(
            "If set, follower rank skips ddp forward and only participates in "
            "trainer_step() sync. Default enabled via SKIP_FOLLOWER_FORWARD=1."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _configure_asymmetric_env(args.trainer_rank)

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["RANK"] = str(args.rank)

    is_store_master = args.rank == 0
    print(
        f"[rank{args.rank}] init_pg_start backend=gloo master={args.master_addr}:{args.master_port} "
        f"world_size={args.world_size} timeout_sec={args.init_timeout_sec} "
        f"is_store_master={is_store_master} iface={os.environ.get('GLOO_SOCKET_IFNAME', '<unset>')}",
        flush=True,
    )
    if args.rank != 0:
        print(
            "[init hint] rank0 must be running first for env:// rendezvous (TCPStore host).",
            flush=True,
        )
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        timeout=timedelta(seconds=args.init_timeout_sec),
    )
    print(f"[rank{args.rank}] init_pg_done", flush=True)
    rank = args.rank

    if rank == args.trainer_rank:
        if not torch.cuda.is_available():
            raise RuntimeError("Trainer rank requires CUDA.")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print(
            f"[rank{rank}] gpu_check cuda_available={torch.cuda.is_available()} "
            f"device_count={torch.cuda.device_count()}",
            flush=True,
        )
    else:
        device = torch.device("cpu")
        print(
            f"[rank{rank}] gpu_check cuda_available={torch.cuda.is_available()} "
            f"device_count={torch.cuda.device_count()} "
            f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
            flush=True,
        )

    torch.manual_seed(2026)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2026)

    data_cfg, model_cfg = _load_cfg()
    epochs = int(args.epochs) if args.epochs is not None else int(model_cfg["epochs"])

    df = load_province_temporal_data(
        provinces=data_cfg["provinces"], status=data_cfg["status"]
    )
    x_train, y_train, _x_test, _y_test, _train, _test, _scaler = preprocess_data_for_stgnn(
        df, data_cfg["split_date"], int(data_cfg["time_steps"])
    )
    target_nodes = int(model_cfg.get("target_nodes", x_train.shape[1]))
    x_train, y_train = _expand_node_data(x_train, y_train, target_nodes)

    train_dl = DataLoader(
        PairDataset(x_train, y_train),
        batch_size=int(model_cfg["batch_size"]),
        shuffle=False,
    )

    base_adj = _build_adj().cpu().numpy()
    adj = torch.tensor(_expand_adj(base_adj, target_nodes), dtype=torch.float32).to(device)

    model = STGNN(
        int(model_cfg["temp_feat"]),
        int(model_cfg["in_feat"]),
        int(model_cfg["hidden_feat"]),
        int(model_cfg["out_feat"]),
        int(model_cfg["pred_feat"]),
        float(model_cfg["drop_rate"]),
        bool(model_cfg["bias"]),
    ).to(device)

    if device.type == "cuda":
        ddp = DDP(model, device_ids=[0], output_device=0, broadcast_buffers=False)
    else:
        ddp = DDP(model, broadcast_buffers=False)

    is_trainer = ddp.is_trainer_rank()
    optimizer = torch.optim.Adam(ddp.parameters(), lr=0.001) if is_trainer else None
    loss_fn = torch.nn.MSELoss()

    print(
        f"[rank{rank}] stgnn_hetero device={device} nodes={target_nodes} "
        f"batches_per_epoch={len(train_dl)} epochs={epochs} "
        f"skip_follower_forward={args.skip_follower_forward} "
        f"cfg={ddp.get_asymmetric_mode_config()}",
        flush=True,
    )
    dist.barrier()

    ckpt_dir = Path(args.save_dir) if args.save_every_epochs > 0 else None
    if ckpt_dir is not None and (not is_trainer):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[rank{rank}] checkpoint_dir={ckpt_dir}", flush=True)

    t_run0 = time.perf_counter()
    global_step = 0

    for epoch in range(epochs):
        t_ep0 = time.perf_counter()
        ddp.train()
        losses: list[float] = []
        for batch_idx, (x_batch, y_batch) in enumerate(train_dl):
            x_batch = x_batch.squeeze(0).to(device)
            y_batch = y_batch.T.to(device)

            if is_trainer:
                optimizer.zero_grad(set_to_none=True)
                y_pred = ddp(x_batch, adj)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                loss_scalar = float(loss.item())
            elif args.skip_follower_forward:
                # Follower participates in parameter sync only; no local forward.
                loss_scalar = float("nan")
            else:
                y_pred = ddp(x_batch, adj)
                loss = loss_fn(y_pred, y_batch)
                loss_scalar = float(loss.item())

            ddp.trainer_step(optimizer if is_trainer else None)

            if not np.isnan(loss_scalar):
                losses.append(loss_scalar)
            if (
                is_trainer
                and args.log_interval > 0
                and (batch_idx % args.log_interval == 0)
            ):
                print(
                    f"metric rank={rank} epoch={epoch} batch={batch_idx} step={global_step} "
                    f"train_mse={loss_scalar:.6f} param_sum={_param_sum(ddp.module):.4f}",
                    flush=True,
                )
            global_step += 1

        epoch_mse = (sum(losses) / len(losses)) if losses else float("nan")
        epoch_sec = time.perf_counter() - t_ep0
        print(
            f"[bench] run=hetero rank={rank} device={device} epoch={epoch + 1}/{epochs} "
            f"epoch_mse={epoch_mse:.6f} epoch_sec={epoch_sec:.3f}",
            flush=True,
        )

        if (
            (not is_trainer)
            and ckpt_dir is not None
            and args.save_every_epochs > 0
            and ((epoch + 1) % args.save_every_epochs == 0)
        ):
            ckpt_path = ckpt_dir / f"follower_epoch_{epoch + 1}.pt"
            torch.save(
                {"model": ddp.module.state_dict(), "epoch": epoch + 1, "saved_at": time.time()},
                ckpt_path,
            )
            print(f"[rank{rank}] checkpoint_saved={ckpt_path}", flush=True)

    print(
        f"[bench] run=hetero rank={rank} device={device} train_done epochs={epochs} "
        f"total_sec={time.perf_counter() - t_run0:.3f}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
