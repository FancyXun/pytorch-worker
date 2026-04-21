from __future__ import annotations

"""
Normal local STGNN training (single process, local CUDA if available).

Benchmark: each epoch prints `[bench] run=local ... epoch_sec=...` (same layout as ddp_train.py).
Override epochs: `--epochs N` or env `EPOCHS=N` (else `stgnn_config.json`).
"""
# pyright: reportMissingImports=false

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


STGNN_ROOT = os.path.dirname(__file__)
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
    # Block-diagonal repeat of the original graph, then crop.
    big = np.kron(np.eye(reps, dtype=np.float32), adj_mx)
    return big[:target_nodes, :target_nodes]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local STGNN train (single process)")
    p.add_argument("--epochs", type=int, default=None, help="Override stgnn_config.json epochs")
    return p.parse_args()


def main():
    args = _parse_args()
    data_cfg, model_cfg = _load_cfg()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    epochs = int(model_cfg["epochs"])
    if args.epochs is not None:
        epochs = int(args.epochs)
    elif os.environ.get("EPOCHS"):
        epochs = int(os.environ["EPOCHS"])

    torch.manual_seed(2026)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2026)

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

    model = STGNN(
        int(model_cfg["temp_feat"]),
        int(model_cfg["in_feat"]),
        int(model_cfg["hidden_feat"]),
        int(model_cfg["out_feat"]),
        int(model_cfg["pred_feat"]),
        float(model_cfg["drop_rate"]),
        bool(model_cfg["bias"]),
    ).to(device)
    base_adj = _build_adj().cpu().numpy()
    adj = torch.tensor(_expand_adj(base_adj, target_nodes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    print(
        f"[local train] device={device}, nodes={target_nodes}, "
        f"train_samples={len(train_dl.dataset)}, epochs={epochs}"
    )

    t_run0 = time.perf_counter()
    for epoch in range(epochs):
        t_ep0 = time.perf_counter()
        model.train()
        losses = []
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.squeeze(0).to(device)
            y_batch = y_batch.T.to(device)

            optimizer.zero_grad(set_to_none=True)
            y_pred = model(x_batch, adj)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_mse = sum(losses) / len(losses)
        epoch_sec = time.perf_counter() - t_ep0
        print(
            f"[bench] run=local device={device} epoch={epoch + 1}/{epochs} "
            f"epoch_mse={epoch_mse:.6f} epoch_sec={epoch_sec:.3f}"
        )
    print(
        f"[bench] run=local device={device} train_done epochs={epochs} "
        f"total_sec={time.perf_counter() - t_run0:.3f}"
    )


if __name__ == "__main__":
    main()

