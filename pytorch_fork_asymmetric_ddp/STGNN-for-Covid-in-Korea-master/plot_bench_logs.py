#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

EPOCH_RE = re.compile(
    r"^\[bench\] run=(?P<run>\w+)\s+"
    r"(?:rank=(?P<rank>\d+)\s+)?"
    r"device=(?P<device>\S+)\s+"
    r"epoch=(?P<epoch>\d+)/(?P<epochs>\d+)\s+"
    r"epoch_mse=(?P<mse>[-+0-9.eEnNaA]+)\s+"
    r"epoch_sec=(?P<sec>[-+0-9.eE]+)"
)


def parse_float(token: str) -> float:
    if token.lower() == "nan":
        return math.nan
    return float(token)


def parse_log(path: Path) -> Dict[int, Dict[str, float]]:
    rows: Dict[int, Dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = EPOCH_RE.match(line.strip())
        if not m:
            continue
        epoch = int(m.group("epoch"))
        rows[epoch] = {
            "mse": parse_float(m.group("mse")),
            "sec": float(m.group("sec")),
        }
    if not rows:
        raise ValueError(f"No [bench] lines parsed from {path}")
    return rows


def ordered_xy(rows: Dict[int, Dict[str, float]], field: str) -> tuple[List[int], List[float]]:
    xs = sorted(rows.keys())
    ys = [rows[e][field] for e in xs]
    return xs, ys


def mean_sec(rows: Dict[int, Dict[str, float]], warmup_epochs: int) -> float:
    xs = sorted(rows.keys())
    vals = [rows[e]["sec"] for e in xs if e > warmup_epochs]
    if not vals:
        vals = [rows[e]["sec"] for e in xs]
    return sum(vals) / len(vals)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate benchmark comparison plots from STGNN logs")
    p.add_argument("--local-gpu-log", required=True)
    p.add_argument("--local-cpu-log", required=True)
    p.add_argument("--hetero-trainer-log", required=True)
    p.add_argument("--hetero-follower-log", required=True)
    p.add_argument("--out-dir", default="logs/plots", help="Directory to write png figures")
    p.add_argument("--warmup-epochs", type=int, default=1, help="Exclude first N epochs for mean bars")
    p.add_argument("--title-prefix", default="STGNN", help="Prefix for plot titles")
    args = p.parse_args()

    datasets = {
        "local_gpu": parse_log(Path(args.local_gpu_log)),
        "local_cpu": parse_log(Path(args.local_cpu_log)),
        "hetero_trainer": parse_log(Path(args.hetero_trainer_log)),
        "hetero_follower": parse_log(Path(args.hetero_follower_log)),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Epoch time curves
    plt.figure(figsize=(11, 6))
    for name in ("local_gpu", "local_cpu", "hetero_trainer", "hetero_follower"):
        xs, ys = ordered_xy(datasets[name], "sec")
        plt.plot(xs, ys, label=name, linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("epoch_sec")
    plt.title(f"{args.title_prefix} - Epoch Time Comparison")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    time_png = out_dir / "epoch_time_comparison.png"
    plt.savefig(time_png, dpi=160)
    plt.close()

    # 2) Epoch MSE curves
    plt.figure(figsize=(11, 6))
    for name in ("local_gpu", "local_cpu", "hetero_trainer", "hetero_follower"):
        xs, ys = ordered_xy(datasets[name], "mse")
        plt.plot(xs, ys, label=name, linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("epoch_mse")
    plt.title(f"{args.title_prefix} - Epoch MSE Comparison")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    mse_png = out_dir / "epoch_mse_comparison.png"
    plt.savefig(mse_png, dpi=160)
    plt.close()

    # 3) Mean epoch_sec bar chart (excluding warmup)
    labels = ["local_gpu", "local_cpu", "hetero_trainer", "hetero_follower"]
    means = [mean_sec(datasets[name], args.warmup_epochs) for name in labels]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, means)
    for b, v in zip(bars, means):
        plt.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height(),
            f"{v:.3f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.ylabel("mean epoch_sec")
    plt.title(f"{args.title_prefix} - Mean Epoch Time (warmup>{args.warmup_epochs})")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    bar_png = out_dir / "mean_epoch_time_bar.png"
    plt.savefig(bar_png, dpi=160)
    plt.close()

    print(f"generated: {time_png}")
    print(f"generated: {mse_png}")
    print(f"generated: {bar_png}")


if __name__ == "__main__":
    main()
