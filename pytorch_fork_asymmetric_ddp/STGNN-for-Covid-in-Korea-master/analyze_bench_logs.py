#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

EPOCH_RE = re.compile(
    r"^\[bench\] run=(?P<run>\w+)\s+"
    r"(?:rank=(?P<rank>\d+)\s+)?"
    r"device=(?P<device>\S+)\s+"
    r"epoch=(?P<epoch>\d+)/(?P<epochs>\d+)\s+"
    r"epoch_mse=(?P<mse>[-+0-9.eEnNaA]+)\s+"
    r"epoch_sec=(?P<sec>[-+0-9.eE]+)"
)


def _parse_float(token: str) -> float:
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
            "mse": _parse_float(m.group("mse")),
            "sec": float(m.group("sec")),
        }
    if not rows:
        raise ValueError(f"No [bench] epoch lines found in {path}")
    return rows


def summarize_seconds(rows: Dict[int, Dict[str, float]], warmup_epochs: int) -> Dict[str, float]:
    items = sorted(rows.items())
    filtered = [r["sec"] for epoch, r in items if epoch > warmup_epochs]
    if not filtered:
        filtered = [r["sec"] for _, r in items]
    return {
        "epochs_counted": len(filtered),
        "mean_sec": statistics.fmean(filtered),
        "median_sec": statistics.median(filtered),
        "min_sec": min(filtered),
        "max_sec": max(filtered),
    }


def compare_metric(
    left: Dict[int, Dict[str, float]],
    right: Dict[int, Dict[str, float]],
) -> Dict[str, float]:
    common_epochs = sorted(set(left.keys()) & set(right.keys()))
    diffs: List[float] = []
    for e in common_epochs:
        lv = left[e]["mse"]
        rv = right[e]["mse"]
        if math.isnan(lv) or math.isnan(rv):
            continue
        diffs.append(abs(lv - rv))
    if not diffs:
        return {"epochs_compared": 0, "max_abs_diff": math.nan, "mean_abs_diff": math.nan}
    return {
        "epochs_compared": len(diffs),
        "max_abs_diff": max(diffs),
        "mean_abs_diff": statistics.fmean(diffs),
    }


def fmt(num: float) -> str:
    if math.isnan(num):
        return "nan"
    return f"{num:.6f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze STGNN [bench] logs for speed and metric consistency")
    p.add_argument("--local-gpu-log", required=True)
    p.add_argument("--local-cpu-log", required=True)
    p.add_argument("--hetero-trainer-log", required=True)
    p.add_argument("--hetero-follower-log", required=True)
    p.add_argument("--warmup-epochs", type=int, default=1, help="Ignore first N epochs in perf stats")
    p.add_argument("--mse-tol", type=float, default=1e-6, help="Consistency threshold for mse max abs diff")
    p.add_argument("--json-out", default="", help="Optional path to write machine-readable report")
    args = p.parse_args()

    logs = {
        "local_gpu": parse_log(Path(args.local_gpu_log)),
        "local_cpu": parse_log(Path(args.local_cpu_log)),
        "hetero_trainer": parse_log(Path(args.hetero_trainer_log)),
        "hetero_follower": parse_log(Path(args.hetero_follower_log)),
    }

    perf = {name: summarize_seconds(rows, args.warmup_epochs) for name, rows in logs.items()}
    cmp_gpu_vs_hetero = compare_metric(logs["local_gpu"], logs["hetero_trainer"])
    cmp_hetero_ranks = compare_metric(logs["hetero_trainer"], logs["hetero_follower"])

    speedup_local_cpu_over_hetero = perf["local_cpu"]["mean_sec"] / perf["hetero_trainer"]["mean_sec"]
    slowdown_hetero_vs_local_gpu = perf["hetero_trainer"]["mean_sec"] / perf["local_gpu"]["mean_sec"]

    metric_ok = (
        (not math.isnan(cmp_gpu_vs_hetero["max_abs_diff"]))
        and (cmp_gpu_vs_hetero["max_abs_diff"] <= args.mse_tol)
        and (not math.isnan(cmp_hetero_ranks["max_abs_diff"]))
        and (cmp_hetero_ranks["max_abs_diff"] <= args.mse_tol)
    )

    print("== Performance (epoch_sec) ==")
    for name in ("local_gpu", "local_cpu", "hetero_trainer", "hetero_follower"):
        s = perf[name]
        print(
            f"{name:16s} mean={s['mean_sec']:.3f}s median={s['median_sec']:.3f}s "
            f"min={s['min_sec']:.3f}s max={s['max_sec']:.3f}s counted={s['epochs_counted']}"
        )

    print("\n== Speed ratios (using mean epoch_sec) ==")
    print(f"local_cpu / hetero_trainer : {speedup_local_cpu_over_hetero:.3f}x")
    print(f"hetero_trainer / local_gpu : {slowdown_hetero_vs_local_gpu:.3f}x")

    print("\n== Metric consistency (epoch_mse abs diff) ==")
    print(
        "local_gpu vs hetero_trainer : "
        f"epochs={cmp_gpu_vs_hetero['epochs_compared']} "
        f"max={fmt(cmp_gpu_vs_hetero['max_abs_diff'])} "
        f"mean={fmt(cmp_gpu_vs_hetero['mean_abs_diff'])}"
    )
    print(
        "hetero_trainer vs follower  : "
        f"epochs={cmp_hetero_ranks['epochs_compared']} "
        f"max={fmt(cmp_hetero_ranks['max_abs_diff'])} "
        f"mean={fmt(cmp_hetero_ranks['mean_abs_diff'])}"
    )
    print(f"\nmetric_consistency_ok (tol={args.mse_tol}): {metric_ok}")

    report = {
        "perf": perf,
        "speed_ratios": {
            "local_cpu_over_hetero_trainer": speedup_local_cpu_over_hetero,
            "hetero_trainer_over_local_gpu": slowdown_hetero_vs_local_gpu,
        },
        "metric_consistency": {
            "local_gpu_vs_hetero_trainer": cmp_gpu_vs_hetero,
            "hetero_trainer_vs_follower": cmp_hetero_ranks,
            "tol": args.mse_tol,
            "ok": metric_ok,
        },
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\njson_report: {out}")


if __name__ == "__main__":
    main()
