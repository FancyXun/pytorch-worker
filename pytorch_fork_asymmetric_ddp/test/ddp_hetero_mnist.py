#!/usr/bin/env python3
"""Asymmetric DDP MNIST demo with cross-rank consistency checks.

Goal:
1) Train on open-source MNIST dataset.
2) Verify follower (CPU/container side) sees consistent loss and parameters.
3) Print accuracy so behavior is observable end-to-end.
"""

from __future__ import annotations

import argparse
import gzip
import os
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:  # Optional fast path
    from torchvision import datasets, transforms

    HAS_TORCHVISION = True
    TORCHVISION_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    HAS_TORCHVISION = False
    TORCHVISION_IMPORT_ERROR = exc


MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Asymmetric DDP MNIST demo")
    parser.add_argument("--rank", type=int, required=True, choices=(0, 1))
    parser.add_argument("--trainer-rank", type=int, default=0, choices=(0, 1))
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29621)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--tol-loss", type=float, default=2e-4)
    parser.add_argument("--tol-param", type=float, default=1e-4)
    parser.add_argument("--data-dir", default="/tmp/mnist_data")
    parser.add_argument("--save-every-steps", type=int, default=200)
    parser.add_argument("--save-dir", default="/tmp/ddp_hetero_mnist_ckpt")
    return parser.parse_args()


def _configure_asymmetric_env(trainer_rank: int) -> None:
    os.environ.setdefault("TORCH_DDP_ASYMMETRIC_MODE", "1")
    os.environ.setdefault("TORCH_DDP_TRAINER_RANK", str(trainer_rank))
    os.environ.setdefault("TORCH_DDP_SKIP_ALLREDUCE", "1")
    os.environ.setdefault("TORCH_DDP_NON_TRAINER_FORWARD_ONLY", "1")
    os.environ.setdefault("TORCH_DDP_NON_TRAINER_BACKWARD", "error")
    os.environ.setdefault("TORCH_DDP_SYNC_INTERVAL", "1")
    os.environ.setdefault("TORCH_DDP_HETERO_PARAM_SYNC", "1")


def _all_gather_scalar(value: float, world_size: int, device: torch.device) -> list[float]:
    local = torch.tensor([value], dtype=torch.float64, device=device)
    gathered = [torch.zeros(1, dtype=torch.float64, device=device) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return [float(t.item()) for t in gathered]


def _param_checksum(model: torch.nn.Module) -> float:
    total = 0.0
    with torch.no_grad():
        for p in model.parameters():
            total += float(p.detach().float().sum().item())
    return total


class SmallCNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    urllib.request.urlretrieve(url, path)  # nosec B310


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise RuntimeError(f"Invalid image IDX magic in {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise RuntimeError(f"Invalid label IDX magic in {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.shape[0] != num:
        raise RuntimeError(f"Label count mismatch in {path}: {data.shape[0]} vs {num}")
    return data


class RawMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        if images.shape[0] != labels.shape[0]:
            raise RuntimeError("MNIST images/labels length mismatch")
        self.images = torch.from_numpy(images).to(torch.float32).div_(255.0).unsqueeze(1)
        self.labels = torch.from_numpy(labels.astype(np.int64))
        # Match the common MNIST normalization used by torchvision examples.
        self.images.sub_(0.1307).div_(0.3081)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


def build_mnist_datasets(data_dir: Path) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    data_dir.mkdir(parents=True, exist_ok=True)
    if HAS_TORCHVISION:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_set = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)
        return train_set, test_set

    # torchvision may be incompatible with custom-built torch.
    for fname in MNIST_FILES.values():
        _download(f"{MNIST_BASE_URL}{fname}", data_dir / fname)

    train_images = _read_idx_images(data_dir / MNIST_FILES["train_images"])
    train_labels = _read_idx_labels(data_dir / MNIST_FILES["train_labels"])
    test_images = _read_idx_images(data_dir / MNIST_FILES["test_images"])
    test_labels = _read_idx_labels(data_dir / MNIST_FILES["test_labels"])
    return RawMNISTDataset(train_images, train_labels), RawMNISTDataset(test_images, test_labels)


def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss.item())
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    model.train()
    return total_loss / max(1, total), correct / max(1, total)


def main() -> None:
    args = parse_args()
    _configure_asymmetric_env(args.trainer_rank)

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["RANK"] = str(args.rank)

    dist.init_process_group(backend="gloo", init_method="env://")
    rank = args.rank
    is_trainer = rank == args.trainer_rank

    if is_trainer:
        if not torch.cuda.is_available():
            raise RuntimeError("Trainer rank requires CUDA, but no GPU is visible.")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print(
            f"[rank{rank}] gpu_check cuda_available={torch.cuda.is_available()} "
            f"device_count={torch.cuda.device_count()} current_device={torch.cuda.current_device()}",
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

    data_dir = Path(args.data_dir)
    if rank == 0 and (not HAS_TORCHVISION):
        print(
            "[rank0] torchvision import failed; using raw MNIST IDX fallback. "
            f"reason={TORCHVISION_IMPORT_ERROR!r}",
            flush=True,
        )
    train_set, test_set = build_mnist_datasets(data_dir)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = SmallCNN().to(device)
    ddp = DDP(model, device_ids=[0], output_device=0) if device.type == "cuda" else DDP(model)
    optimizer = torch.optim.SGD(ddp.parameters(), lr=args.lr, momentum=args.momentum) if is_trainer else None

    cfg = ddp.get_asymmetric_mode_config()
    print(f"[rank{rank}] asymmetric_cfg={cfg}", flush=True)
    dist.barrier()

    ckpt_dir = Path(args.save_dir) if args.save_every_steps > 0 else None
    if ckpt_dir is not None and (not is_trainer):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[rank{rank}] checkpoint_dir={ckpt_dir} save_every_steps={args.save_every_steps}", flush=True)

    start = time.time()
    step = 0

    for _epoch in range(args.epochs):
        for x_cpu, y_cpu in train_loader:
            if step >= args.max_steps:
                break

            if is_trainer:
                batch_x = x_cpu.contiguous()
                batch_y = y_cpu.contiguous()
            else:
                batch_x = torch.zeros(args.batch_size, 1, 28, 28, dtype=torch.float32)
                batch_y = torch.zeros(args.batch_size, dtype=torch.int64)

            dist.broadcast(batch_x, src=args.trainer_rank)
            dist.broadcast(batch_y, src=args.trainer_rank)

            x = batch_x.to(device, non_blocking=True)
            y = batch_y.to(device, non_blocking=True)

            if is_trainer:
                optimizer.zero_grad(set_to_none=True)

            logits = ddp(x)
            loss = torch.nn.functional.cross_entropy(logits, y)

            # Compare same-batch same-model pre-step loss across ranks.
            loss_values = _all_gather_scalar(float(loss.item()), args.world_size, device)
            loss_gap = max(loss_values) - min(loss_values)
            if loss_gap > args.tol_loss:
                raise RuntimeError(
                    f"Loss mismatch too large at step {step}: values={loss_values}, tol={args.tol_loss}"
                )

            if is_trainer:
                loss.backward()
            ddp.trainer_step(optimizer if is_trainer else None)

            if step % max(1, args.log_interval) == 0:
                param_values = _all_gather_scalar(_param_checksum(ddp.module), args.world_size, device)
                param_gap = max(param_values) - min(param_values)
                if param_gap > args.tol_param:
                    raise RuntimeError(
                        f"Param checksum mismatch at step {step}: values={param_values}, tol={args.tol_param}"
                    )
                elapsed = time.time() - start
                if rank == 0:
                    print(
                        f"step={step} loss_pair={loss_values} loss_gap={loss_gap:.3e} "
                        f"param_pair={param_values} param_gap={param_gap:.3e} elapsed={elapsed:.1f}s",
                        flush=True,
                    )

            if (step > 0) and (step % max(1, args.eval_interval) == 0):
                eval_loss, eval_acc = evaluate(ddp.module, eval_loader, device)
                eval_loss_values = _all_gather_scalar(eval_loss, args.world_size, device)
                eval_acc_values = _all_gather_scalar(eval_acc, args.world_size, device)
                if rank == 0:
                    print(
                        f"[eval step={step}] loss_pair={eval_loss_values} acc_pair={eval_acc_values}",
                        flush=True,
                    )

            if (not is_trainer) and ckpt_dir is not None and ((step + 1) % args.save_every_steps == 0):
                ckpt_path = ckpt_dir / f"follower_mnist_step_{step + 1}.pt"
                torch.save(
                    {
                        "model": ddp.module.state_dict(),
                        "step": step + 1,
                        "saved_at": time.time(),
                    },
                    ckpt_path,
                )
                print(f"[rank{rank}] checkpoint_saved={ckpt_path}", flush=True)

            dist.barrier()
            step += 1

        if step >= args.max_steps:
            break

    final_loss, final_acc = evaluate(ddp.module, eval_loader, device)
    final_loss_values = _all_gather_scalar(final_loss, args.world_size, device)
    final_acc_values = _all_gather_scalar(final_acc, args.world_size, device)

    if rank == 0:
        print(
            f"[final] steps={step} test_loss_pair={final_loss_values} test_acc_pair={final_acc_values}",
            flush=True,
        )
        print(f"Hetero MNIST demo: PASS (elapsed={time.time() - start:.1f}s)", flush=True)
    elif ckpt_dir is not None:
        saved = sorted(ckpt_dir.glob("follower_mnist_step_*.pt"))
        print(
            f"[rank{rank}] checkpoint_summary saved_files={len(saved)} "
            f"latest={(saved[-1] if saved else 'none')}",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

