import argparse
import random
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from dpt import DepthSense
from util import common
from util.loss import Loss


common.set_random_seeds()


class DepthNormalDataset(Dataset):
    """
    Utility and wrapper for loading datasets.

    Loads images from a directory and resizes them to a specified size.
    """

    def __init__(self, root: Path, size: tuple[int, int]):
        self.paths: list[Path] = [d for d in root.iterdir() if d.is_dir()]
        self.img_size: tuple[int, int] = size

    def _load(self, path: Path, name: str) -> np.ndarray:
        x = np.load(path / f"{name}.npy").astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=1.0e4, neginf=0.0)
        if x.ndim == 2:
            x = x[..., None]
        x = (
            common.normalize_frame(x)
            if name == "frame"
            else common.normalize_depths(x) if name == "depth" else x
        )
        x = torch.from_numpy(common.resize(x, self.img_size))
        return common.to_bchw(x).float()

    def __getitem__(self, idx):
        path: Path = self.paths[idx]
        image: Tensor = self._load(path, "frame")
        depth: Tensor = self._load(path, "depth")
        normal: Tensor = self._load(path, "normal")
        return image, depth, normal

    def __len__(self):
        return len(self.paths)


def visualize(t: Tensor) -> Tensor:
    t = t.clone()
    t[~torch.isfinite(t)] = 0.0
    return (t + 1.0) / 2.0 if t.shape[1] == 3 else t


def train(args: Namespace):
    print("Arguments:", args)
    # Shorthand properties from args.
    batch_size: int = args.batch_size
    betas: tuple[float, float] = args.betas
    checkpoint: str = args.checkpoint
    ckpt_dir: Path = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dataset: str = args.dataset
    decay: float = args.decay
    encoder: str = args.encoder
    epochs: int = args.epochs
    eps: float = args.eps
    features: int = args.features
    logdir: str = args.logdir
    lr: float = args.lr
    shuffle: bool = args.shuffle
    size: tuple[int, int] = args.size
    # Additional properties.
    device: str = common.get_device()
    writer: SummaryWriter = SummaryWriter(logdir)

    # Model initialization.
    model: Module = DepthSense(encoder, features).to(device)
    model = model.to(memory__format=torch.channels_last)

    criterion: Module = Loss()
    optimizer: Optimizer = AdamW(model.parameters(), lr, betas, eps, decay)

    print(f"Using model {encoder} running on device {device}.")

    # Dataset splitting.
    dataset: Dataset = DepthNormalDataset(dataset, size)
    data_size: int = len(dataset)
    train_size: int = int(0.8 * data_size)
    val_size: int = int(0.1 * len(dataset))
    test_size: int = data_size - train_size - val_size
    # Reduce boundary mixing of scenes by loading sequentially
    all_indices = list(range(data_size))
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size : train_size + val_size]
    test_indices = all_indices[train_size + val_size :]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    print(f"data_size: {data_size}", end=", ")
    print(f"train_size: {train_size}", end=", ")
    print(f"val_size: {val_size}", end=", ")
    print(f"test_size: {test_size}")

    loader_properties: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 4,
    }
    train_loader: DataLoader = DataLoader(train_set, **loader_properties)
    val_loader: DataLoader = DataLoader(val_set, **loader_properties)

    if checkpoint and Path(checkpoint).exists():
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch: int = ckpt["epoch"] + 1
    else:
        start_epoch: int = 0

    for epoch in range(start_epoch, epochs):
        print(f"Epoch: {epoch}/{start_epoch + epochs}...")

        # Training.
        model.train()
        total_loss, depth_loss, normal_loss = 0.0, 0.0, 0.0
        for i, (x, d_gt, n_gt) in enumerate(train_loader):
            x = common.to_bchw(x.to(device))
            d_gt = common.normalize_depths(d_gt.to(device))
            n_gt = common.to_bchw(n_gt.to(device))

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                z_hat, n_hat = model(x)

            z_loss, n_loss = criterion(z_hat, d_gt, n_hat, n_gt)
            if torch.isnan(z_loss).any() or torch.isnan(n_loss).any():
                print("Encountered nan in model output.", end=" ")
                print("Training will fail from here.")
                breakpoint()
                raise ValueError("nan in model output")

            loss: Tensor = z_loss + n_loss
            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics recollection and display.
            total_loss += loss.item()
            depth_loss += z_loss.item()
            normal_loss += n_loss.item()

            if i % 10 == 0 or i == 0:
                avg_loss: float = total_loss / (10 if i > 0 else 1)
                avg_d_loss: float = depth_loss / (10 if i > 0 else 1)
                avg_n_loss: float = normal_loss / (10 if i > 0 else 1)
                print(
                    f"Epoch {epoch}, iter {i + 1}/{max_iters} — Loss: {avg_loss:.4f}, Loss Norm: {avg_n_loss:.4f}, Loss Depth: {avg_d_loss:.4f}"
                )
                writer.add_scalar("loss/normal", avg_n_loss, t)
                writer.add_scalar("loss/depth", avg_d_loss, t)
                writer.add_scalar("loss/total", avg_loss, t)
                writer.add_scalar("param/log_sigma_d", criterion.log_sigma_d.item(), t)
                writer.add_scalar("param/log_sigma_n", criterion.log_sigma_n.item(), t)

        writer.add_scalar("train/total_loss", total_loss, epoch)
        writer.add_scalar("train/depth_loss", depth_loss, epoch)
        writer.add_scalar("train/normal_loss", normal_loss, epoch)

        # Validation.
        model.eval()
        val_loss, val_depth, val_normal = 0.0, 0.0, 0.0
        with torch.no_grad():
            for x, d_gt, n_gt in val_loader:
                x = common.to_bchw(x.to(device))
                d_gt = common.normalize_depths(d_gt.to(device))
                n_gt = common.to_bchw(n_gt.to(device))

                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    z_hat, n_hat = model(x)

                z_loss, n_loss = criterion(z_hat, d_gt, n_hat, n_gt)
                val_loss += (z_loss + n_loss).item()
                val_depth += z_loss.item()
                val_normal += n_loss.item()

        writer.add_scalar("val/total_loss", val_loss, epoch)
        writer.add_scalar("val/depth_loss", val_depth, epoch)
        writer.add_scalar("val/normal_loss", val_normal, epoch)

        idx = random.randint(0, x.size(0) - 1)
        grid = lambda t: make_grid(
            visualize(t[idx : idx + 1].cpu()),
            normalize=True,
        )
        writer.add_image("image", grid(x), epoch)
        writer.add_image("depth_gt", grid(d_gt), epoch)
        writer.add_image("normal_gt", grid(n_gt), epoch)
        writer.add_image("depth_pred", grid(z_hat), epoch)
        writer.add_image("normal_pred", grid(n_hat), epoch)

        print(f"Saving checkpoint for epoch {epoch}...")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            Path(ckpt_dir) / f"student_epoch_{epoch}.pt",
        )
        print("Checkpoint saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--decay", type=float, default=1e-2)
    parser.add_argument("--encoder", type=str, default="vits")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--logdir", type=str, default="runs/student")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--size", type=int, nargs=2, default=(252, 196))
    train(parser.parse_args())
