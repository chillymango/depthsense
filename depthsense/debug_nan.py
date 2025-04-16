import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from dpt import DepthSense
from util.loss import GeoNetLoss

torch.random.manual_seed(7643)


class DepthSenseDataset(Dataset):
    """
    Utility and wrapper for loading datasets.
    """

    def __init__(self, root_dir: str):
        self.root_dir: Path = Path(root_dir)
        self.directories: list[str] = [
            d for d in self.root_dir.iterdir() if d.is_dir()
        ]

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, Tensor]:
        cur: Path = self.directories[i]
        image: Tensor = torch.from_numpy(np.load(f"{cur}/frame.npy")).float()
        depth: Tensor = torch.from_numpy(np.load(f"{cur}/depth.npy")).float()
        normal: Tensor = torch.from_numpy(np.load(f"{cur}/normal.npy")).float()
        return image, depth, normal

    def __len__(self) -> int:
        return len(self.directories)


def main():
    description: str = "DepthSense for Metric Depth and Normal Estimation"
    model_path: str = "models/teacher_{}.pth"
    
    batch_size: int = 4
    betas: tuple[float, float] = 0.9, 0.999
    dataset_name: Dataset = "hypersim"
    decay: float = 1e-2
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device {device}")
    encoder: str = "vits"
    epochs: int = 10
    eps: float = 1e-8
    features: int = 128
    lr: float = 1e-4
    
    # Data splitting.
    dataset: Dataset = DepthSenseDataset(f"/data/{dataset_name}")
    data_size: int = len(dataset)
    train_size: int = int(0.8 * data_size)
    val_size: int = int(0.1 * data_size)
    test_size: int = len(dataset) - train_size - val_size
    train_set, val_set, test_set = data.random_split(dataset, [train_size, val_size, test_size])
    print(f"data_size: {data_size}, train_size: {train_size}, val_size: {val_size}, test_size: {test_size}")
    
    # Model initialization.
    model_name: str = model_path.replace("{}", dataset_name)
    model: DepthSense = DepthSense(encoder, features).to(device)
    criterion: Module = GeoNetLoss()
    optimizer: Optimizer = AdamW(model.parameters(), lr, betas, eps, decay)

    model.train()
    
    train_loader: DataLoader = DataLoader(train_set, batch_size, shuffle=True)

    for i, (x_i, z_gt, n_gt) in enumerate(train_loader):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
        # Move to appropriate device.
        x_i = x_i.to(device).permute(0, 3, 1, 2) # (B, H, W, C) → (B, C, H, W)
        z_gt = z_gt.to(device)
        n_gt = n_gt.to(device).permute(0, 3, 1, 2) # (B, H, W, C) → (B, C, H, W)
        #print(f"   [input] x: {x.shape}, z_gt: {z_gt.shape}, n_gt: {n_gt.shape}")
        print(f"input: {x_i[:, 0, 0, 0]}")
        print(f"depth: {z_gt[:, 0, 0]}")
        print(f"normal: {n_gt[:, 0, 0, 0]}")


if __name__ == "__main__":
    main()
