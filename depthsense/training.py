import cv2
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from dpt import DepthSense
from util.loss import Loss

random.seed(7643)
np.random.seed(7643)
torch.manual_seed(7643)
torch.random.manual_seed(7643)
torch.cuda.manual_seed(7643)


class DepthSenseDataset(Dataset):
    """
    Utility and wrapper for loading datasets.
    """

    def __init__(self, root_dir: str):
        self.root_dir: Path = Path(root_dir)
        self.directories: list[str] = [
            d for d in self.root_dir.iterdir() if d.is_dir()
        ]

    # TODO: make scaling configurable
    def _read_array(self, i: int, name: str):
        cur: Path = self.directories[i]
        path = f"{cur}/{name}.npy"
        array = np.load(path).astype('float32')
        if array.ndim == 2:
            array = array[..., np.newaxis]

        resized = cv2.resize(array, (array.shape[1] // 2, array.shape[0] // 2), interpolation=cv2.INTER_AREA)

        tensor = torch.from_numpy(resized).float()

        return tensor


    def __getitem__(self, i: int) -> tuple[int, Tensor, Tensor, Tensor]:
        return i, self._read_array(i, "frame"), self._read_array(i, "depth"), self._read_array(i, "normal")

    def __len__(self) -> int:
        return len(self.directories)


if __name__ == "__main__":
    # Parameters and hyperparameters used for training.
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
    criterion: Module = Loss(depth_scale=0.03)
    optimizer: Optimizer = AdamW(model.parameters(), lr, betas, eps, decay)

    # Training.
    model.train()
    
    train_loader: DataLoader = DataLoader(train_set, batch_size, shuffle=True)
    max_iters: int = len(train_loader)

    for e in range(epochs):
        print(f"Epoch: {e}/{epochs}...")
        rl: float = 0.0
        rdl: float = 0.0
        rnl: float = 0.0
        for i, (j, x, z_gt, n_gt) in enumerate(train_loader):
            # debug by overfitting to one sample
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Move to appropriate device.
            x = x.to(device).permute(0, 3, 1, 2)
            # TODO: maybe something smarter for nan interpolation, but for now we just fill
            if torch.isnan(x).sum() > 10000:
                print(f"Warning: image {j} has {torch.isnan(x).sum()} nan")
            x = torch.nan_to_num(x, 0.0)

            z_gt = z_gt.to(device)
            z_gt = torch.clamp(z_gt, min=0.0, max=80.0)
            n_gt = n_gt.to(device).permute(0, 3, 1, 2) # (B, H, W, C) → (B, C, H, W)
        
            # TODO: see if we can autocast for memory improvement?
            #with torch.autocast(device_type=device, dtype=torch.bfloat16):

            # Forward pass.
            z_hat, n_hat = model(x)
            depth_loss, norm_loss = criterion(z_hat, z_gt, n_hat, n_gt)
            if torch.isnan(depth_loss).any() or torch.isnan(norm_loss).any():
                breakpoint()
            loss = depth_loss + norm_loss
            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics recollection and display.
            rl += loss.item()
            rnl += norm_loss.item()
            rdl += depth_loss.item()
            if (i + 1) % 50 == 0 or i == 0:
                avg_loss = rl / (i + 1)
                avg_n_loss = rnl / (i + 1)
                avg_d_loss = rdl / (i + 1)
                print(f"Epoch {e}, iter {i + 1}/{max_iters} — Loss: {avg_loss:.4f}, Loss Norm: {avg_n_loss:.4f}, Loss Depth: {avg_d_loss:.4f}")
            
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            #print(f"   [CUDA] allocated: {allocated:.2f} GB, peak: {peak:.2f} GB")
            
    # Save current model.
    torch.save(model.state_dict(), model_name)
