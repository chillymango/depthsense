import cv2
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

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

        # if reading RGB frame image, apply simple tonemap
        if name == "frame":
            array = np.nan_to_num(array, nan=0.0, posinf=1e4, neginf=0)
            array = np.clip(array, 0, 1e4)
            array = array / (1.0 + array)
        # if reading depth map, clip to (0, 80)
        elif name == "depth":
            array = np.nan_to_num(array, nan=80.0, posinf=80.0, neginf=0.0)
            array = np.clip(array, 0, 80.0)

        resized = cv2.resize(array, (array.shape[1] // 2, array.shape[0] // 2), interpolation=cv2.INTER_AREA)

        tensor = torch.from_numpy(resized).float()

        return tensor

    def __getitem__(self, i: int) -> tuple[int, Tensor, Tensor, Tensor]:
        return i, self._read_array(i, "frame"), self._read_array(i, "depth"), self._read_array(i, "normal")

    def __len__(self) -> int:
        return len(self.directories)


def visualize_normals(normals: torch.Tensor) -> torch.Tensor:
    vis = (normals + 1.0) / 2.0
    vis[~torch.isfinite(vis)] = 0.0
    return vis.clamp(0, 1)



def visualize_depth(depth: torch.Tensor) -> torch.Tensor:
    depth_vis = []
    for d in depth:
        mask = torch.isfinite(d)
        if mask.any():
            d_clean = d.clone()
            d_clean[~mask] = 0.0
            d_min = d_clean[mask].min()
            d_max = d_clean[mask].max()
            norm = (d_clean - d_min) / (d_max - d_min + 1e-8)
        else:
            norm = torch.zeros_like(d)
        depth_vis.append(norm.clamp(0, 1))
    return torch.stack(depth_vis).unsqueeze(1)



if __name__ == "__main__":
    # Parameters and hyperparameters used for training.
    description: str = "DepthSense for Metric Depth and Normal Estimation"
    model_path: str = "models/teacher_{}.pth"
    
    writer = SummaryWriter(log_dir="runs/experiment_01")

    batch_size: int = 8
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
    lr: float = 2e-4
    
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
    criterion: Module = Loss()
    params = [{"params": model.parameters()}, {"params": [criterion.log_sigma_d, criterion.log_sigma_n]}]
    optimizer: Optimizer = AdamW(params, lr, betas, eps, decay)

    checkpoint = None
    start_epoch = 0
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # Training.
    model.train()
    
    # DEBUGGING: DETERMINISTIC LOAD ORDER
    shuffle = False
    train_loader: DataLoader = DataLoader(train_set, batch_size, shuffle=shuffle)
    max_iters: int = len(train_loader)

    for e in range(start_epoch, epochs):
        print(f"Epoch: {e}/{epochs}...")
        rl: float = 0.0
        rdl: float = 0.0
        rnl: float = 0.0
        for i, (j, x, z_gt, n_gt) in enumerate(train_loader):
            # DEBUG: overtrain on single sample

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Move to appropriate device.
            x = x.to(device).permute(0, 3, 1, 2)

            z_gt = z_gt.to(device)
            z_gt = torch.nan_to_num(z_gt, nan=80.0, posinf=80.0, neginf=0.0)
            z_gt = torch.clamp(z_gt, min=0.0, max=80.0)
            n_gt = n_gt.to(device).permute(0, 3, 1, 2) # (B, H, W, C) → (B, C, H, W)

            # TODO: see if we can autocast for memory improvement?
            #with torch.autocast(device_type=device, dtype=torch.bfloat16):

            # Forward pass.
            z_hat, n_hat = model(x)
            depth_loss, norm_loss = criterion(z_hat, z_gt, n_hat, n_gt)
            if torch.isnan(depth_loss).any() or torch.isnan(norm_loss).any():
                print("Encountered nan in model output. Training will fail from here.")
                breakpoint()
                raise ValueError("nan in model output")

            loss = depth_loss + norm_loss
            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics recollection and display.
            rl += loss.item()
            rnl += norm_loss.item()
            rdl += depth_loss.item()
            if (i + 1) % 10 == 0 or i == 0:
                avg_loss = rl / (i + 1)
                avg_n_loss = rnl / (i + 1)
                avg_d_loss = rdl / (i + 1)
                print(f"Epoch {e}, iter {i + 1}/{max_iters} — Loss: {avg_loss:.4f}, Loss Norm: {avg_n_loss:.4f}, Loss Depth: {avg_d_loss:.4f}")
                writer.add_scalar("loss/normal", avg_n_loss, i)
                writer.add_scalar("loss/depth", avg_d_loss, i)
                writer.add_scalar("loss/total", avg_loss, i)
                writer.add_scalar("param/log_sigma_d", criterion.log_sigma_d.item(), i)
                writer.add_scalar("param/log_smg_an", criterion.log_sigma_n.item(), i)

            if (i + 1) % 100 == 0 or i == 0:
                # TODO: compute validation loss
                # render label and prediction
                n_pred_vis = visualize_normals(n_hat[:4])
                n_gt_vis = visualize_normals(n_gt[:4])
                z_pred_vis = visualize_depth(z_hat[:4])
                z_gt_vis = visualize_depth(z_gt[:4])
        
                n_pred_grid = torchvision.utils.make_grid(n_pred_vis)
                n_gt_grid = torchvision.utils.make_grid(n_gt_vis)
                z_pred_grid = torchvision.utils.make_grid(z_pred_vis)
                z_gt_grid = torchvision.utils.make_grid(z_gt_vis)
        
                writer.add_image("Normals/Prediction", n_pred_grid, i)
                writer.add_image("Normals/GroundTruth", n_gt_grid, i)
                writer.add_image("Depth/Prediction", z_pred_grid, i)
                writer.add_image("Depth/GroundTruth", z_gt_grid, i)

        print(f'Saving checkpoint for epoch {i}')
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f'checkpoint_epoch_{i}.pt')

    # Save current model.
    torch.save(model.state_dict(), model_name)
