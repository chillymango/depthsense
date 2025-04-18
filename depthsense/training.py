import cv2
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

from dpt import DepthSense
from util import common
from util.loss import Loss


common.set_random_seeds()

PRETRAINING = False


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
            array = np.nan_to_num(array, nan=20.0, posinf=20.0, neginf=0.0)
            array = np.clip(array, 0, 20.0)

        # needs to be divisible by 14
        resized = cv2.resize(array, (252, 196), interpolation=cv2.INTER_AREA)

        tensor = torch.from_numpy(resized).float()

        return tensor

    def __getitem__(self, i: int) -> tuple[int, Tensor, Tensor, Tensor]:
        return i, self._read_array(i, "frame"), self._read_array(
            i,
            "depth"), self._read_array(i, "normal")

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

    writer = SummaryWriter(log_dir="runs/experiment_03")

    betas: tuple[float, float] = 0.9, 0.999
    dataset_name: Dataset = "hypersim"
    decay: float = 5e-2
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device {device}")

    # Data splitting.
    dataset: Dataset = DepthSenseDataset(f"/data/{dataset_name}")
    data_size: int = len(dataset)
    train_size: int = int(0.8 * data_size)
    val_size: int = int(0.1 * data_size)
    test_size: int = len(dataset) - train_size - val_size
    
    # Reduce boundary mixing of scenes by loading sequentially
    all_indices = list(range(data_size))
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
    print(f"data_size: {data_size}, train_size: {train_size}, val_size: {val_size}, test_size: {test_size}")

    encoder: str = "vitg"
    criterion: Module = Loss()
    if PRETRAINING:
        print(f"Pretraining, freezing backbone")
        epochs: int = 3
        batch_size: int = 128
        eps: float = 1e-8
        features: int = 128
        lr: float = 1e-4

        start_epoch = 0
        start_step = 0

        # Model initialization.
        model_name: str = model_path.replace("{}", dataset_name)
        model: DepthSense = DepthSense(encoder, features).to(device).to(memory_format=torch.channels_last)
        # TODO: look into using registers
        #pretrained = torch.load('dinov2_vits14_reg4_pretrain.pth')

        # load pretrained weights
        #pretrained = torch.load('dinov2_vitg14_pretrain.pth')
        #model.pretrained.load_state_dict(pretrained)
        
        # load previous checkpoint for another few epochs
        checkpoint = torch.load('checkpoint_epoch_1.pt')
        model.load_state_dict(checkpoint['model_state_dict'])

        for param in model.pretrained.parameters():
            param.requires_grad = False

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer: Optimizer = AdamW(trainable_params, lr, betas, eps, decay)

        # load previous checkpoint
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print(f"Fine tuning")
        epochs: int = 75  # this is on top of where we start
        batch_size: int = 64
        eps: float = 1e-8
        features: int = 128
        lr: float = 1e-6

        # Model initialization.
        model_name: str = model_path.replace("{}", dataset_name)
        model: DepthSense = DepthSense(encoder, features).to(device).to(memory_format=torch.channels_last)

        # Load from checkpoint
        checkpoint = torch.load('prev-checkpoint.pt')
        start_epoch = checkpoint['epoch'] + 1
        start_step = 15050

        model.load_state_dict(checkpoint['model_state_dict'])

        # Uncomment below to start from fine-tune base
        #pretrain = torch.load('vitg-fine-tune-base.pth')
        #model.load_state_dict(pretrain)
        params = [{"params": model.parameters()},
                {"params": [criterion.log_sigma_d, criterion.log_sigma_n]}]
        optimizer: Optimizer = AdamW(params, lr, betas, eps, decay)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Training.
    model.train()

    shuffle = True
    train_loader: DataLoader = DataLoader(
        train_set,
        batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=8,
        prefetch_factor=4,
    )
    val_loader: DataLoader = DataLoader(
        val_set,
        batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=8,
        prefetch_factor=4
    )
    max_iters: int = len(train_loader)

    t = start_step + 1
    for e in range(start_epoch, start_epoch + epochs):
        print(f"Epoch: {e}/{start_epoch + epochs}...")
        model.train()

        rl: float = 0.0
        rdl: float = 0.0
        rnl: float = 0.0
        for i, (j, x, z_gt, n_gt) in enumerate(train_loader):
            torch.cuda.empty_cache()

            # Move to appropriate device.
            x = x.to(device).permute(0, 3, 1, 2)

            z_gt = z_gt.to(device)
            z_gt = torch.nan_to_num(z_gt, nan=20.0, posinf=20.0, neginf=0.0)
            z_gt = torch.clamp(z_gt, min=0.0, max=20.0)
            n_gt = n_gt.to(device).permute(
                0,
                3,
                1,
                2)  # (B, H, W, C) → (B, C, H, W)

            # Forward pass.
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
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
            t += 1
            if (i + 1) % 10 == 0 or i == 0:
                avg_loss = rl / (10 if i > 0 else 1)
                avg_n_loss = rnl / (10 if i > 0 else 1)
                avg_d_loss = rdl / (10 if i > 0 else 1)
                print(f"Epoch {e}, iter {i + 1}/{max_iters} — Loss: {avg_loss:.4f}, Loss Norm: {avg_n_loss:.4f}, Loss Depth: {avg_d_loss:.4f}")
                writer.add_scalar("loss/normal", avg_n_loss, t)
                writer.add_scalar("loss/depth", avg_d_loss, t)
                writer.add_scalar("loss/total", avg_loss, t)
                writer.add_scalar("param/log_sigma_d", criterion.log_sigma_d.item(), t)
                writer.add_scalar("param/log_sigma_n", criterion.log_sigma_n.item(), t)

                # reset running counters, don't aggregate over entire epoch
                rl = 0.0
                rnl = 0.0
                rdl = 0.0

            if (i + 1) % 50 == 0 or i == 0:
                # render label and prediction
                n_pred_vis = visualize_normals(n_hat[:4])
                n_gt_vis = visualize_normals(n_gt[:4])
                z_pred_vis = visualize_depth(z_hat[:4])
                z_gt_vis = visualize_depth(z_gt[:4])

                n_pred_grid = torchvision.utils.make_grid(n_pred_vis)
                n_gt_grid = torchvision.utils.make_grid(n_gt_vis)
                z_pred_grid = torchvision.utils.make_grid(z_pred_vis)
                z_gt_grid = torchvision.utils.make_grid(z_gt_vis)

                writer.add_image("Normals/Prediction", n_pred_grid, t)
                writer.add_image("Normals/GroundTruth", n_gt_grid, t)
                writer.add_image("Depth/Prediction", z_pred_grid, t)
                writer.add_image("Depth/GroundTruth", z_gt_grid, t)

        model.eval()
        val_rl = 0
        val_rdl = 0
        val_rnl = 0
        with torch.no_grad():
            count = 0
            for j, (idx, x_val, z_gt_val, n_gt_val) in enumerate(val_loader):
                count += 1
                if count > 100:
                    break
                x_val = x_val.to(device).permute(0, 3, 1, 2)
                z_gt_val = torch.nan_to_num(
                    z_gt_val.to(device), nan=20.0, posinf=20.0, neginf=0.0
                ).clamp(0, 20.0)
                n_gt_val = n_gt_val.to(device).permute(0, 3, 1, 2)

                z_hat_val, n_hat_val = model(x_val)
                d_loss, n_loss = criterion(z_hat_val, z_gt_val, n_hat_val, n_gt_val)
                total_val = d_loss + n_loss
                val_rl += total_val.item()
                val_rdl += d_loss.item()
                val_rnl += n_loss.item()

        num_batches = count
        avg_val = val_rl / num_batches
        avg_val_d = val_rdl / num_batches
        avg_val_n = val_rnl / num_batches
        print(
            f"Validation — Loss: {avg_val:.4f}, Depth: {avg_val_d:.4f}, Norm: {avg_val_n:.4f}"
        )
        writer.add_scalar("val/total", avg_val, e)
        writer.add_scalar("val/depth", avg_val_d, e)
        writer.add_scalar("val/normal", avg_val_n, e)

        # render the most recent
        n_pred_vis = visualize_normals(n_hat_val[:4])
        n_gt_vis = visualize_normals(n_gt_val[:4])
        z_pred_vis = visualize_depth(z_hat_val[:4])
        z_gt_vis = visualize_depth(z_gt_val[:4])

        n_pred_grid = torchvision.utils.make_grid(n_pred_vis)
        n_gt_grid = torchvision.utils.make_grid(n_gt_vis)
        z_pred_grid = torchvision.utils.make_grid(z_pred_vis)
        z_gt_grid = torchvision.utils.make_grid(z_gt_vis)

        writer.add_image("Validation Normals/Prediction", n_pred_grid, e)
        writer.add_image("Validation Normals/GroundTruth", n_gt_grid, e)
        writer.add_image("Validation Depth/Prediction", z_pred_grid, e)
        writer.add_image("Validation Depth/GroundTruth", z_gt_grid, e)

        print(f'Saving checkpoint for epoch {e}')
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'checkpoint_epoch_{e}.pt')

    # Save current model.
    if PRETRAINING:
        print(f"Saving pretrained decoder-only to `pretrain.pth`, rename to `fine-tune-base.pth` to use")
        torch.save(model.state_dict(), "pretrain.pth")
    else:
        print(f"Saving final model")
        torch.save(model.state_dict(), f"{model_name}.pth")
