import argparse
import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from dpt import DepthSense
from training import DepthSenseDataset
from util import common

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Depth-Anything-V2')))
from depth_anything_v2.dpt import DepthAnythingV2



@torch.no_grad()
def run_inference(model: torch.nn.Module, x: torch.Tensor, device: str, max_depth: float, model_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference on a batch of images and return predicted depth and normal maps.

    For DepthAnythingV2, normals are estimated from depth.

    Args:
        model (torch.nn.Module): Depth/normal estimation model.
        x (torch.Tensor): Batch of input images (B, H, W, C).
        device (str): Device to run model on.
        max_depth (float): Maximum depth clamp value.
        model_type (str): model_type

    Returns:
        tuple: (B, 1, H, W) depth and (B, 3, H, W) normalized surface normals.
    """
    model.eval()

    x = x.to(device).permute(0, 3, 1, 2)

    if model_type == 'depthanythingv2':
        z_pred = model(x)  # only depth
        n_pred = depth_to_normal(z_pred)
    else:
        z_pred, n_pred = model(x)

    z_pred = z_pred.clamp(min=1e-3, max=max_depth)
    n_pred = F.normalize(n_pred, p=2, dim=1)

    return z_pred, n_pred

@torch.no_grad()
def depth_to_normal(depth: torch.Tensor) -> torch.Tensor:
    """
    Estimate surface normals from predicted depth using central differences.

    Args:
        depth (torch.Tensor): (B, 1, H, W) predicted depth.

    Returns:
        torch.Tensor: (B, 3, H, W) normalized surface normals.
    """
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)

    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"Expected input of shape (B, 1, H, W), got {depth.shape}")

    dzdx = F.pad(depth[:, :, :, 2:] - depth[:, :, :, :-2], (1, 1, 0, 0), mode='replicate') / 2.0
    dzdy = F.pad(depth[:, :, 2:, :] - depth[:, :, :-2, :], (0, 0, 1, 1, 0, 0), mode='replicate') / 2.0

    normal_x = -dzdx
    normal_y = -dzdy
    normal_z = torch.ones_like(depth)

    normals = torch.cat([normal_x, normal_y, normal_z], dim=1)
    normals = F.normalize(normals, p=2, dim=1)
    return normals

@torch.no_grad()
def run_inference_batch(model: torch.nn.Module, dataloader: DataLoader, device: str, max_depth: float, output_dir: Path, model_type: str) -> None:
    """
    Run inference across dataset and save depth and normals as .npy and .png.

    Args:
        model (torch.nn.Module): Model to use for inference.
        dataloader (DataLoader): Torch dataloader for dataset.
        device (str): Device identifier.
        max_depth (float): Max depth range for clamping.
        output_dir (Path): Directory to save results.
        model_type (str): Type of model used (e.g., 'depthanythingv2' or 'depthsense').
    """
    model.eval()

    # subdirectory 'model_type' under output_dir
    output_dir = output_dir / model_type

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (_i, x, _depth, _normal) in enumerate(tqdm(dataloader, desc="Inferencing")):
        z_pred, n_pred = run_inference(model, x, device, max_depth, model_type)

        if model_type == 'depthanythingv2':
            n_pred = depth_to_normal(z_pred)
            # print(f"z_pred.shape: {z_pred.shape}")
            # print(f"n_pred.shape: {n_pred.shape}")

        for b in range(z_pred.shape[0]):
            depth_slice = z_pred[b]  # shape could be (1, H, W) or (H, W)
            if depth_slice.ndim == 3:  # (1, H, W)
                depth_np = depth_slice[0].cpu().numpy()
            else:  # (H, W)
                depth_np = depth_slice.cpu().numpy()

            normal_np = n_pred[b].permute(1, 2, 0).cpu().numpy()
            frame_np = x[b].cpu().numpy()  # x is still (B, H, W, 3)
            index = idx * dataloader.batch_size + b

            # print(f"depth_np.shape: {depth_np.shape}")
            # print(f"normal_np.shape: {normal_np.shape}")

            np.save(output_dir / f"depth_{index:06d}.npy", depth_np)
            np.save(output_dir / f"normal_{index:06d}.npy", normal_np)

            # Images
            depth_img = (255 * (depth_np / max_depth)).clip(0, 255).astype(np.uint8)
            normal_img = ((normal_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            frame_img = (frame_np * 255).clip(0, 255).astype(np.uint8)

            cv2.imwrite(str(output_dir / f"depth_{index:06d}.png"), depth_img)
            cv2.imwrite(str(output_dir / f"normal_{index:06d}.png"), cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(output_dir / f"frame_{index:06d}.png"), cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: str, max_depth: float, model_type: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate depth and normal accuracy metrics over a dataset.

    Args:
        model (torch.nn.Module): Model to evaluate.
        dataloader (DataLoader): Test dataloader.
        device (str): Evaluation device.
        max_depth (float): Clamp value for depth predictions.
        model_type (str): Type of model used (e.g., 'depthanythingv2' or 'depthsense').

    Returns:
        tuple: depth_metrics (np.ndarray), normal_metrics (np.ndarray)
    """
    model.eval()

    depth_metrics: list[list[float]] = []
    normal_metrics: list[list[float]] = []

    for _, x, z_gt, n_gt in tqdm(dataloader, desc="Evaluating"):
        z_gt: torch.Tensor = z_gt.unsqueeze(1).to(device)
        n_gt: torch.Tensor = n_gt.permute(0, 3, 1, 2).to(device)

        z_pred, n_pred = run_inference(model, x, device, max_depth, model_type)

        # depth
        abs_rel = torch.mean(torch.abs(z_pred - z_gt) / z_gt)
        sq_rel = torch.mean((z_pred - z_gt) ** 2 / z_gt)
        rmse = torch.sqrt(torch.mean((z_pred - z_gt) ** 2))
        rmse_log = torch.sqrt(torch.mean((torch.log(z_pred + 1e-6) - torch.log(z_gt + 1e-6)) ** 2))
        thresh = torch.max(z_gt / z_pred, z_pred / z_gt)
        a1 = (thresh < 1.25).float().mean()
        a2 = (thresh < 1.25 ** 2).float().mean()
        a3 = (thresh < 1.25 ** 3).float().mean()

        depth_metrics.append([abs_rel.item(), sq_rel.item(), rmse.item(), rmse_log.item(),
                              a1.item(), a2.item(), a3.item()])

        # normal
        n_gt = F.normalize(n_gt, p=2, dim=1)
        cos_sim = torch.clamp((n_pred * n_gt).sum(dim=1), -1, 1)
        ang_err = torch.acos(cos_sim) * 180 / np.pi
        mean_ang = ang_err.mean()
        med_ang = ang_err.median()
        acc_11 = (ang_err < 11.25).float().mean()
        acc_22 = (ang_err < 22.5).float().mean()
        acc_30 = (ang_err < 30.0).float().mean()

        normal_metrics.append([mean_ang.item(), med_ang.item(), acc_11.item(), acc_22.item(), acc_30.item()])

    depth_metrics = np.mean(depth_metrics, axis=0)
    normal_metrics = np.mean(normal_metrics, axis=0)

    return depth_metrics, normal_metrics

def main():
    """
    Main entry point for benchmarking DepthSense or DepthAnythingV2 models.

    Loads a dataset, applies inference, saves output predictions, and computes benchmark metrics.
    Outputs are saved as both .npy and .png, and metrics are logged to a JSON file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='vitg') # Our baseline
    parser.add_argument('--max-depth', type=float, default=80.0) # dpt.py default
    parser.add_argument('--model', type=str, required=True, choices=['depthsense', 'depthanythingv2'])
    parser.add_argument('--model-path', type=str, required=False, help='Path to model weights for DepthSense.')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--output-json', type=str, default='benchmark_results.json')
    args = parser.parse_args()

    print("Arguments: " + ", ".join(f"{k}={v}" for k, v in vars(args).items()))

    # Set random seed
    common.set_random_seeds()

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device {device}")

    # ===== [ Same as trainig.py to get the same test_indices]
    # Data splitting.
    dataset: Dataset = DepthSenseDataset(args.data_dir)
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

    test_loader: DataLoader = DataLoader(
        test_set,
        args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        prefetch_factor=4
    )

    # #Features for depthsense
    features: int = 128

    print(f"Loading {args.model} model")
    if args.model == 'depthsense':
        if not args.model_path:
            raise ValueError("--model-path must be specified for model=depthsense")
        if not os.path.isfile(args.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")

        try:
            model = DepthSense(encoder=args.encoder, features=features).to(device)
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            raise RuntimeError(f"Failed to initialize or load DepthSense model: {e}")

    elif args.model == 'depthanythingv2':
        print(f"DepthAnythingV2 vitg not available... failling back to vitl")
        args.encoder = 'vitl'
        try:
            model = DepthAnythingV2(
                encoder='vitl',
                features=256,
                out_channels=[256, 512, 1024, 1024],
            ).to(device)
            model.load_state_dict(torch.load("./checkpoints/depth_anything_v2_vitl.pth", map_location=device))
        except Exception as e:
            raise RuntimeError(f"Failed to initialize or load DepthAnythingV2 model: {e}")
    else:
        raise ValueError(f"Unsupported model type '{args.model}'. Expected 'depthsense' or 'depthanythingv2'.")

    # Perform inference and save visualizations
    run_inference_batch(model, test_loader, device, args.max_depth, Path(args.output_dir), args.model)

    # Compute evaluation metrics
    depth_metrics, normal_metrics = evaluate(model, test_loader, device, args.max_depth, args.model)

    # Save results
    results: dict = {
        'config': vars(args),
        'depth': {
            'abs_rel': depth_metrics[0],
            'sq_rel': depth_metrics[1],
            'rmse': depth_metrics[2],
            'rmse_log': depth_metrics[3],
            'a1': depth_metrics[4],
            'a2': depth_metrics[5],
            'a3': depth_metrics[6],
        },
        'normals': {
            'mean': normal_metrics[0],
            'median': normal_metrics[1],
            '<11.25': normal_metrics[2],
            '<22.5': normal_metrics[3],
            '<30': normal_metrics[4],
        }
    }

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Benchmark results saved to {args.output_json}")


if __name__ == '__main__':
    main()
