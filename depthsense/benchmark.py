import argparse
import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dpt import DepthSense
from util import common

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Depth-Anything-V2')))
from depth_anything_v2.dpt import DepthAnythingV2


class DIODE(Dataset):
    """
    Dataset loader for benchmark inference on depthsense/data/val/indoors/.

    Each sample consists of:
      - RGB .png image (input)
      - *_depth.npy (ground-truth depth)
      - *_normal.npy (ground-truth normals)

    Only images with valid corresponding depth/normal files are used.
    """

    def __init__(self, root_dir: str):
        self.samples: list[tuple[Path, Path, Path]] = []

        root_path: Path = Path(root_dir) / "val" / "indoors"

        # Loop over all .png files and infer matching .npy file paths
        for image_file in root_path.rglob("*.png"):
            base_name: str = image_file.stem
            dir_path: Path = image_file.parent

            depth_file: Path = dir_path / f"{base_name}_depth.npy"
            normal_file: Path = dir_path / f"{base_name}_normal.npy"

            if depth_file.exists() and normal_file.exists():
                self.samples.append((image_file, depth_file, normal_file))
            else:
                print(f"Skipping: missing depth or normal file for {image_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        num_samples = len(self.samples)
        preview_limit = min(3, num_samples)
        preview_lines = "\n".join(
            f"  [{i}] img={img.name}, depth={depth.name}, normal={normal.name}"
            for i, (img, depth, normal) in enumerate(self.samples[:preview_limit])
        )
        return f"DIODE Dataset\nTotal samples: {num_samples}\nSample files:\n{preview_lines}"

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path: Path
        depth_path: Path
        normal_path: Path

        img_path, depth_path, normal_path = self.samples[idx]

        # --- Load and preprocess RGB image ---
        img: np.ndarray = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (252, 196), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img_tensor: torch.Tensor = torch.from_numpy(img)  # (H, W, 3)

        # --- Load and preprocess depth ---
        depth: np.ndarray = np.load(depth_path).astype(np.float32)
        if depth.ndim == 2:
            depth = depth[..., np.newaxis]
        depth = np.nan_to_num(depth, nan=20.0, posinf=20.0, neginf=0.0)
        depth = np.clip(depth, 0, 20.0)
        depth = cv2.resize(depth, (252, 196), interpolation=cv2.INTER_AREA)
        depth_tensor: torch.Tensor = torch.from_numpy(depth)  # (H, W, 1)

        # --- Load and preprocess normals ---
        normal: np.ndarray = np.load(normal_path).astype(np.float32)
        if normal.ndim == 2:
            normal = normal[..., np.newaxis]
        normal = cv2.resize(normal, (252, 196), interpolation=cv2.INTER_AREA)
        normal_tensor: torch.Tensor = torch.from_numpy(normal)  # (H, W, 3)

        return idx, img_tensor, depth_tensor, normal_tensor

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

    for idx, (_i, frame, _depth, _normal) in enumerate(tqdm(dataloader, desc="Inferencing")):
        z_pred, n_pred = run_inference(model, frame, device, max_depth, model_type)

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
            frame_np = frame[b].cpu().numpy()  # frame is still (B, H, W, 3)
            index = idx * dataloader.batch_size + b

            # print(f"depth_np.shape: {depth_np.shape}")
            # print(f"normal_np.shape: {normal_np.shape}")

            np.save(output_dir / f"{index:06d}_depth.npy", depth_np)
            np.save(output_dir / f"{index:06d}_normal.npy", normal_np)

            # Images
            depth_img = (255 * (depth_np / max_depth)).clip(0, 255).astype(np.uint8)
            normal_img = ((normal_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            frame_img = (frame_np * 255).clip(0, 255).astype(np.uint8)

            cv2.imwrite(str(output_dir / f"{index:06d}_depth.png"), depth_img)
            cv2.imwrite(str(output_dir / f"{index:06d}_normal.png"), cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(output_dir / f"{index:06d}_frame.png"), cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))


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

    for _, frame, z_gt, n_gt in tqdm(dataloader, desc="Evaluating"):
        z_gt: torch.Tensor = z_gt.unsqueeze(1).to(device)
        n_gt: torch.Tensor = n_gt.permute(0, 3, 1, 2).to(device)

        z_pred, n_pred = run_inference(model, frame, device, max_depth, model_type)

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
    parser.add_argument('--data-dir', type=str, default='/home/hice1/ylee904/scratch/depthsense/depthsense/data')
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

    # Load benchmark dataset DIODE
    dataset: Dataset = DIODE(args.data_dir)
    print(dataset)

    test_loader: DataLoader = DataLoader(
        dataset,
        args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        prefetch_factor=4
    )

    #Features for depthsense
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
