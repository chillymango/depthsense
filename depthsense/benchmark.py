import argparse
import os
import json
import cv2
import numpy as np
import matplotlib
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dpt import DepthSense
from util import common

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Depth-Anything-V2/metric_depth')))
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

            # Ground Truths
            depth_file: Path = dir_path / f"{base_name}_depth.npy"
            normal_file: Path = dir_path / f"{base_name}_normal.npy"
            mask_file: Path = dir_path / f"{base_name}_depth_mask.npy"

            if depth_file.exists() and normal_file.exists():
                self.samples.append((image_file, depth_file, normal_file, mask_file))

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        num_samples = len(self.samples)
        preview_limit = min(3, num_samples)
        preview_lines = "\n".join(
            f"  [{i+1}/{num_samples}] img={img.name}, depth={depth.name}, normal={normal.name}, mask={mask.name}"
            for i, (img, depth, normal, mask) in enumerate(self.samples[:preview_limit])
        )
        return f"DIODE Dataset (preview):\n{preview_lines}"

    def __getitem__(self, idx: int) -> tuple[int, str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Resize image, same as training.py
        """
        img_path: Path
        resized_img_path: Path
        depth_path: Path
        normal_path: Path
        mask_path: Path

        img_path, depth_path, normal_path, mask_path = self.samples[idx]

        base_name = img_path.stem
        dir_path = img_path.parent
        resized_img_path = dir_path / f"{base_name}_resized.png"

        # --- Load and preprocess image  ---
        raw_img: np.ndarray = cv2.imread(str(img_path))  # BGR
        resized_img = cv2.resize(raw_img, (252, 196), interpolation=cv2.INTER_AREA)
        # if not resized_img_path.exists():
        cv2.imwrite(str(resized_img_path), (resized_img * 255).astype(np.uint8))

        # --- Load and preprocess depth (GT) ---
        depth: np.ndarray = np.load(depth_path).astype(np.float32)
        depth = np.nan_to_num(depth, nan=20.0, posinf=20.0, neginf=0.0)
        depth = np.clip(depth, 0, 20.0)
        depth = cv2.resize(depth, (252, 196), interpolation=cv2.INTER_AREA)
        depth_tensor: torch.Tensor = torch.from_numpy(depth).float()  # (H, W, 1)

        # --- Load and preprocess normals (GT) ---
        normal: np.ndarray = np.load(normal_path).astype(np.float32)
        normal = cv2.resize(normal, (252, 196), interpolation=cv2.INTER_AREA)
        normal_tensor: torch.Tensor = torch.from_numpy(normal).float()  # (H, W, 3)

        # --- Load and preprocess depth mask (GT) ---
        mask: np.ndarray = np.load(mask_path).astype(np.bool_)
        mask = cv2.resize(mask.astype(np.uint8), (252, 196), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask.astype(bool))  # (H, W)

        if idx < 1:
            print(f"[{idx+1}] {str(img_path)}")
            print(f"[{idx+1}] z_gt={depth_tensor.shape}, n_gt={normal_tensor.shape}, mask_gt={mask_tensor.shape}")

        return idx, str(resized_img_path), depth_tensor, normal_tensor, mask_tensor


@torch.no_grad()
def depth_to_normal(depth: np.ndarray) -> np.ndarray:
    """
    Estimate surface normals from predicted depth using central differences.

    Args:
        depth (np.ndarray): (H, W) predicted depth.

    Returns:
        np.ndarray: (H, W, 3) normalized surface normals.
    """
    depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)

    dzdx = F.pad(depth[:, :, :, 2:] - depth[:, :, :, :-2], (1, 1, 0, 0), mode='replicate') / 2.0
    dzdy = F.pad(depth[:, :, 2:, :] - depth[:, :, :-2, :], (0, 0, 1, 1, 0, 0), mode='replicate') / 2.0

    normal_x = -dzdx
    normal_y = -dzdy
    normal_z = torch.ones_like(depth)

    normals = torch.cat([normal_x, normal_y, normal_z], dim=1)
    normals = F.normalize(normals, p=2, dim=1)

    normals = normals.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    return normals

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    max_depth: float,
    output_dir: Path,
    model_type: str
) -> None:
    """
    Run inference across dataset and save predicted depth and normals as .npy and .png.

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

    # Create subdir for ground truth images
    gt_dir = output_dir / "GT"
    gt_dir.mkdir(parents=True, exist_ok=True)

    for idx, (_i, img_path, z_gt, n_gt, mask_gt) in enumerate(tqdm(dataloader, desc="Inferencing")):
        # --- Read image ---
        img_path = img_path[0]
        raw_image = cv2.imread(img_path)

        # --- Save GT image/depth/normal ---
        cv2.imwrite(str(gt_dir / f"{idx:06d}_frame.png"), raw_image)

        z_gt_np = z_gt.squeeze().cpu().numpy()  # (H, W)
        z_gt_np = (z_gt_np - z_gt_np.min()) / (z_gt_np.max() - z_gt_np.min() + 1e-8) * 255.0
        z_gt_np = z_gt_np.astype(np.uint8)
        cv2.imwrite(str(gt_dir / f"{idx:06d}_depth_gt.png"), z_gt_np)

        n_gt_np = n_gt.squeeze(0).cpu().numpy()  # (H, W, 3)
        n_gt_np = (n_gt_np + 1.0) / 2.0 * 255.0
        n_gt_np[~np.isfinite(n_gt_np)] = 0.0
        n_gt_np = n_gt_np.astype(np.uint8)
        cv2.imwrite(str(gt_dir / f"{idx:06d}_normal_gt.png"), n_gt_np)

        # --- Predict depth and normal from model ----
        if model_type == 'depthanythingv2':
            z_pred = model.infer_image(raw_image)  # only depth (H, W)
            n_pred = depth_to_normal(z_pred)
        else:
            z_pred, n_pred = model.infer_image(raw_image)

        if idx < 1:
            print(f"[{idx+1}] z_pred={z_pred.shape}, z_gt={z_gt.shape}, n_pred={n_pred.shape}, n_gt={n_gt.shape}")

        # --- Save Prediction ---
        np.save(output_dir / f"{idx:06d}_depth.npy", z_pred)
        np.save(output_dir / f"{idx:06d}_normal.npy", n_pred)

        # --- Visualize Prediction ---
        z_pred = (z_pred - z_pred.min()) / (z_pred.max() - z_pred.min() + 1e-8) * 255.0
        z_pred = z_pred.astype(np.uint8)

        n_pred = (n_pred + 1.0) / 2.0 * 255.0
        n_pred[~np.isfinite(n_pred)] = 0.0
        n_pred = n_pred.astype(np.uint8)

        cv2.imwrite(str(output_dir / f"{idx:06d}_depth.png"), z_pred)
        cv2.imwrite(str(output_dir / f"{idx:06d}_normal.png"), n_pred)


@torch.no_grad()
def evaluate(
    dataloader: DataLoader,
    device: str,
    max_depth: float,
    model_type: str,
    output_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate depth and normal accuracy metrics using pre-saved .npy predictions.

    Returns:
        Tuple[np.ndarray, np.ndarray]: depth and normal metric arrays.
    """

    output_dir: Path = output_dir / model_type
    depth_metrics: list[list[float]] = []
    normal_metrics: list[list[float]] = []

    for idx, (_i, _img_path, z_gt, n_gt, mask) in enumerate(tqdm(dataloader, desc="Evaluating")):
        # ---- GT ----
        z_gt: torch.Tensor = z_gt.unsqueeze(1).to(device)  # (1, H, W) -> (1, 1, H, W)
        n_gt: torch.Tensor = n_gt.permute(0, 3, 1, 2).to(device)  # (1, H, W, 3) -> (1, 3, H, W)
        mask: torch.Tensor = mask.unsqueeze(1).to(device)  # (1, H, W) -> (1, 1, H, W)

        # ---- Prediction ----
        z_pred_path: Path = output_dir / f"{idx:06d}_depth.npy"
        n_pred_path: Path = output_dir / f"{idx:06d}_normal.npy"

        z_pred_np: np.ndarray = np.load(z_pred_path) # (H, W)
        n_pred_np: np.ndarray = np.load(n_pred_path) # (H, W, 3)

        z_pred: torch.Tensor = torch.from_numpy(z_pred_np).float().to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        n_pred: torch.Tensor = torch.from_numpy(n_pred_np).float().to(device).unsqueeze(0).permute(0, 3, 1, 2)  # (1, 3, H, W)

        valid_mask: torch.Tensor = (mask & (z_gt > 0) & (z_gt < max_depth))  # (1, 1, H, W)

        # --- Depth metrics ---
        abs_rel: torch.Tensor = torch.sum(torch.abs(z_pred - z_gt)[valid_mask] / z_gt[valid_mask]) / valid_mask.sum()
        sq_rel: torch.Tensor = torch.sum(((z_pred - z_gt) ** 2)[valid_mask] / z_gt[valid_mask]) / valid_mask.sum()
        rmse: torch.Tensor = torch.sqrt(torch.mean((z_pred[valid_mask] - z_gt[valid_mask]) ** 2))
        rmse_log: torch.Tensor = torch.sqrt(torch.mean((torch.log(z_pred[valid_mask] + 1e-6) - torch.log(z_gt[valid_mask] + 1e-6)) ** 2))

        thresh: torch.Tensor = torch.max(z_gt[valid_mask] / z_pred[valid_mask], z_pred[valid_mask] / z_gt[valid_mask])
        a1: torch.Tensor = (thresh < 1.25).float().mean()
        a2: torch.Tensor = (thresh < 1.25 ** 2).float().mean()
        a3: torch.Tensor = (thresh < 1.25 ** 3).float().mean()

        depth_metrics.append([
            abs_rel.item(), sq_rel.item(), rmse.item(), rmse_log.item(),
            a1.item(), a2.item(), a3.item()
        ])

        # --- Normal metrics ---
        n_pred = F.normalize(n_pred, p=2, dim=1) # (1, 3, H, W)
        n_gt = F.normalize(n_gt, p=2, dim=1) # (1, 3, H, W)
        cos_sim: torch.Tensor = torch.clamp((n_pred * n_gt).sum(dim=1), -1, 1)  # (1, H, W)
        ang_err: torch.Tensor = torch.acos(cos_sim) * 180.0 / np.pi  # (1, H, W)

        mean_ang: torch.Tensor = ang_err[valid_mask.squeeze(1)].mean()
        med_ang: torch.Tensor = ang_err[valid_mask.squeeze(1)].median()
        acc_11: torch.Tensor = (ang_err[valid_mask.squeeze(1)] < 11.25).float().mean()
        acc_22: torch.Tensor = (ang_err[valid_mask.squeeze(1)] < 22.5).float().mean()
        acc_30: torch.Tensor = (ang_err[valid_mask.squeeze(1)] < 30.0).float().mean()

        normal_metrics.append([
            mean_ang.item(), med_ang.item(),
            acc_11.item(), acc_22.item(), acc_30.item()
        ])

    depth_result: np.ndarray = np.mean(depth_metrics, axis=0)
    normal_result: np.ndarray = np.mean(normal_metrics, axis=0)

    return depth_result, normal_result


def main():
    """
    Main entry point for benchmarking DepthSense or DepthAnythingV2 models.

    Loads a dataset, applies inference, saves output predictions, and computes benchmark metrics.
    Outputs are saved as both .npy and .png, and metrics are logged to a JSON file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='vitl') # Our baseline
    parser.add_argument('--max-depth', type=float, default=20.0) # 20 for indoor, 80 for outdoor
    parser.add_argument('--model', type=str, required=True, choices=['depthsense', 'depthanythingv2'])
    parser.add_argument('--model-path', type=str, required=False, help='Path to model weights for DepthSense.')
    parser.add_argument('--data-dir', type=str, default='/home/hice1/ylee904/scratch/depthsense/depthsense/data')
    parser.add_argument('--output-dir', type=str, required=True)
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

    # --- Load benchmark dataset DIODE ---
    dataset: Dataset = DIODE(args.data_dir)
    print(dataset)

    test_loader: DataLoader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # --- Features for depthsense ---
    features: int = 128

    print(f"Loading {args.model} model: {args.model_path}")
    if not args.model_path:
        raise ValueError("--model-path must be specified")
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")

    if args.model == 'depthsense':
        try:
            model = DepthSense(encoder=args.encoder, features=features).to(device)
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            raise RuntimeError(f"Failed to initialize or load DepthSense model: {e}")

    elif args.model == 'depthanythingv2':
        if args.encoder == 'vitg':
            args.encoder = 'vitl'
            args.model_path = './checkpoints/depth_anything_v2_vitl.pth'
            print(f"DepthAnythingV2 vitg not available... failling back to vitl - {args.model_path}")
        try:
            model = DepthAnythingV2(
                encoder=args.encoder,
                features=256,
                out_channels=[256, 512, 1024, 1024],
            ).to(device)
            model.load_state_dict(torch.load(args.model_path, map_location=device))
        except Exception as e:
            raise RuntimeError(f"Failed to initialize or load DepthAnythingV2 model: {e}")
    else:
        raise ValueError(f"Unsupported model type '{args.model}'. Expected 'depthsense' or 'depthanythingv2'.")

    # Perform inference and save visualizations
    run_inference(model, test_loader, device, args.max_depth, Path(args.output_dir), args.model)

    # Compute evaluation metrics (mean value)
    depth_metrics, normal_metrics = evaluate(test_loader, device, args.max_depth, args.model, Path(args.output_dir))

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
