"""
This module contains common utility functions that only depend on standard
library modules.
"""
import random

import cv2
import numpy as np
import torch
from torch import Tensor


def get_device() -> str:
    """
    Returns the best available device for PyTorch.
    """
    return (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalize the given `img`.
    """
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min())


def normalize_depths(depths: Tensor) -> Tensor:
    """
    Normalizes `depths` to avoid numerical instabilities.
    """
    depths = torch.nan_to_num(depths, 20.0, 20.0, 0.0)
    depths = torch.clamp(depths, 0.0, 20.0)
    return depths


def resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """
    Resizes `img` to the given `size`.
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def set_random_seeds(seed: int = 7643) -> None:
    """
    Initializes random seeds, for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def to_bchw(array: Tensor) -> Tensor:
    """
    Permutes `array` to be in ``(C, H, W)`` or ``(B, C, H, W)`` format, if
    ``array`` is a 3D or 4D tensor, respectively.
    """
    if array.ndim == 3:
        return array.permute(2, 0, 1)
    elif array.ndim == 4:
        return array.permute(0, 3, 1, 2)
    return array


def visualize_depth(depths: Tensor) -> Tensor:
    """
    Normalizes `depths` for visualization.
    """
    depth_vis = []
    for d in depths:
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


def visualize_normals(normals: Tensor) -> Tensor:
    """
    Normalizes `normals` for visualization.
    """
    vis: Tensor = (normals + 1.0) / 2.0
    vis[~torch.isfinite(vis)] = 0.0
    return vis.clamp(0, 1)
