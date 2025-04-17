"""
This module contains logic for a model to generate pseudo-labels (ideally,
the teacher network).
"""

from pathlib import Path

import numpy as np
import torch

from dpt import DepthSense
from util import common


common.set_random_seeds()

# Parameters and hyperparameters used for training.
dataset_path: Path = Path(f"data/pseudo/hypersim")
model_path: str = "models/teacher_hypersim.pth"

device: str = "cpu"
encoder: str = "vits"
features: int = 128

# Model initialization.
model: DepthSense = DepthSense(encoder, features)
model.load_state_dict(torch.load(model_path, device, weights_only=True))

# Pseudo-labeling.
print("Pseudo-labeling started...")
model.eval()

for file in dataset_path.iterdir():
    if file.is_dir():
        # Move to appropriate device.
        x = (
            torch.from_numpy(np.load(file / "frame.npy"))
            .to(device)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .float()
        )
        # Forward pass.
        z, n = model.infer_image(x)
        z = z.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
        n = n.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
        np.save(file / "depth.npy", z)
        np.save(file / "normal.npy", n)

print("Pseudo-labeling finished.")
