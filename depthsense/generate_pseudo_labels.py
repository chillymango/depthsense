"""
This module contains logic for a model to generate pseudo-labels (ideally,
the teacher network).
"""

from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from dpt import DepthSense
from util import common


common.set_random_seeds()

parser: ArgumentParser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="hypersim")
parser.add_argument("--model", type=str, default="model-small.pth")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--encoder", type=str, default="vits")
parser.add_argument("--features", type=int, default=128)
parser.add_argument("--shuffle", type=bool, default=False)
parser.add_argument("--size", type=int, nargs=2, default=(252, 196))


class UnlabeledDataset(Dataset):
    """
    Wrapper for unlabeled data.
    """

    def __init__(self, root: Path, img_size: tuple[int, int]):
        self.files: list[Path] = [f for f in root.iterdir() if f.is_dir()]
        self.size: tuple[int, int] = img_size

    def __getitem__(self, i: int) -> tuple[np.ndarray, Path]:
        file: Path = self.files[i]
        # Assumes shape (H, W, C).
        x = common.resize(np.load(file / "frame.npy"), self.size)
        # Reshapes to (C, H, W) to pass through the model.
        x = torch.from_numpy(x).permute(2, 0, 1)
        return x, str(file)

    def __len__(self) -> int:
        return len(self.files)


def generate_labels():
    # Parameters.
    args: Namespace = parser.parse_args()
    print("Arguments:", args)

    data_path: Path = Path(f"data/pseudo/{args.dataset}")
    model_path: str = f"models/{args.model}"
    batch_size: int = args.batch_size
    device: str = args.device
    encoder: str = args.encoder
    features: int = args.features
    shuffle: bool = args.shuffle
    size: tuple[int, int] = args.size

    # Data preparation.
    dataset: Dataset = UnlabeledDataset(data_path, size)
    loader: DataLoader = DataLoader(dataset, batch_size, shuffle)

    # Model initialization.
    model: DepthSense = DepthSense(encoder, features).to(device)
    model.load_state_dict(torch.load(model_path, device, weights_only=True))
    model.eval()

    print(f"Using model {model_path} running on device {device}.")

    # Pseudo-labeling.
    print("Pseudo-labeling started...")

    for i, (x_batch, files) in enumerate(loader):
        print("=" * 50)
        print(f"Labeling batch {i + 1}, of {len(x_batch)} images...")
        x_batch = x_batch.to(device).float()
        # Forward pass.
        z_batch, n_batch = model.infer_image(x_batch)
        z_batch = z_batch.permute(0, 2, 3, 1).cpu().numpy()
        n_batch = n_batch.permute(0, 2, 3, 1).cpu().numpy()
        # Saving the predicted depth and normal maps.
        print("Saving predicted depth and normal maps...")
        for z, n, file in zip(z_batch, n_batch, files):
            file = Path(file)
            np.save(file / "depth.npy", z)
            np.save(file / "normal.npy", n)

    print("=" * 50)
    print("Pseudo-labeling completed.")


if __name__ == "__main__":
    generate_labels()
