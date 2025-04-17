import random

import numpy as np
import torch


def normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalize the given `img`.
    """
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min())


def set_random_seeds(seed: int = 7643) -> None:
    """
    Initializes random seeds, for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
