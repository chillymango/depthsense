import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import MSELoss
from torch.nn import Module


class GeoNetLoss(nn.Module):
    """
    GeoNet loss function.
    """

    def __init__(self):
        super().__init__()
        self.loss: Module = MSELoss()

    def forward(
        self,
        z_hat: Tensor,
        z_gt: Tensor,
        n_hat: Tensor,
        n_gt: Tensor,
    ) -> Tensor:
        # Coarse depth prediction was log-depth in TF, convert to linear
        z_gt = torch.pow(2.0, z_gt)
        return self.loss(z_hat, z_gt) + self.loss(n_hat, n_gt)
