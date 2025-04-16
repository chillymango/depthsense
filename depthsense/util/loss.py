from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import MSELoss
from torch.nn import Module


class Loss(nn.Module):
    """
    GeoNet loss function.
    """

    def __init__(self, depth_scale: float = 1.0, normal_scale: float = 1.0):
        super().__init__()
        self.loss: Module = MSELoss()
        self.ds = depth_scale
        self.ns = normal_scale

    def forward(
        self,
        z_hat: Tensor,
        z_gt: Tensor,
        n_hat: Tensor,
        n_gt: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        z_mask = ~torch.isnan(z_hat) & ~torch.isnan(z_gt)
        n_mask = ~torch.isnan(n_hat) & ~torch.isnan(n_gt)
        # TODO: the below makes the loss way too high
        # Coarse depth prediction was log-depth in TF, convert to linear
        #z_gt[z_mask] = torch.pow(2.0, z_gt[z_mask])
        return self.ds * self.loss(z_hat[z_mask], z_gt[z_mask]), self.ns * self.loss(n_hat[n_mask], n_gt[n_mask])
