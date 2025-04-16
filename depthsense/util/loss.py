from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CosineSimilarity
from torch.nn import MSELoss
from torch.nn import Module


class Loss(nn.Module):
    """
    GeoNet loss function.
    """

    def __init__(self, depth_scale: float = 1.0, normal_scale: float = 1.0):
        super().__init__()
        self.loss: Module = MSELoss()
        self.cs = CosineSimilarity(dim=0, eps=1e-6)
        self.ds = depth_scale
        self.ns = normal_scale
        self.log_sigma_d = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_n = nn.Parameter(torch.tensor(0.0))

    def depth_loss(self, z_hat: Tensor, z_gt: Tensor):
        ## TODO: the below makes the loss way too high
        ## Coarse depth prediction was log-depth in TF, convert to linear
        ##z_gt[z_mask] = torch.pow(2.0, z_gt[z_mask])
        z_mask = ~torch.isnan(z_gt)
        z_hat_mask = z_hat[z_mask]
        z_gt_mask = z_gt[z_mask]
        return 1 / (2 * torch.exp(self.log_sigma_d)) * self.loss(z_hat_mask / z_hat_mask.mean(), z_gt_mask / z_gt_mask.mean()) + self.log_sigma_d / 2

    def normal_loss(self, n_hat: Tensor, n_gt: Tensor):
        # n_hat should already be normalized and n_gt should already be normalized
        n_mask = ~torch.isnan(n_gt).any(dim=1, keepdim=True).expand_as(n_hat)
        return 1 / (2 * torch.exp(self.log_sigma_n)) * (1 - self.cs(n_hat[n_mask], n_gt[n_mask])) + self.log_sigma_n / 2

    def forward(
        self,
        z_hat: Tensor,
        z_gt: Tensor,
        n_hat: Tensor,
        n_gt: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        return self.depth_loss(z_hat, z_gt), self.normal_loss(n_hat, n_gt)
