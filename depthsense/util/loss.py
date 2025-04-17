from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.log_sigma_d = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_n = nn.Parameter(torch.tensor(0.0))

    def depth_loss(self, z_hat: Tensor, z_gt: Tensor):
        ## TODO: the below makes the loss way too high
        ## Coarse depth prediction was log-depth in TF, convert to linear
        #z_gt[z_mask] = torch.pow(2.0, z_gt[z_mask])
        z_mask = ~torch.isnan(z_gt)
        z_hat_mask = z_hat[z_mask]
        z_gt_mask = z_gt[z_mask]
        ##return 1 / (2 * torch.exp(self.log_sigma_d)) * self.loss(z_hat_mask / z_hat_mask.mean(), z_gt_mask / z_gt_mask.mean()) + self.log_sigma_d / 2
        return self.loss(z_hat_mask / z_hat_mask.mean(), z_gt_mask / z_gt_mask.mean()) * self.ds
        #return self.loss(z_hat_mask, z_gt_mask)

    def normal_loss(self, n_hat: Tensor, n_gt: Tensor):
        n_mask = ~torch.isnan(n_gt)
        return self.loss(n_hat[n_mask], n_gt[n_mask]) * self.ns

        # TODO: why does cosine loss work so much worse???
        # n_hat should already be normalized and n_gt should already be normalized
        #n_hat_flat = n_hat.permute(0, 2, 3, 1).reshape(-1, 3)
        #n_gt_flat = n_gt.permute(0, 2, 3, 1).reshape(-1, 3)

        #n_mask = ~torch.isnan(n_gt_flat).any(dim=1)

        #cos_sim = F.cosine_similarity(n_hat_flat[n_mask], n_gt_flat[n_mask], dim=1)
        #return 1 / (2 * torch.exp(self.log_sigma_n)) * (1 - cos_sim).mean() + self.log_sigma_n / 2
        #return (1 - cos_sim).mean() * self.ns

    def forward(
        self,
        z_hat: Tensor,
        z_gt: Tensor,
        n_hat: Tensor,
        n_gt: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        return self.depth_loss(z_hat, z_gt), self.normal_loss(n_hat, n_gt)
