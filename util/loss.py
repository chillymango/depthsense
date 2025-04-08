import torch
import torch.nn as nn

class GeoNetLoss(nn.Module):
    """
    GeoNet Loss Function
    Includes:
    - Absolute error of coarse depth (loss1)
    - Absolute error of refined depth (loss2)
    - Absolute error of coarse normals (loss3)
    - Absolute error of refined normals (loss4)
    Only loss2 and loss4 are used for optimization.
    """
    def __init__(self):
        super().__init__()

    def forward(self, fc8_upsample, final_depth,
                fc8_upsample_norm, norm_pred_noise,
                batch_depths, batch_norms,
                batch_depth_masks, batch_masks):

        # Coarse depth prediction was log-depth in TF, convert to linear
        exp_depth = torch.pow(2.0, fc8_upsample)

        # Depth losses
        loss1 = torch.sum(torch.abs(exp_depth - batch_depths) * batch_depth_masks) / (torch.sum(batch_depth_masks) + 1.0)
        loss2 = torch.sum(torch.abs(final_depth - batch_depths) * batch_depth_masks) / (torch.sum(batch_depth_masks) + 1.0)

        # Normal losses
        batch_masks = batch_masks.expand_as(batch_norms)
        loss3 = torch.sum(torch.abs(fc8_upsample_norm - batch_norms) * batch_masks) / (torch.sum(batch_masks) + 1.0)
        loss4 = torch.sum(torch.abs(norm_pred_noise - batch_norms) * batch_masks) / (torch.sum(batch_masks) + 1.0)

        total_loss = loss2 + loss4

        return total_loss, {
            "loss1 (coarse depth)": loss1.item(),
            "loss2 (refined depth)": loss2.item(),
            "loss3 (coarse normals)": loss3.item(),
            "loss4 (refined normals)": loss4.item(),
        }
