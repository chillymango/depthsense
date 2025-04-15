import torch
import torch.nn as nn
import torch.nn.functional as F

from util.blocks import FeatureFusionBlock, _make_scratch

def _make_fusion_block(features, use_bn, size=None):
    # Helper function to instantiate a FeatureFusionBlock
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

class DepthSenseHead(nn.Module):
    """
    DepthSenseHead:
    - Dual-head decoder for predicting depth and surface normals.
    - Builds on a DPT-style multi-scale fusion decoder using ViT backbone features.
    """
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        device="cpu",
    ):
        super(DepthSenseHead, self).__init__()
        self.device = device

        self.use_clstoken = use_clstoken

        # Project tokens from ViT into CNN-compatible feature maps
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_ch, kernel_size=1, device=device)
            for out_ch in out_channels
        ])

        # Resize token features to spatial resolution
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, device=device),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, device=device),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1, device=device)
        ])

        if use_clstoken:
            # Optional readout projection when using CLS token from ViT
            self.readout_projects = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * in_channels, in_channels, device=device,),
                    nn.GELU()
                ) for _ in self.projects
            ])

        # Build scratch fusion decoder blocks
        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # Depth decoder head (1-channel)
        self.depth_head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1, device=device),
            nn.ReLU(True),
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1, device=device),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, device=device),
            nn.Sigmoid()
        )

        # Normal decoder head (3-channel)
        self.normal_head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features // 2, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Tanh()  # Raw output, normalized later
        )

    def forward(self, out_features, patch_h, patch_w):
        # Reshape ViT tokens into spatial maps and apply projection + upsampling
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        # Multi-level feature fusion
        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # Upsample and decode predictions
        depth = F.interpolate(path_1, scale_factor=16, mode="bilinear", align_corners=True)
        depth = self.depth_head(depth)

        normals = F.interpolate(path_1, scale_factor=16, mode="bilinear", align_corners=True)
        normals = self.normal_head(normals)

        return depth, normals
