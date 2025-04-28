import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from dinov2 import DINOv2
from util.transform import Resize, NormalizeImage, PrepareForNet
from depthsense_head import DepthSenseHead


class DepthSense(nn.Module):
    """
    DepthSense:
    - Monocular depth and surface normal estimation model
    - Combines a DINOv2 ViT encoder, a DPT-style decoder, and GeoNet-style refinement.
    """
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        max_depth=80.0
    ):
        super().__init__()

        # Model parameters
        self.max_depth = max_depth
        self.encoder = encoder

        # Choose ViT block indices to extract intermediate features from
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39],
        }

        # DINOv2 ViT backbone
        self.pretrained = DINOv2(model_name=encoder)

        # Depth + normal dual-head decoder
        self.head = DepthSenseHead(
            in_channels=self.pretrained.embed_dim,
            features=features,
            use_bn=use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
        )

    def forward(self, x, edge_refine=False):
        # Compute patch size for reshaping ViT tokens into spatial maps
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        # Extract ViT features from selected layers
        features = self.pretrained.get_intermediate_layers(
            x,
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True,
        )

        # Predict raw depth and normal maps
        depth, normals = self.head(features, patch_h, patch_w)
        depth = depth.squeeze(1) * self.max_depth

        norm = torch.norm(normals, dim=1, p=2, keepdim=True).clamp(min=1e-6)
        normals = normals / norm

        return depth, normals

    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)

        depth, normal = self.forward(image)

        # Upsample to original resolution
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)  # (1, 1, H, W)
        normal = F.interpolate(normal, (h, w), mode="bilinear", align_corners=True)        # (1, 3, H, W)

        # Remove batch/channel dims for output
        depth_np = depth[0, 0].cpu().numpy()        # (H, W)
        normal_np = normal[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

        return depth_np, normal_np

    def image2tensor(self, raw_image, input_size=518):
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        h, w = raw_image.shape[:2]

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)

        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)

        return image, (h, w)
