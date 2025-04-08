import torch
import torch.nn.functional as F
import torch.nn as nn
import np, cv2

crop_size_h = 481
crop_size_w = 641
k = 9
rate = 4
thresh = 0.95

# -----------------------------------------------------------------------------
# depth_to_normal and normal_to_depth from GeoNet paper's formula
# -----------------------------------------------------------------------------

def depth_to_normal(depth, intrinsics):
    """
    Estimate normals from depth using finite difference method and camera intrinsics.
    GeoNet - 3.1. Pinhole Camera Model

    Args:
        depth: (B, 1, H, W) depth map
        intrinsics: (B, 3, 3) camera intrinsics
    Returns:
        normals: (B, 3, H, W) surface normals (unit vectors)
    """
    B, _, H, W = depth.shape
    fx = intrinsics[:, 0, 0].view(B, 1, 1, 1)
    fy = intrinsics[:, 1, 1].view(B, 1, 1, 1)
    cx = intrinsics[:, 0, 2].view(B, 1, 1, 1)
    cy = intrinsics[:, 1, 2].view(B, 1, 1, 1)

    device = depth.device
    y, x = torch.meshgrid(
        torch.arange(0, H, device=device),
        torch.arange(0, W, device=device), indexing='ij'
    )
    x = x.float().view(1, 1, H, W)
    y = y.float().view(1, 1, H, W)

    Z = depth
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    pts = torch.cat([X, Y, Z], dim=1)  # (B, 3, H, W)

    dx = F.pad(pts[:, :, :, 1:] - pts[:, :, :, :-1], (0, 1), mode='replicate')
    dy = F.pad(pts[:, :, 1:, :] - pts[:, :, :-1, :], (0, 0, 0, 1), mode='replicate')

    normal = torch.cross(dx, dy, dim=1)
    normal = normal + (normal == 0).float() * 1e-6
    normal = F.normalize(normal, p=2, dim=1)

    return normal


def normal_to_depth(normals, intrinsics):
    """
    Estimate depth from surface normals using perspective geometry and intrinsics.
    GeoNet - 3.2 Normal-to-Depth Network

    Args:
        normals: (B, 3, H, W)
        intrinsics: (B, 3, 3)
    Returns:
        relative depth map: (B, 1, H, W)
    """
    B, _, H, W = normals.shape
    fx = intrinsics[:, 0, 0].view(B, 1, 1, 1)
    fy = intrinsics[:, 1, 1].view(B, 1, 1, 1)
    cx = intrinsics[:, 0, 2].view(B, 1, 1, 1)
    cy = intrinsics[:, 1, 2].view(B, 1, 1, 1)

    device = normals.device
    y, x = torch.meshgrid(
        torch.arange(0, H, device=device),
        torch.arange(0, W, device=device), indexing='ij'
    )
    x = x.float().view(1, 1, H, W)
    y = y.float().view(1, 1, H, W)

    nx, ny, nz = normals[:, 0:1], normals[:, 1:2], normals[:, 2:3] + 1e-6

    x_shifted = (x - cx) / fx
    y_shifted = (y - cy) / fy

    denom = x_shifted * nx + y_shifted * ny + nz
    denom = denom + (denom == 0).float() * 1e-6

    depth = 1.0 / denom
    return depth


def myfunc_canny(img):
    """
    canny edge extractor
    """
    crop_size_h = 481
    crop_size_w = 641

    img = np.squeeze(img)
    img = img + 128.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(img.shape())
    img = ((img-img.min())/(img.max()-img.min()))*255.0
    edges = cv2.Canny(img.astype('uint8'), 100, 220)
    edges = edges.astype(np.float32)
    edges = edges.reshape((1,crop_size_h,crop_size_w,1))
    edges = 1 - edges/255.0
    return edges

def propagate(data, dlr, drl, dud, ddu, dim):
    """
    Directional propagation over depth or normals using learned edge-aware weights.
    Args:
        data: (B, C, H, W) tensor to propagate (e.g., depth or normals)
        dlr, drl, dud, ddu: (B, 1, H, W) edge weights for 4 directions
        dim: int, number of channels in data (1 or 3)
    Returns:
        propagated: (B, C, H, W)
    """
    B, _, H, W = data.shape
    if dim > 1:
        dlr = dlr.repeat(1, dim, 1, 1)
        drl = drl.repeat(1, dim, 1, 1)
        dud = dud.repeat(1, dim, 1, 1)
        ddu = ddu.repeat(1, dim, 1, 1)

    x = torch.zeros((B, dim, H, 1), device=data.device)
    current = torch.cat([x, data], dim=3)[..., :W]
    out = current * dlr + data * (1 - dlr)

    x = torch.zeros((B, dim, H, 1), device=data.device)
    current = torch.cat([out, x], dim=3)[..., 1:]
    out = current * drl + out * (1 - drl)

    x = torch.zeros((B, dim, 1, W), device=data.device)
    current = torch.cat([x, out], dim=2)[:, :, :H, :]
    out = current * dud + out * (1 - dud)

    x = torch.zeros((B, dim, 1, W), device=data.device)
    current = torch.cat([out, x], dim=2)[:, :, 1:, :]
    out = current * ddu + out * (1 - ddu)

    return out


class NormalRefinement(nn.Module):
    def __init__(self, batch_size):
        super(NormalRefinement, self).__init__()
        self.batch_size = batch_size

        # Matches TF slim.repeat blocks from conv1_noise to encode_norm_noise
        self.cnn_refine = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 3, kernel_size=3, padding=1)
        )

        # Matches final slim.repeat and conv stack (conv1_norm_noise_new → norm_conv3_noise_new)
        self.final_refine = nn.Sequential(
            nn.Conv2d(9, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, fc8_upsample_norm, fc8_upsample, grid, inputs):
        # tf.squeeze + tf.reshape
        fc8_upsample_norm = fc8_upsample_norm.squeeze()
        fc8_upsample_norm = fc8_upsample_norm.reshape(self.batch_size, crop_size_h, crop_size_w, 3)

        # tf.extract_image_patches -> unfold on normalized feature map
        fc8_nchw = fc8_upsample_norm.permute(0, 3, 1, 2)
        norm_matrix = nn.Unfold(kernel_size=k, dilation=rate, padding=(rate * (k - 1)) // 2, stride=1)(fc8_nchw)
        norm_matrix = norm_matrix.view(self.batch_size, 3, k * k, crop_size_h, crop_size_w)
        matrix_c = norm_matrix.permute(0, 3, 4, 2, 1)

        # tf.expand_dims
        fc8_expanded = fc8_upsample_norm.unsqueeze(-1)
        # tf.matmul
        angle = torch.matmul(matrix_c, fc8_expanded)
        # tf.greater
        valid_condition = angle > thresh
        # tf.tile
        valid_condition_all = valid_condition.repeat(1, 1, 1, 1, 3)

        # tf.exp(x * ln(2)) equivalent to 2^x
        exp_depth = torch.exp(fc8_upsample * torch.log(torch.tensor(2.0)))
        depth_repeat = exp_depth.repeat(1, 1, 1, 3)
        # tf.multiply
        points = grid * depth_repeat.permute(0, 3, 1, 2)

        # tf.extract_image_patches on depth-scaled grid points
        points_nchw = points.permute(0, 1, 2, 3) if points.shape[1] == 3 else points.permute(0, 3, 1, 2)
        point_matrix = nn.Unfold(kernel_size=k, dilation=rate, padding=(rate * (k - 1)) // 2, stride=1)(points_nchw)
        point_matrix = point_matrix.view(self.batch_size, 3, k * k, crop_size_h, crop_size_w)
        matrix_a = point_matrix.permute(0, 3, 4, 2, 1)

        # tf.zeros_like + tf.where
        matrix_a_zero = torch.zeros_like(matrix_a)
        matrix_a_valid = torch.where(valid_condition_all, matrix_a, matrix_a_zero)

        # tf.matrix_transpose
        matrix_a_trans = matrix_a_valid.transpose(-2, -1)
        # tf.ones
        matrix_b = torch.ones((self.batch_size, crop_size_h, crop_size_w, k * k, 1), device=matrix_a.device)

        # tf.matmul
        point_multi = torch.matmul(matrix_a_trans, matrix_a_valid)
        # tf.matrix_determinant
        matrix_deter = torch.linalg.det(point_multi.cpu()).to(matrix_a.device)
        # tf.greater + tf.expand_dims + tf.tile
        inverse_condition = (matrix_deter > 1e-5).unsqueeze(-1).unsqueeze(-1)
        inverse_condition_all = inverse_condition.repeat(1, 1, 1, 3, 3)

        # tf.diag + tf.tile
        diag_matrix = torch.eye(3, device=matrix_a.device).reshape(1, 1, 1, 3, 3).repeat(self.batch_size, crop_size_h, crop_size_w, 1, 1)
        # tf.where
        inversible_matrix = torch.where(inverse_condition_all, point_multi, diag_matrix)
        # tf.matrix_inverse
        inv_matrix = torch.linalg.inv(inversible_matrix.cpu()).to(matrix_a.device)

        # tf.matmul
        generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans), matrix_b)

        # slim.unit_norm
        norm_normalize = F.normalize(generated_norm, p=2, dim=3)
        # tf.reshape
        norm_normalize = norm_normalize.view(self.batch_size, crop_size_h, crop_size_w, 3)
        # norm_scale = * 10 (scale-up for subsequent CNN input)
        norm_normalize *= 10.0

        # CNN refinement
        norm_input = norm_normalize.permute(0, 3, 1, 2)
        encode_norm_noise = self.cnn_refine(norm_input)
        # tf.image.resize_images
        encode_norm_upsample_noise = F.interpolate(encode_norm_noise, size=(crop_size_h, crop_size_w), mode='bilinear', align_corners=True)

        # tf.add
        sum_norm_noise = norm_normalize + encode_norm_upsample_noise.permute(0, 2, 3, 1)
        # slim.unit_norm
        norm_pred_noise = F.normalize(sum_norm_noise, p=2, dim=3)

        # tf.concat + input normalization (1/255)
        inputs_scaled = inputs / 255.0
        concat = torch.cat([
            fc8_upsample_norm,
            norm_pred_noise,
            inputs_scaled.permute(0, 2, 3, 1)
        ], dim=3)  # [B, H, W, 9]

        # Final CNN stack
        concat = concat.permute(0, 3, 1, 2)  # [B, 9, H, W]
        norm_pred_final = self.final_refine(concat)  # [B, 3, H, W]
        return F.normalize(norm_pred_final.permute(0, 2, 3, 1), p=2, dim=3)  # [B, H, W, 3] normalized output

class DepthRefinement(nn.Module):
    def __init__(self, batch_size):
        super(DepthRefinement, self).__init__()
        self.batch_size = batch_size

        # Final refinement CNN layers (conv1_depth_noise_new → depth_conv3_noise_new)
        self.refine_depth = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=3, padding=1)
        )

    def forward(self, matrix_a, matrix_c, grid, angle, valid_condition, exp_depth, inputs):
        # tf.extract_image_patches(grid)
        grid_nchw = grid.permute(0, 3, 1, 2)
        grid_patch = nn.Unfold(kernel_size=k, dilation=rate, padding=(rate * (k - 1)) // 2, stride=1)(grid_nchw)
        grid_patch = grid_patch.view(self.batch_size, 3, k * k, crop_size_h, crop_size_w).permute(0, 3, 4, 2, 1)

        # tf.split(matrix_a)[2] — extract depth_data from last channel
        depth_data = matrix_a[..., 2:3]  # [B, H, W, k*k, 1]

        # tf.where(valid_condition, angle, 0)
        valid_angle = torch.where(valid_condition, angle, torch.zeros_like(angle))

        # tf.expand_dims(grid)
        grid_exp = grid.unsqueeze(-1)  # [B, H, W, 3, 1]
        # tf.matmul(matrix_c, grid_exp)
        lower_matrix = torch.matmul(matrix_c, grid_exp)

        # tf.where + tf.reciprocal
        condition = lower_matrix > 1e-5
        lower = torch.where(condition, lower_matrix, torch.ones_like(lower_matrix))
        lower = 1.0 / lower

        # tf.where(..., angle, 0)
        valid_angle = torch.where(condition, valid_angle, torch.zeros_like(valid_angle))

        # tf.reduce_sum(matrix_c * grid_patch, axis=4)
        upper = torch.sum(matrix_c * grid_patch, dim=4)
        # tf.expand_dims(upper)
        ratio = lower * upper.unsqueeze(-1)
        # tf.multiply(ratio, depth_data)
        estimate_depth = ratio * depth_data

        # tf.tile + tf.reduce_sum + reciprocal (angle normalization)
        valid_sum = torch.sum(valid_angle, dim=(3, 4), keepdim=True) + 1e-5
        valid_angle = valid_angle / valid_sum

        # tf.reduce_sum(estimate_depth * valid_angle, axis=[3, 4])
        depth_stage1 = torch.sum(estimate_depth * valid_angle, dim=(3, 4))
        depth_stage1 = depth_stage1.unsqueeze(2)
        # tf.clip_by_value
        depth_stage1 = torch.clamp(depth_stage1, 0.0, 10.0)

        # squeeze + expand to match TensorFlow behavior
        exp_depth = exp_depth.squeeze(3).unsqueeze(2)

        # input normalization
        input_scaled = inputs.squeeze(3) / 255.0

        # tf.concat + tf.expand_dims
        depth_all = torch.cat([depth_stage1, exp_depth, input_scaled], dim=2).unsqueeze(0)

        # CNN refinement on stacked features
        depth_pred_final = self.refine_depth(depth_all)
        return depth_pred_final

class EdgeRefinement(nn.Module):
    def __init__(self):
        super(EdgeRefinement, self).__init__()
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=3, padding=1)
        )

    def forward(self, image, depth, normals):
        edge = myfunc_canny(image)

        # edge prediction for depth
        edge_inputs = torch.cat([image * (2./256.), edge], dim=1)
        weights = self.edge_encoder(edge_inputs)
        weights = torch.clip(weights + edge.repeat(1, 8, 1, 1), 0.0, 1.0)

        dlr, drl, dud, ddu, nlr, nrl, nud, ndu = torch.chunk(weights, chunks=8, dim=1)

        # 4-pass depth refinement
        for _ in range(4):
            depth = propagate(depth, dlr, drl, dud, ddu, dim=1)

        # 4-pass normal refinement + normalization
        for _ in range(4):
            normals = propagate(normals, nlr, nrl, nud, ndu, dim=3)
            normals = F.normalize(normals, p=2, dim=1)

        return depth, normals