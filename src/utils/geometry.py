import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


def angle_axis_to_quaternion(angle_axis: torch.Tensor) -> torch.Tensor:
    """
    Convert an angle axis to a quaternion.
    Args:
        angle_axis (Tensor): Tensor of Nx3 representing the rotations
    Returns:
        quaternions (Tensor): Tensor of Nx4 representing the quaternions
    References:
        https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py

    Equations
    qx = ax * sin(angle/2)
    qy = ay * sin(angle/2)
    qz = az * sin(angle/2)
    qw = cos(angle/2)

    where:

    the axis is normalised so: ax*ax + ay*ay + az*az = 1
    the quaternion is also normalised so cos(angle/2)2 + ax*ax * sin(angle/2)2 + ay*ay * sin(angle/2)2+ az*az * sin(angle/2)2 = 1
    """
    angle = torch.norm(angle_axis, p=2, dim=-1, keepdim=True)  # N, 1
    half_angle = 0.5 * angle
    eps = 1e-6
    small_angle = angle.data.abs() < eps
    sin_half_angle = torch.sin(half_angle)
    cos_half_angle = torch.cos(half_angle)
    # for small angle, use taylor series
    sin_half_angle = torch.where(
        small_angle, half_angle - 0.5 * half_angle**3, sin_half_angle
    )
    cos_half_angle = torch.where(small_angle, 1 - 0.5 * half_angle**2, cos_half_angle)
    quaternions = torch.cat(
        [cos_half_angle, sin_half_angle * F.normalize(angle_axis, dim=-1)], dim=-1
    )
    return quaternions


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    # Unpack terms
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)

    # Standard Hamilton product
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (XYZW, scalar-last) to rotation matrices (..., 3, 3)."""
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack([
        1 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1 - two_s * (i * i + j * j),
    ], -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices (..., 3, 3) to quaternions (XYZW, scalar-last)."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1))

    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)

    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    out = out[..., [1, 2, 3, 0]]  # rijk → ijkr (scalar-last)
    return torch.where(out[..., 3:4] < 0, -out, out)  # standardize sign


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """torch.sqrt(max(0, x)) with zero subgradient where x == 0."""
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

@torch.jit.script
def affine_inverse(A: torch.Tensor) -> torch.Tensor:
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)


def normalize_extrinsics(ex_t: torch.Tensor) -> torch.Tensor:
    transform = affine_inverse(ex_t[:, :1])
    ex_t_norm = ex_t @ transform
    c2ws = affine_inverse(ex_t_norm)
    translations = c2ws[..., :3, 3]
    dists = translations.norm(dim=-1)
    median_dist = torch.median(dists)
    median_dist = torch.clamp(median_dist, min=1e-1)
    ex_t_norm[..., :3, 3] = ex_t_norm[..., :3, 3] / median_dist
    return ex_t_norm


def extri_intri_to_pose_encoding(extrinsics, intrinsics, image_size_hw=None):
    R = extrinsics[:, :, :3, :3]
    T = extrinsics[:, :, :3, 3]
    quat = mat_to_quat(R)
    H, W = image_size_hw
    fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
    return torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()


def pose_encoding_to_extri_intri(pose_encoding, image_size_hw=None):
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]
    R = quat_to_mat(quat)
    extrinsics = torch.cat([R, T[..., None]], dim=-1)
    H, W = image_size_hw
    fy = (H / 2.0) / torch.clamp(torch.tan(fov_h / 2.0), 1e-6)
    fx = (W / 2.0) / torch.clamp(torch.tan(fov_w / 2.0), 1e-6)
    intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = W / 2
    intrinsics[..., 1, 2] = H / 2
    intrinsics[..., 2, 2] = 1.0
    return extrinsics, intrinsics


def batchify_unproject_depth_map_to_point_map(
    depth_map: torch.Tensor,
    extrinsics_cam: torch.Tensor,
    intrinsics_cam: torch.Tensor,
) -> torch.Tensor:
    """Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map: (B, V, H, W) or (B, V, H, W, 1)
        extrinsics_cam: (B, V, 3, 4) w2c matrices
        intrinsics_cam: (B, V, 3, 3)

    Returns:
        (B, V, H, W, 3) world-space 3D points
    """
    if depth_map.dim() == 5:
        depth_map = depth_map.squeeze(-1)

    B, V, H, W = depth_map.shape

    intrinsics_cam = intrinsics_cam.flatten(0, 1)   # (B*V, 3, 3)
    extrinsics_cam = extrinsics_cam.flatten(0, 1)   # (B*V, 3, 4)
    depth_map      = depth_map.flatten(0, 1)        # (B*V, H, W)

    fu = intrinsics_cam[:, 0, 0]
    fv = intrinsics_cam[:, 1, 1]
    cu = intrinsics_cam[:, 0, 2]
    cv = intrinsics_cam[:, 1, 2]

    u = torch.arange(W, device=depth_map.device)[None, None, :].expand(B * V, H, W)
    v = torch.arange(H, device=depth_map.device)[None, :, None].expand(B * V, H, W)

    x_cam = (u - cu[:, None, None]) * depth_map / fu[:, None, None]
    y_cam = (v - cv[:, None, None]) * depth_map / fv[:, None, None]
    cam_coords = torch.stack((x_cam, y_cam, depth_map), dim=-1)  # (B*V, H, W, 3)

    cam_to_world = _closed_form_inverse_se3(extrinsics_cam)  # (B*V, 4, 4)
    homo_pts = torch.cat((cam_coords, torch.ones_like(cam_coords[..., :1])), dim=-1).flatten(1, 2)
    world_coords = torch.bmm(cam_to_world, homo_pts.transpose(1, 2)).transpose(1, 2)[:, :, :3]

    return world_coords.view(B, V, H, W, 3)


def _closed_form_inverse_se3(se3: torch.Tensor) -> torch.Tensor:
    """Batch-invert a (N, 3, 4) or (N, 4, 4) SE3 matrix via R^T."""
    R = se3[:, :3, :3]
    T = se3[:, :3, 3:]
    R_t = R.transpose(1, 2)
    inv = torch.eye(4, device=se3.device, dtype=se3.dtype)[None].expand(len(R), -1, -1).clone()
    inv[:, :3, :3] = R_t
    inv[:, :3, 3:] = -torch.bmm(R_t, T)
    return inv


def normalize_intrinsics(
    intrinsic: Float[Tensor, "... 3 3"],
    width: int,
    height: int,
) -> Float[Tensor, "... 3 3"]:
    return torch.stack(
        [
            intrinsic[..., 0, :] / width,
            intrinsic[..., 1, :] / height,
            intrinsic[..., 2, :],
        ],
        dim=-2,
    )
