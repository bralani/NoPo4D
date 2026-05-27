import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from src.utils.geometry import get_normal_map


def _normal_map_loss(
    depth_a: Float[Tensor, "n h w"],
    depth_b: Float[Tensor, "n h w"],
    intrinsics: Float[Tensor, "n 3 3"],
) -> Float[Tensor, ""]:
    """Cosine + L1 loss between normal maps derived from two depth maps."""
    normals_a = get_normal_map(depth_a, intrinsics)
    normals_b = get_normal_map(depth_b, intrinsics)
    cosine = (1 - (normals_a * normals_b).sum(-1)).mean()
    l1     = F.l1_loss(normals_a, normals_b)
    return torch.nan_to_num((cosine + l1) / 2, nan=0.0)
