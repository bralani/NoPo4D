from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor
from .types import UnbatchedExample


def convert_intrinsics(meta_data: dict[str, Any]) -> Float[Tensor, "3 3"]:
    """Convert stored meta data into a normalized 3x3 intrinsics matrix.

    Args:
        meta_data: Dictionary containing keys 'h', 'w', 'fl_x' ('fx'), 'fl_y' ('fy'), 'cx', 'cy'.

    Returns:
        A 3x3 float32 tensor with normalized intrinsics (values in [0,1] w.r.t stored image size).
    """
    store_h, store_w = meta_data["h"], meta_data["w"]
    fx = meta_data.get("fl_x", meta_data.get("fx"))
    fy = meta_data.get("fl_y", meta_data.get("fy"))
    cx = meta_data["cx"]
    cy = meta_data["cy"]
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics[0, 0] = float(fx) / float(store_w)
    intrinsics[1, 1] = float(fy) / float(store_h)
    intrinsics[0, 2] = float(cx) / float(store_w)
    intrinsics[1, 2] = float(cy) / float(store_h)
    return intrinsics


def make_baseline_one(
    extrinsics: Float[Tensor, "num_views 4 4"],
    context_indices: Float[Tensor, "context_views"],
    baseline_min: float,
    baseline_max: float,
) -> float:
    """Resize the world to make the baseline equal to 1.

    This mutates the provided ``extrinsics`` tensor in-place by scaling the
    translation components ("t"). 

    Args:
        extrinsics: Tensor of shape [N, 4, 4] containing camera-to-world poses.
        context_indices: Indices selecting the context views (indexable into
            ``extrinsics``).
        baseline_min: Minimum allowed baseline value.
        baseline_max: Maximum allowed baseline value.

    Returns:
        scale (float): The scale that was applied to the translations.

    Raises:
        Exception: if the computed baseline is out of the allowed range.
    """
    context_extrinsics = extrinsics[context_indices]
    first_cam_pos = context_extrinsics[0, :3, 3]
    last_cam_pos = context_extrinsics[-1, :3, 3]
    scale = (first_cam_pos - last_cam_pos).norm().item()
    if scale < baseline_min or scale > baseline_max:
        print(
            f"Skipped because of baseline out of range: "
            f"{scale:.6f}"
        )
        raise Exception("baseline out of range")

    # mutate in-place to preserve downstream behaviour
    extrinsics[:, :3, 3] /= scale

    return float(scale)


def prepare_pts3d_and_normalize(
    example: UnbatchedExample,
    target_images: Float[Tensor, "M C H W"],
    normalize_by_pts3d: bool = False
) -> UnbatchedExample:
    """
    Add 3D point tensors and valid masks to the example dict for context and target views.
    Optionally normalize all relevant fields by the mean norm of valid context 3D points.

    Args:
        example: Dictionary with 'context' and 'target' keys containing image, depth, extrinsics.
        target_images: Tensor of shape [M, C, H, W] for target images.
        normalize_by_pts3d: If True, normalize by mean norm of valid context 3D points.

    Returns:
        Modified example dict with 'pts3d' and 'valid_mask' fields added, and normalized if requested.
    """
    assert "context" in example and "target" in example, "Example must contain 'context' "
    "and 'target' keys."
    context_image = example["context"]["image"]  # [N, C, H, W]

    context_pts3d = torch.ones_like(context_image).permute(0, 2, 3, 1)  # [N, H, W, 3]
    context_valid_mask = torch.ones_like(context_image)[:, 0].bool()      # [N, H, W]

    target_pts3d = torch.ones_like(target_images).permute(0, 2, 3, 1)    # [M, H, W, 3]
    target_valid_mask = torch.ones_like(target_images)[:, 0].bool()       # [M, H, W]

    if normalize_by_pts3d:
        valid_pts3d = context_pts3d[context_valid_mask]  # [num_valid, 3]
        scene_factor = valid_pts3d.norm(dim=-1).mean().clip(min=1e-8)

        context_pts3d /= scene_factor
        example["context"]["depth"] /= scene_factor
        example["context"]["extrinsics"][:, :3, 3] /= scene_factor

        target_pts3d /= scene_factor
        example["target"]["depth"] /= scene_factor
        example["target"]["extrinsics"][:, :3, 3] /= scene_factor

    example["context"]["pts3d"] = context_pts3d
    example["target"]["pts3d"] = target_pts3d
    example["context"]["valid_mask"] = context_valid_mask * -1
    example["target"]["valid_mask"] = target_valid_mask * -1
    return example
