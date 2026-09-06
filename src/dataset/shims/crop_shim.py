"""Rescale-and-crop shim applied to image/intrinsic pairs."""

import random

import torch
from jaxtyping import Float
from torch import Tensor
import torchvision.transforms.functional as F

from ..types import AnyViews, UnbatchedExample


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],
    Float[Tensor, "*#batch 3 3"],
]:
    """Crop images to shape from center and adjust intrinsics."""
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    images = images[..., :, row : row + h_out, col : col + w_out]

    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    intr_aug: bool = False,
    scale_range: tuple[float, float] = (0.77, 1.0),
    scale: float | None = None,
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],
    Float[Tensor, "*#batch 3 3"],
]:
    """Resize images to fill shape, optionally with random scale augmentation."""
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    if intr_aug:
        if scale is None:
            scale = random.uniform(*scale_range)
        h_scale = round(h_out * scale)
        w_scale = round(w_out * scale)
    else:
        h_scale = h_out
        w_scale = w_out

    scale_factor = max(h_scale / h_in, w_scale / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_scale or w_scaled == w_scale

    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = F.resize(images, (h_scaled, w_scaled), interpolation=F.InterpolationMode.BILINEAR, antialias=False)
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    images, intrinsics = center_crop(images, intrinsics, (h_scale, w_scale))

    if intr_aug:
        images = F.resize(images, size=(h_out, w_out))

    return images, intrinsics


def apply_crop_shim_to_views(
    views: AnyViews,
    shape: tuple[int, int],
    intr_aug: bool = False,
    scale: float | None = None,
) -> AnyViews:
    """Apply rescale-and-crop to a views dict, updating image and intrinsics."""
    images, intrinsics = rescale_and_crop(
        views["image"], views["intrinsics"], shape, intr_aug=intr_aug, scale=scale
    )
    return {
        **views,
        "image": images,
        "intrinsics": intrinsics,
    }


def apply_crop_shim(
    example: UnbatchedExample,
    shape: tuple[int, int],
    intr_aug: bool = False,
    scale_range: tuple[float, float] = (0.77, 1.0),
) -> UnbatchedExample:
    """Crop images in the example."""
    scale = random.uniform(*scale_range) if intr_aug else None
    context_cropped = apply_crop_shim_to_views(example["context"], shape, intr_aug, scale)
    target_cropped = apply_crop_shim_to_views(example["target"], shape, intr_aug, scale)

    return UnbatchedExample(
        context=context_cropped,
        target=target_cropped,
        scene=example["scene"],
        num_cameras=example["num_cameras"]
    )
