"""Random horizontal-flip augmentation shim."""

import torch
from jaxtyping import Float
from torch import Tensor

from ..types import AnyViews, UnbatchedExample


def reflect_extrinsics(
    extrinsics: Float[Tensor, "*batch 4 4"],
) -> Float[Tensor, "*batch 4 4"]:
    """Reflect camera extrinsics across the x-axis."""
    reflect = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    reflect[0, 0] = -1
    return reflect @ extrinsics @ reflect


def reflect_views(views: AnyViews) -> AnyViews:
    """Horizontally flip images and adjust extrinsics accordingly."""
    return {
        **views,
        "image": views["image"].flip(-1),
        "extrinsics": reflect_extrinsics(views["extrinsics"]),
    }


def apply_augmentation_shim(
    example: UnbatchedExample,
    generator: torch.Generator | None = None,
) -> UnbatchedExample:
    """Randomly augment the training images."""
    if torch.rand(tuple(), generator=generator) < 0.5:
        return example

    return UnbatchedExample(
        context=reflect_views(example["context"]),
        target=reflect_views(example["target"]),
        scene=example["scene"],
        num_cameras=example["num_cameras"]
    )
