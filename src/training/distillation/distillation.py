"""Distillation utilities for encoder."""

from jaxtyping import Float
from torch import Tensor

from src.training.distillation.types import DistillationGeometryOutput
from src.training.distillation.distillation_geometry import DistillationGeometry


class DistillationManager:
    """Manages knowledge distillation operations."""

    distillation_geometry: DistillationGeometry

    def __init__(self, cfg) -> None:
        self.distillation_geometry = DistillationGeometry.create(
            intermediate_layer_idx=cfg.backbone.intermediate_layer_idx
        )

    def run(
        self,
        image: Float[Tensor, "batch view 3 height width"],
        num_cameras: int,
        extrinsics: Float[Tensor, "batch view 4 4"] | None = None,
        intrinsics: Float[Tensor, "batch view 3 3"] | None = None,
    ) -> DistillationGeometryOutput:
        """Run geometry distillation and return outputs."""
        return self.distillation_geometry.run(
            image=image,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )
