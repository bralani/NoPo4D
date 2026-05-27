from typing import TypedDict

from jaxtyping import Bool, Float
from torch import Tensor

from src.model.encoder.types import PoseEncoding


class DistillationGeometryOutput(TypedDict):
    """Output from distillation geometry forward pass."""

    pred_pose_enc_list: list[PoseEncoding]
    depth_map: Float[Tensor, "batch view height width"]
    conf_mask: Bool[Tensor, "batch view height width"]


DistillationOutput = DistillationGeometryOutput
