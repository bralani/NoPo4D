"""Encoder I/O types: scene input, pose/depth/flow dicts, output dataclass, and abstract Encoder."""

from abc import ABC, abstractmethod
from typing import Generic, TypedDict, TypeVar

import torch.nn as nn
from dataclasses import dataclass
from jaxtyping import Float
from torch import Tensor

from ..types import Gaussians


# Internal token / pose type alias

# 9-D per-view camera encoding produced by the camera head.
# Layout: [tx, ty, tz | qx, qy, qz, qw | fov_h, fov_w]
#   tx / ty / tz       — w2c translation
#   qx / qy / qz / qw  — rotation quaternion
#   fov_h / fov_w      — vertical and horizontal field of view in radians
PoseEncoding = Float[Tensor, "batch view 9"]


@dataclass(frozen=True)
class SceneInput:
    """Input context for the encoder: groups per-scene data passed in from outside."""
    image:            Float[Tensor, "batch view 3 height width"]
    num_cameras:      int
    timestamps:       Float[Tensor, "batch view"] | None = None
    input_extrinsics: Float[Tensor, "batch view 4 4"] | None = None  # w2c
    input_intrinsics: Float[Tensor, "batch view 3 3"] | None = None  # pixel-space


# Typed output dicts used in EncoderOutput

class CameraPoseDict(TypedDict):
    extrinsic_c2w: Float[Tensor, "batch view 4 4"]  # c2w
    extrinsic_w2c: Float[Tensor, "batch view 4 4"]  # w2c
    intrinsic:     Float[Tensor, "batch view 3 3"]  # pixel-space
    encodings:     list[PoseEncoding] | None        # iterative 9-D pose encodings from camera head


class DepthDict(TypedDict):
    depth: Float[Tensor, "batch view height width 1"]          # raw depth values
    depth_conf: Float[Tensor, "batch view height width"]       # raw confidence scores


class OpticalFlowDict(TypedDict):
    # Per-pixel 2D flow predicted by the motion encoder (u, v offsets in pixel space).
    motion_flow_fwd: Float[Tensor, "batch view height width 2"] | None  # frame t to t+1
    motion_flow_bwd: Float[Tensor, "batch view height width 2"] | None  # frame t to t-1

    # Per-pixel confidence of the motion prediction (range [0, 1]).
    motion_prob_fwd: Float[Tensor, "batch view height width"] | None    # confidence for fwd flow
    motion_prob_bwd: Float[Tensor, "batch view height width"] | None    # confidence for bwd flow


@dataclass
class EncoderOutput:
    camera_pose:  CameraPoseDict
    depth:        DepthDict
    gaussians:    Gaussians | None = None
    optical_flow: OpticalFlowDict | None = None


# Abstract encoder base class

T = TypeVar("T")

class Encoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        images: Float[Tensor, "batch view 3 height width"],
    ) -> EncoderOutput:
        raise NotImplementedError()
