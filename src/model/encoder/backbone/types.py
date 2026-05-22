"""Backbone abstraction: token types and abstract Backbone interface.

Any new backbone must subclass BackboneCfg (in config.py) and Backbone here.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from src.model.encoder.types import DepthDict
from src.model.encoder.backbone.config import BackboneCfg


# Per-layer token pair returned by the backbone aggregator.
#   patch_tokens: one feature vector per ViT image patch
#   cam_token:    a single learned camera token per view
LayerTokens = tuple[
    Float[Tensor, "batch view num_patches embed_dim"],  # patch tokens
    Float[Tensor, "batch view 1 embed_dim"],            # camera token
]


class Backbone(nn.Module, ABC):

    def __init__(self, cfg: BackboneCfg) -> None:
        super().__init__()
        self.cfg = cfg

    @property
    def num_export_layers(self) -> int:
        return len(self.cfg.intermediate_layer_idx)

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    @abstractmethod
    def normalize(
        self,
        image: Float[Tensor, "batch view 3 height width"],
    ) -> Float[Tensor, "batch view 3 height width"]: ...

    @abstractmethod
    def aggregator(
        self,
        image: Float[Tensor, "batch view 3 height width"],
        cam_token: Float[Tensor, "batch view embed_dim"] | None,
        timestamps: Float[Tensor, "batch view"] | None,
        num_cameras: int,
    ) -> list[LayerTokens]: ...

    @abstractmethod
    def camera_head(
        self,
        cam_tokens: Float[Tensor, "batch view 1 embed_dim"],
    ) -> Float[Tensor, "batch view 9"]: ...

    @abstractmethod
    def camera_enc(
        self,
        c2w: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        image_size: tuple[int, int],
    ) -> Float[Tensor, "batch view embed_dim"] | None: ...

    @abstractmethod
    def depth_head(
        self,
        feats: list[LayerTokens],
        image_size: tuple[int, int],
    ) -> DepthDict: ...

    @property
    @abstractmethod
    def vit_block(self) -> type[nn.Module]:
        """Return the transformer Block class used by this backbone."""
        ...
