"""Depth Anything 3 backbone for NoPo4D.

TemporalVisionTransformer extends the DA3 ViT (DinoVisionTransformer) with
sinusoidal timestamp injection.

BackboneDA3 loads the pretrained DA3 model and exposes its sub-modules
(ViT, camera encoder/decoder, DPT depth head) through the Backbone interface.
"""
from dataclasses import dataclass, field
from typing import Literal

import torch
try:
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.model.dinov2.layers.block import Block
    from depth_anything_3.model.dinov2.vision_transformer import DinoVisionTransformer
    from depth_anything_3.model.reference_view_selector import (
        reorder_by_reference,
        restore_original_order,
        select_reference_view,
    )
    from depth_anything_3.utils.constants import THRESH_FOR_REF_SELECTION
except ImportError as e:
    raise ImportError(
        "depth_anything_3 is required for BackboneDA3. "
        "Install it from the Depth-Anything-3 submodule."
    ) from e
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from src.model.encoder.types import DepthDict
from src.utils.temporal import sinusoidal_time_encoding
from .config import BackboneCfg
from .types import Backbone, LayerTokens


@dataclass
class BackboneDA3Cfg(BackboneCfg):
    name: Literal["da3"] = "da3"
    backbone_checkpoint_name: str = "da3-large-1.1"
    intermediate_layer_idx: list[int] = field(default_factory=lambda: [11, 15, 19, 23])
    backbone_temporal_encoding: bool = True
    input_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    input_std:  tuple[float, float, float] = (0.229, 0.224, 0.225)


class TemporalVisionTransformer(DinoVisionTransformer):
    """DinoVisionTransformer extended with sinusoidal timestamp injection before alternating attention layers."""

    @classmethod
    def from_vit(cls, vit: DinoVisionTransformer) -> "TemporalVisionTransformer":
        """Upgrade a DinoVisionTransformer in-place to support temporal timestamp encoding."""
        vit.__class__ = cls
        return vit

    def _insert_time_tokens(self, x: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        B, S, N, C = x.shape
        time_tokens = sinusoidal_time_encoding(timestamps, self.embed_dim).view(B, S, 1, C)
        insert_idx = 1 + (self.num_register_tokens if self.register_tokens is not None else 0)
        return torch.cat([x[:, :, :insert_idx], time_tokens, x[:, :, insert_idx:]], dim=2)

    def _prepare_rope(self, B, S, H, W, device, has_time_token=False):
        pos = pos_nodiff = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=device)
            pos = rearrange(pos, "(b s) n c -> b s n c", b=B)
            pos_nodiff = torch.zeros_like(pos)
            pretrain_special = self.patch_start_idx + self.num_register_tokens
            if pretrain_special > 0:
                pos = pos + pretrain_special
                pos_nodiff = pos_nodiff + pretrain_special
            n_special = pretrain_special + (1 if has_time_token else 0)
            if n_special > 0:
                zeros = torch.zeros(B, S, n_special, 2, device=device, dtype=pos.dtype)
                pos = torch.cat([zeros, pos], dim=2)
                pos_nodiff = torch.cat([torch.zeros_like(zeros), pos_nodiff], dim=2)
        return pos, pos_nodiff

    def _strip_time_token(self, x: torch.Tensor) -> torch.Tensor:
        insert_idx = 1 + (self.num_register_tokens if self.register_tokens is not None else 0)
        return torch.cat([x[:, :, :insert_idx], x[:, :, insert_idx + 1 :]], dim=2)

    def _maybe_select_reference_view(
        self,
        x: torch.Tensor,
        local_x: torch.Tensor,
        kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if (
            self.alt_start == -1
            or x.shape[1] < THRESH_FOR_REF_SELECTION
            or kwargs.get("cam_token", None) is not None
        ):
            return x, local_x, None
        strategy = kwargs.get("ref_view_strategy", "saddle_balanced")
        b_idx = select_reference_view(x, strategy=strategy)
        return reorder_by_reference(x, b_idx), reorder_by_reference(local_x, b_idx), b_idx

    def _get_intermediate_layers_not_chunked(self, x, n=1, export_feat_layers=[], **kwargs):
        B, S, _, H, W = x.shape
        timestamps = kwargs.get("timestamps")
        x = self.prepare_tokens_with_masks(x)
        output, aux_output = [], []
        total_block_len = len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device, has_time_token=False)
        has_time_token = False
        local_x = x
        b_idx = None

        for i, blk in enumerate(self.blocks):
            g_pos, l_pos = (None, None) if (i < self.rope_start or self.rope is None) else (pos_nodiff, pos)

            if i == self.alt_start - 1:
                x, local_x, b_idx = self._maybe_select_reference_view(x, local_x, kwargs)

            if self.alt_start != -1 and i == self.alt_start:
                if kwargs.get("cam_token") is not None:
                    cam_token = kwargs["cam_token"]
                else:
                    ref = self.camera_token[:, :1].expand(B, -1, -1)
                    src = self.camera_token[:, 1:].expand(B, S - 1, -1)
                    cam_token = torch.cat([ref, src], dim=1)
                x[:, :, 0] = cam_token

            if timestamps is not None and self.alt_start != -1 and i == self.alt_start:
                x = self._insert_time_tokens(x, timestamps)
                local_x = self._insert_time_tokens(local_x, timestamps)
                has_time_token = True
                pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device, has_time_token=True)
                g_pos, l_pos = (None, None) if (i < self.rope_start or self.rope is None) else (pos_nodiff, pos)

            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                x = self.process_attention(x, blk, "global", pos=g_pos, attn_mask=kwargs.get("attn_mask"))
            else:
                x = self.process_attention(x, blk, "local", pos=l_pos)
                local_x = x

            if i in blocks_to_take:
                x_out = self._strip_time_token(x) if has_time_token else x
                lx_out = self._strip_time_token(local_x) if has_time_token else local_x
                out_x = torch.cat([lx_out, x_out], dim=-1) if self.cat_token else x_out
                if b_idx is not None:
                    out_x = restore_original_order(out_x, b_idx)
                output.append((out_x[:, :, 0], out_x))
            if i in export_feat_layers:
                aux_output.append(self._strip_time_token(x) if has_time_token else x)
        return output, aux_output


class BackboneDA3(Backbone):
    input_mean: Tensor
    input_std: Tensor

    def __init__(self, cfg: BackboneDA3Cfg) -> None:
        super().__init__(cfg)
        model_full = DepthAnything3.from_pretrained(
            f"depth-anything/{cfg.backbone_checkpoint_name.upper()}"
        )
        TemporalVisionTransformer.from_vit(model_full.model.backbone.pretrained)
        self._aggregator = model_full.model.backbone.to(torch.bfloat16)
        self._camera_head = model_full.model.cam_dec
        self._camera_enc = model_full.model.cam_enc
        self._depth_head = model_full.model.head

        # Register input normalization buffers
        self.register_buffer("input_mean", torch.tensor(cfg.input_mean).view(1, 1, -1, 1, 1))
        self.register_buffer("input_std", torch.tensor(cfg.input_std).view(1, 1, -1, 1, 1))

    def normalize(
        self,
        image: Float[Tensor, "batch view 3 height width"],
    ) -> Float[Tensor, "batch view 3 height width"]:
        mean = self.input_mean.to(image)
        std = self.input_std.to(image)
        return (image - mean) / std

    def aggregator(
        self,
        image: Float[Tensor, "batch view 3 height width"],
        cam_token: Float[Tensor, "batch view embed_dim"] | None,
        timestamps: Float[Tensor, "batch view"] | None,
        num_cameras: int,
    ) -> list[LayerTokens]:
        tokens, _ = self._aggregator(
            image.to(torch.bfloat16),
            cam_token=cam_token,
            export_feat_layers=self.cfg.intermediate_layer_idx,
            timestamps=timestamps if self.cfg.backbone_temporal_encoding else None,
            num_cameras=num_cameras,
        )
        return tokens

    def camera_head(
        self,
        cam_tokens: Float[Tensor, "batch view 1 embed_dim"],
    ) -> Float[Tensor, "batch view 9"]:
        return self._camera_head(cam_tokens)

    def camera_enc(
        self,
        c2w: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        image_size: tuple[int, int],
    ) -> Float[Tensor, "batch view embed_dim"] | None:
        if self._camera_enc is None:
            return None
        return self._camera_enc(c2w, intrinsics, image_size)

    def depth_head(
        self,
        feats: list[LayerTokens],
        image_size: tuple[int, int],
    ) -> DepthDict:
        H, W = image_size
        return self._depth_head(feats=feats, H=H, W=W, patch_start_idx=0)

    @property
    def vit_block(self) -> type:
        return Block
