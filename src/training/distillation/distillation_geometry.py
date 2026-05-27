"""Distillation geometry utilities for encoder.

This module handles geometry-focused knowledge distillation operations
and provides helpers to run teacher backbones and extract geometry
predictions (poses, depth, and 3D points).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import List

import torch
from jaxtyping import Float, Bool
from torch import Tensor, nn

import torchvision.transforms as T
from src.training.distillation.types import DistillationGeometryOutput
from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.utils.transform import extri_intri_to_pose_encoding
from src.training.distillation.distillation_base import DistillationBase
from src.utils.geometry import denormalize_intrinsics, normalize_intrinsics
from depth_anything_3.utils.geometry import affine_inverse


class DistillationGeometry(DistillationBase):
    """Base class for distillation geometry utilities.

    This class defines the interface for geometry-focused distillation and
    provides common utilities for freezing modules and moving them between
    devices (CPU/GPU) during teacher inference.
    """

    _CONF_QUANTILE = 0.3  # Keep top 70% most confident predictions

    def __init__(self, intermediate_layer_idx: List[int] | None) -> None:
        """Initialize distillation manager."""
        self._intermediate_layer_idx = intermediate_layer_idx

    @abstractmethod
    def run(
        self,
        image: Float[Tensor, "batch view 3 height width"],
        extrinsics: Float[Tensor, "batch view 4 4"] | None = None,
        intrinsics: Float[Tensor, "batch view 3 3"] | None = None,
    ) -> DistillationGeometryOutput:
        """Run distillation and return distillation outputs."""
        pass

    @staticmethod
    def create(
        intermediate_layer_idx: List[int] | None,
    ) -> DistillationGeometry:
        """Create the DA3 distillation manager."""
        return DistillationManagerDA3(intermediate_layer_idx)

    @staticmethod
    def _compute_conf_mask(
        depth_conf: Float[Tensor, "batch view height width"],
        quantile: float,
    ) -> Bool[Tensor, "batch view height width"]:
        """Compute confidence mask by thresholding at given quantile."""
        conf_threshold = torch.quantile(
            depth_conf.flatten(2, 3), quantile, dim=-1, keepdim=True
        )
        return depth_conf > conf_threshold.unsqueeze(-1)

    @staticmethod
    def _normalize_extrinsics(
        extrinsics: Float[Tensor, "batch view 3 4"],
        scene_scale: Float[Tensor, "batch"],
    ) -> Float[Tensor, "batch view 3 4"]:
        """Normalize extrinsic translation by scene scale.
        
        Converts w2c translation to c2w, normalizes, then converts back to w2c.
        """
        scale = scene_scale.view(-1, 1, 1)
        
        # Build 4x4 homogeneous w2c extrinsic
        extrinsic_w2c = torch.cat([
            extrinsics,
            torch.zeros_like(extrinsics[..., :1, :])
        ], dim=-2)
        extrinsic_w2c[..., 3, 3] = 1.0
        
        # Convert w2c to c2w
        extrinsic_c2w = affine_inverse(extrinsic_w2c)
        translation_c2w = extrinsic_c2w[..., :3, 3]
        
        # Normalize c2w translation
        translation_c2w_normalized = translation_c2w / scale
        
        # Reconstruct normalized c2w extrinsic and convert back to w2c
        extrinsic_c2w_normalized = extrinsic_c2w.clone()
        extrinsic_c2w_normalized[..., :3, 3] = translation_c2w_normalized
        extrinsic_w2c_normalized = affine_inverse(extrinsic_c2w_normalized)
        return extrinsic_w2c_normalized


class DistillationManagerDA3(DistillationGeometry):
    """Distillation geometry manager for Depth Anything 3 backbone."""

    _DA3_DISTILL_MODEL = "depth-anything/DA3-GIANT-1.1"
    NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    PROCESS_RES = 504 # DA3 default processing resolution

    def __init__(self, intermediate_layer_idx: List[int] | None) -> None:
        """Initialize DA3 distillation geometry manager."""
        super().__init__(intermediate_layer_idx)
        self._model = DepthAnything3.from_pretrained(self._DA3_DISTILL_MODEL)
        self._move_to_cpu((self._model,))

    def _resize_longest_side(
        self,
        img: Float[Tensor, "batch view *dims height width"],
        target_size: int,
    ) -> Float[Tensor, "batch view *dims new_height new_width"]:
        """Resize images so that the longest side matches target_size.
        
        Handles both (B, V, C, H, W) and (B, V, H, W) tensor formats.
        """
        # Handle both (B, V, C, H, W) and (B, V, H, W)
        if img.dim() == 5:
            b, v, c, h, w = img.shape
            has_channel = True
        elif img.dim() == 4:
            b, v, h, w = img.shape
            has_channel = False
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {img.dim()}D")
        
        longest = max(h, w)
        
        if longest == target_size:
            return img
        
        scale = target_size / float(longest)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        
        mode = "bicubic" if scale > 1.0 else "area"
        
        if has_channel:
            img_reshaped = img.view(b * v, c, h, w)
            img_resized = torch.nn.functional.interpolate(
                img_reshaped,
                size=(new_h, new_w),
                mode=mode,
                align_corners=False if mode == "bicubic" else None,
            )
            return img_resized.view(b, v, c, new_h, new_w)
        else:
            # Add channel dimension for interpolation
            img_reshaped = img.view(b * v, 1, h, w)
            img_resized = torch.nn.functional.interpolate(
                img_reshaped,
                size=(new_h, new_w),
                mode=mode,
                align_corners=False if mode == "bicubic" else None,
            )
            return img_resized.view(b, v, new_h, new_w)

    def run(
        self,
        image: Float[Tensor, "batch view 3 height width"],
        extrinsics: Float[Tensor, "batch view 4 4"] | None = None,
        intrinsics: Float[Tensor, "batch view 3 3"] | None = None,
    ) -> DistillationGeometryOutput:
        """Run DA3-based distillation using the full model.
        
        This method mirrors the inference flow from DepthAnything3.forward(),
        adapted for batched tensor inputs.
        """
        b, v, c, h_original, w_original = image.shape
        longest_edge = max(h_original, w_original)

        device = image.device
        distill_image = image.clone().detach()
        # Ensure model is on the correct device (required for multi-GPU DDP training)
        model_dev = next(self._model.parameters()).device
        if model_dev != device:
            self._move_to_device(device, (self._model,))

        distill_image = self._resize_longest_side(distill_image, target_size=self.PROCESS_RES)
        b, v, c, h, w = distill_image.shape

        # normalization
        distill_image = distill_image.view(b * v, c, h, w)
        distill_image = self.NORMALIZE(distill_image)
        distill_image = distill_image.view(b, v, c, h, w)

        with torch.no_grad():
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):

                # condition DA3 distillation on provided extrinsics/intrinsics if available
                extrinsics_conditioning = None
                intrinsics_conditioning = None
                if extrinsics is not None and intrinsics is not None:
                    extrinsics_conditioning = affine_inverse(extrinsics)
                    intrinsics_conditioning = denormalize_intrinsics(
                        intrinsics, width=w, height=h
                    )

                raw_output = self._model.model(
                    distill_image,
                    extrinsics=extrinsics_conditioning,
                    intrinsics=intrinsics_conditioning,
                    export_feat_layers=self._intermediate_layer_idx or [],
                    infer_gs=False,
                )

            # Extract outputs from model
            distill_depth_map: Float[Tensor, "batch view h_original w_original"] = self._resize_longest_side(img=raw_output["depth"].squeeze(-1), target_size=longest_edge)
            distill_depth_conf: Float[Tensor, "batch view h_original w_original"] = self._resize_longest_side(raw_output.get("depth_conf"), target_size=longest_edge)
            distill_extrinsic: Float[Tensor, "batch view 4 4"] = raw_output.get("extrinsics")  # w2c
            intrinsics_normalized: Float[Tensor, "batch view 3 3"] = normalize_intrinsics(
                raw_output.get("intrinsics"), width=w, height=h
            )
            distill_intrinsic = denormalize_intrinsics(
                intrinsics_normalized, width=w_original, height=h_original
            )

            # Compute confidence mask
            conf_mask = self._compute_conf_mask(distill_depth_conf, self._CONF_QUANTILE)
            
            # Convert normalized extrinsics to pose encodings
            distill_pred_pose_enc_list = [
                extri_intri_to_pose_encoding(
                    distill_extrinsic,
                    distill_intrinsic,
                    image_size_hw=image.shape[-2:],
                )
            ]

            output: DistillationGeometryOutput = {
                "pred_pose_enc_list": distill_pred_pose_enc_list,
                "depth_map": distill_depth_map,
                "conf_mask": conf_mask,
            }

            # In DA3, we don't move every time between the CPU and GPU (training is faster)
            #self._move_to_cpu((self._model,))
            torch.cuda.empty_cache()

        return output
