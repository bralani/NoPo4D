"""Velocity DPT head for predicting Gaussian velocities.

This module implements a DPT head specifically for velocity prediction,
outputting ms3 (linear velocity) and omega (angular velocity) parameters.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.encoder.heads.dpt.dpt_head import DPTHead


class VelocityDPTHead(DPTHead):
    """DPT head for velocity prediction.

    Outputs 9 channels: [du(1), dv(1), dZ(1), omega_cam(4), motion_prob(1), cov_t(1)]
    where du/dv are normalized displacements (fraction of image W/H) from t1 to t2;
    dZ is the camera-space depth displacement (Z2 = Z1 + dZ); motion_prob is a
    per-pixel motion probability (sigmoid).
    Based on DPT_GS_Head but with simplified output for velocity only.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        down_ratio: int = 1
    ) -> None:
        """Initialize the velocity DPT head.

        Args:
            dim_in: Input dimension from backbone.
            patch_size: Patch size of the backbone.
            features: Feature dimension for intermediate representations.
            out_channels: Output channels for each intermediate layer.
            intermediate_layer_idx: Indices of layers used for DPT.
            pos_embed: Whether to use positional embedding.
            down_ratio: Downscaling factor for output resolution.
        """
        # Output 9 channels: du (1) + dv (1) + dZ (1) + omega_cam (4) + motion_prob (1) + cov_t (1)
        output_dim = 9

        super().__init__(
            dim_in=dim_in,
            patch_size=patch_size,
            output_dim=output_dim,
            activation="linear",  # No activation, raw outputs
            conf_activation="expp1",  # Not used but required
            features=features,
            out_channels=out_channels,
            intermediate_layer_idx=intermediate_layer_idx,
            pos_embed=pos_embed,
            down_ratio=down_ratio,
        )

        # Override output conv for velocity prediction
        head_features_1 = features // 2  # Match output from scratch.output_conv1
        head_features_2 = 64

        last_conv = nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(last_conv.weight, mean=0.0, std=1e-2)
        nn.init.normal_(last_conv.bias, mean=0.0, std=1e-2)

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            last_conv,
        )

    def forward(
        self,
        encoder_tokens: List[torch.Tensor],
        depths: torch.Tensor,
        spatial_input: torch.Tensor,
        patch_start_idx: int = 0,
        image_size: tuple = None,
        frames_chunk_size: int = 8,
    ) -> torch.Tensor:
        """Forward pass through the velocity head.

        Args:
            encoder_tokens: List of token tensors from motion encoder [layers, B, S, N, C].
            depths: Depth predictions (unused, for interface compatibility).
            spatial_input: Image tensor kept for interface compatibility.
                The current implementation only uses its shape for chunking.
            patch_start_idx: Starting index for patch tokens.
            image_size: Target image size (H, W).
            frames_chunk_size: Number of frames to process per chunk.

        Returns:
            Motion predictions [B, S, 9, H, W]: [du(1), dv(1), dZ(1), omega_cam(4), motion_prob(1), cov_t(1)].
        """
        B, S, _, H, W = spatial_input.shape
        image_size = image_size if image_size is not None else (H, W)

        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(encoder_tokens, spatial_input, patch_start_idx, image_size)

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0

        all_preds = []
        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            chunk_output = self._forward_impl(
                encoder_tokens, spatial_input, patch_start_idx, image_size,
                frames_start_idx, frames_end_idx
            )
            all_preds.append(chunk_output)

        return torch.cat(all_preds, dim=1)

    def _forward_impl(
        self,
        encoder_tokens: List[torch.Tensor],
        spatial_input: torch.Tensor,
        patch_start_idx: int = 0,
        image_size: tuple = None,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> torch.Tensor:
        """Implementation of the forward pass.

        Args:
            encoder_tokens: List of token tensors. Can be either:
                - A list with one element [motion_tokens] of shape [B, S, N, C]
                - A list matching intermediate_layer_idx with shape [B, S, N, C] each
            spatial_input: Image tensor kept for compatibility; only its shape is used.
            patch_start_idx: Starting index for patch tokens.
            image_size: Target image size (H, W).
            frames_start_idx: Starting index for frames to process.
            frames_end_idx: Ending index for frames to process.

        Returns:
            Motion predictions [B, S, 9, H, W]: [du(1), dv(1), dZ(1), omega_cam(4), motion_prob(1), cov_t(1)].
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            spatial_input = spatial_input[:, frames_start_idx:frames_end_idx]

        B, S, _, H, W = spatial_input.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []

        # Handle single motion token input (from MotionEncoder)
        # Reuse the same features for all DPT layers
        if len(encoder_tokens) == 1:
            motion_tokens = encoder_tokens[0]  # [B, S, N, C]
            if frames_start_idx is not None and frames_end_idx is not None:
                motion_tokens = motion_tokens[:, frames_start_idx:frames_end_idx].contiguous()

            for dpt_idx in range(len(self.intermediate_layer_idx)):
                x = motion_tokens[:, :, patch_start_idx:]
                x = x.reshape(B * S, -1, x.shape[-1])
                x = self.norm(x)
                x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

                x = self.projects[dpt_idx](x)
                if self.pos_embed:
                    x = self._apply_pos_embed(x, W, H)
                x = self.resize_layers[dpt_idx](x)
                out.append(x)
        else:
            # Standard multi-layer input
            for dpt_idx, layer_idx in enumerate(self.intermediate_layer_idx):
                if len(encoder_tokens) > 10:
                    x = encoder_tokens[layer_idx][:, :, patch_start_idx:]
                else:
                    list_idx = self.intermediate_layer_idx.index(layer_idx)
                    x = encoder_tokens[list_idx][:, :, patch_start_idx:]

                # Select frames if processing a chunk
                if frames_start_idx is not None and frames_end_idx is not None:
                    x = x[:, frames_start_idx:frames_end_idx].contiguous()

                x = x.view(B * S, -1, x.shape[-1])
                x = self.norm(x)
                x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

                x = self.projects[dpt_idx](x)
                if self.pos_embed:
                    x = self._apply_pos_embed(x, W, H)
                x = self.resize_layers[dpt_idx](x)
                out.append(x)

        # Fuse features from multiple layers
        out = self.scratch_forward(out)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        out = self.scratch.output_conv2(out)
        out = out.view(B, S, *out.shape[1:])

        return out
