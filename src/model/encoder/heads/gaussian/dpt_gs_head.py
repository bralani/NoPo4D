from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.encoder.heads.dpt.dpt_head import DPTHead


class DPT_GS_Head(DPTHead):
    """DPT head specialised for predicting raw Gaussian parameters.

    Extends DPTHead with two key differences:
    1. An image shortcut branch (input_merger) that fuses RGB features directly
       into the DPT output, preserving fine-grained appearance detail.
    2. A wider output_conv2 bottleneck (128 channels instead of 32) to handle
       the large Gaussian parameter vector (opacity + scale + rotation + SH + ...).

    Unlike the base class, _forward_impl returns a single raw-parameter tensor
    (B, S, output_dim, H, W) with no activation applied — the GaussianAdapter
    is responsible for interpreting and activating each channel group.
    """

    def __init__(self,
            dim_in: int,
            patch_size: int = 14,
            output_dim: int = 83,
            activation: str = "inv_log",
            conf_activation: str = "expp1",
            features: int = 256,
            out_channels: List[int] = [256, 512, 1024, 1024],
            intermediate_layer_idx: List[int] = [4, 11, 17, 23],
            pos_embed: bool = True,
            down_ratio: int = 1,
    ):
        super().__init__(dim_in, patch_size, output_dim, activation, conf_activation, features, out_channels, intermediate_layer_idx, pos_embed, down_ratio)

        # Wider bottleneck for large Gaussian param vectors (sh=4 → output_dim > 50)
        head_features_2 = 128 if output_dim > 50 else 32
        head_features_1 = 128

        # Encodes raw RGB into the same feature space as scratch output,
        # then added as a residual to inject high-frequency image detail.
        self.input_merger = nn.Sequential(
            nn.Conv2d(3, head_features_1, 7, 1, 3),
            nn.ReLU(),
        )

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(
        self,
        encoder_tokens: List[torch.Tensor],
        depths,
        imgs: torch.Tensor,
        patch_start_idx: int = 5,
        image_size=None,
        conf=None,
        frames_chunk_size: int = 8,
    ) -> torch.Tensor:
        """Run the GS head, optionally chunked along the frame dimension.

        Args:
            encoder_tokens: Aggregated backbone tokens, either a full list
                (length > 10, indexed by layer) or a short pre-selected list.
            depths: Unused — kept for interface compatibility with the depth head.
            imgs: Input images (B, S, 3, H, W).
            patch_start_idx: Token index where patch tokens begin (skips cls/cam tokens).
            image_size: Unused — output is always upsampled to (H, W) of imgs.
            conf: Unused — kept for interface compatibility.
            frames_chunk_size: Max frames processed at once; None means all at once.

        Returns:
            Raw Gaussian parameters (B, S, output_dim, H, W).
        """
        B, S, _, H, W = imgs.shape

        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(encoder_tokens, imgs, patch_start_idx)

        all_preds = []
        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            all_preds.append(self._forward_impl(encoder_tokens, imgs, patch_start_idx, frames_start_idx, frames_end_idx))

        return torch.cat(all_preds, dim=1)

    def _forward_impl(
        self,
        encoder_tokens: List[torch.Tensor],
        imgs: torch.Tensor,
        patch_start_idx: int = 5,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> torch.Tensor:
        """Core forward for a single chunk of frames.

        Runs the DPT multi-scale fusion, then adds an RGB shortcut residual
        before the final conv to preserve high-frequency image detail.

        Returns:
            Raw Gaussian parameters (B, S, output_dim, H, W).
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            imgs = imgs[:, frames_start_idx:frames_end_idx]

        B, S, _, H, W = imgs.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # Extract and reshape tokens from each selected backbone layer
        out = []
        for dpt_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            # Support both a full token list (indexed by layer_idx) and a
            # pre-selected short list (indexed sequentially by dpt_idx)
            if len(encoder_tokens) > 10:
                x = encoder_tokens[layer_idx][:, :, patch_start_idx:]
            else:
                x = encoder_tokens[dpt_idx][:, :, patch_start_idx:]

            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx].contiguous()

            x = x.view(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            out.append(x)

        # Multi-scale DPT fusion → (B*S, features, patch_h, patch_w)
        out = self.scratch_forward(out)

        # RGB shortcut: encode image pixels and add as a residual after upsampling
        direct_img_feat = self.input_merger(imgs.flatten(0, 1))
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        out = out + direct_img_feat

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        out = self.scratch.output_conv2(out)
        return out.view(B, S, *out.shape[1:])
