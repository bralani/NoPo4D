"""Motion encoder using self-attention blocks.

This module implements motion encoding by concatenating consecutive frame
pairs and processing them with plain self-attention. A learnable temporal
embedding encodes frame positions. Processed tokens are shared for both
forward and backward velocity prediction.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from jaxtyping import Float

from einops import rearrange
from src.model.encoder.backbone.types import LayerTokens
from src.model.encoder.heads.motion.velocity_dpt_head import VelocityDPTHead
from src.model.encoder.types import CameraPoseDict, DepthDict, OpticalFlowDict, SceneInput


@dataclass
class MotionEncoderCfg:
    num_heads: int = 16
    num_layers: int = 1
    mlp_ratio: float = 4.0
    use_checkpoint: bool = True
    velocity_space: Literal["camera", "world"] = "camera"
    skip_motion_branch: bool = False
    use_motion_prob: bool = True


class MotionEncoder(nn.Module):
    """Motion encoding with self-attention and multi-scale DPT head."""

    def __init__(
        self,
        cfg: MotionEncoderCfg,
        embed_dim: int,
        num_export_layers: int,
        patch_size: int,
        vit_block: type[nn.Module],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_export_layers = num_export_layers
        self.use_checkpoint = cfg.use_checkpoint
        self.skip_motion_branch = cfg.skip_motion_branch

        # Independent self-attention block stack per token layer
        self.layer_blocks = nn.ModuleList([
            nn.ModuleList([
                vit_block(dim=embed_dim, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio)
                for _ in range(cfg.num_layers)
            ])
            for _ in range(num_export_layers)
        ])

        # Learnable temporal embedding: index 0 = current, index 1 = neighbour
        self.temporal_embed = nn.Parameter(torch.zeros(2, embed_dim))
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)

        # Velocity DPT head for predicting per-pixel motion from processed tokens.
        self.velocity_space = cfg.velocity_space
        self.velocity_head = VelocityDPTHead(
            dim_in=embed_dim,
            patch_size=patch_size,
        )

    def forward(
        self,
        scene_input: SceneInput,
        layers_tokens: list[LayerTokens],
        camera_pose: CameraPoseDict,
        depth: DepthDict,
    ) -> tuple[
        Float[Tensor, "batch view 13 height width"],
        OpticalFlowDict
    ]:
        """Predict per-pixel motion from multi-scale backbone tokens.

        Args:
            scene_input:   Input images, timestamps, and camera layout.
            layers_tokens: Multi-scale backbone token list, one entry per exported layer.
            camera_pose:   Extrinsics (w2c) and intrinsics for each view.
            depth:         Depth map from the depth head; used to unproject pixel displacements.

        Returns:
            motion:       Per-pixel motion tensor [B, V, 13, H, W].
                          Layout: [cov_t(1), vel_fwd(3), vel_bwd(3), omega_fwd(3), omega_bwd(3)].
            optical_flow: Dict with keys motion_flow_fwd/bwd [B,V,2,H,W] and
                          motion_prob_fwd/bwd [B,V,H,W].
        """
        image_size = scene_input.image.shape[-2:]
        B, V, *_ = scene_input.image.shape
        F = V // scene_input.num_cameras

        # No temporal motion when timestamps are absent or sequence is a single frame.
        if scene_input.timestamps is None or F == 1:
            S  = layers_tokens[0][0].shape[1]
            device, dtype = scene_input.image.device, scene_input.image.dtype
            zero_motion = torch.zeros((B, S, 13, *image_size), device=device, dtype=dtype)
            zero_flow   = torch.zeros((B, V, 2,  *image_size), device=device, dtype=dtype)
            zero_prob   = torch.zeros((B, V,     *image_size), device=device, dtype=dtype)
            return zero_motion, {
                "motion_flow_fwd": zero_flow, "motion_flow_bwd": zero_flow,
                "motion_prob_fwd": zero_prob, "motion_prob_bwd": zero_prob,
            }

        # Run self-attention motion blocks on each exported token layer.
        fwd_processed, bwd_processed = [], []
        for layer_idx in range(self.num_export_layers):
            tokens = layers_tokens[layer_idx][0]  # [B, S, N, C]
            if self.skip_motion_branch:
                fwd_processed.append(tokens)
                bwd_processed.append(tokens)
            else:
                fwd, bwd = self._run_motion_blocks(tokens, scene_input.num_cameras, layer_idx)
                fwd_processed.append(fwd)
                bwd_processed.append(bwd)

        raw_fwd = self.velocity_head(fwd_processed, None, scene_input.image, 0, image_size)
        raw_bwd = self.velocity_head(bwd_processed, None, scene_input.image, 0, image_size)

        # Time deltas: pad boundaries by repeating the last/first interval.
        ts   = scene_input.timestamps.view(B, scene_input.num_cameras, F)
        diff = ts[:, :, 1:] - ts[:, :, :-1]  # [B, C, F-1]
        dt_fwd = torch.cat([diff, diff[:, :, -1:]], dim=2).view(B, V)
        dt_bwd = torch.cat([diff[:, :, :1], diff],  dim=2).view(B, V)

        vel_fwd, omega_fwd, cov_t_fwd, flow_fwd, prob_fwd = self._compute_velocity(raw_fwd, depth["depth"], camera_pose, dt_fwd)
        vel_bwd, omega_bwd, cov_t_bwd, flow_bwd, prob_bwd = self._compute_velocity(raw_bwd, depth["depth"], camera_pose, dt_bwd)

        optical_flow: OpticalFlowDict = {
            "motion_flow_fwd": flow_fwd, "motion_flow_bwd": flow_bwd,
            "motion_prob_fwd": prob_fwd, "motion_prob_bwd": prob_bwd,
        }

        cov_t = (cov_t_fwd + cov_t_bwd) / 2

        # layout: [cov_t(1), vel_fwd(3), vel_bwd(3), omega_fwd(3), omega_bwd(3)]
        motion = torch.cat([
            cov_t.unsqueeze(2),
            vel_fwd,
            vel_bwd,
            omega_fwd,
            omega_bwd,
        ], dim=2)
        return motion, optical_flow

    def _compute_velocity(
        self,
        raw_preds: Float[Tensor, "batch view 9 height width"],
        Z_1: Float[Tensor, "batch view height width"],
        camera_pose: CameraPoseDict,
        dt: Float[Tensor, "batch view"],
    ) -> tuple[
        Float[Tensor, "batch view 3 height width"],
        Float[Tensor, "batch view 3 height width"],
        Float[Tensor, "batch view height width"],
        Float[Tensor, "batch view 2 height width"],
        Float[Tensor, "batch view height width"],
    ]:
        """Convert raw head predictions to world-space velocity.

        Channels in raw_preds: [du, dv, dZ, omega(4), motion_prob, cov_t].

        Returns:
            velocity:    [B, V, 3, H, W] world-space velocity (xyz).
            omega:       [B, V, 3, H, W] angular velocity channels (first 3 of 4 used).
            cov_t:       [B, V, H, W]    temporal covariance.
            motion_flow: [B, V, 2, H, W] pixel-space flow.
            motion_prob: [B, V, H, W]    per-pixel motion probability.
        """
        _, V, _, H, W = raw_preds.shape

        # Invert w2c rotation to get camera-to-world rotation matrix
        R_c2w       = camera_pose["extrinsic_w2c"][:, :, :3, :3].transpose(-1, -2)
        motion_prob = torch.sigmoid(raw_preds[:, :, 7]) if self.cfg.use_motion_prob else torch.ones_like(raw_preds[:, :, 7])
        omega       = raw_preds[:, :, 3:6]
        cov_t       = dt[:, :, None, None].repeat(1, 1, H, W) + raw_preds[:, :, 8]

        if self.velocity_space == "world":
            # Channels 0-2 are already world-space in this case
            velocity   = raw_preds[:, :, 0:3]
            dummy_flow = torch.zeros_like(raw_preds[:, :, :2])
            return velocity, omega, cov_t, dummy_flow, motion_prob

        # tanh keeps displacements bounded: du in [-W, W], dv in [-H, H], dZ in [-1, 1]
        du = torch.tanh(raw_preds[:, :, 0]) * W * motion_prob
        dv = torch.tanh(raw_preds[:, :, 1]) * H * motion_prob
        dZ = torch.tanh(raw_preds[:, :, 2]) * motion_prob
        motion_flow = torch.stack([du, dv], dim=2)

        # Pixel coordinate grid for frame t1
        v_grid, u_grid = torch.meshgrid(
            torch.arange(H, device=raw_preds.device, dtype=raw_preds.dtype),
            torch.arange(W, device=raw_preds.device, dtype=raw_preds.dtype),
            indexing='ij',
        )

        intrinsic = camera_pose["intrinsic"]
        fx = intrinsic[:, :, 0, 0, None, None]
        fy = intrinsic[:, :, 1, 1, None, None]
        cx = intrinsic[:, :, 0, 2, None, None]
        cy = intrinsic[:, :, 1, 2, None, None]

        # Unproject t1 and t2 pixels to camera space, then take the difference
        Z2 = (Z_1 + dZ).clamp(min=1e-6)
        dX = ((u_grid + du - cx) * Z2 - (u_grid - cx) * Z_1) / fx
        dY = ((v_grid + dv - cy) * Z2 - (v_grid - cy) * Z_1) / fy
        disp_cam = torch.stack([dX, dY, Z2 - Z_1], dim=-1)  # [B, V, H, W, 3]

        # Rotate displacement to world space and divide by dt to get velocity
        disp_world = torch.einsum('bvij,bvhwj->bvhwi', R_c2w, disp_cam)
        velocity = disp_world / dt[:, :, None, None, None].clamp(min=1e-6)
        return velocity.permute(0, 1, 4, 2, 3), omega, cov_t, motion_flow, motion_prob

    def _run_motion_blocks(
        self,
        patch_tokens: Float[Tensor, "batch seq tokens dim"],
        num_cameras: int,
        layer_idx: int,
    ) -> tuple[
        Float[Tensor, "batch seq tokens dim"],      # forward-motion tokens
        Float[Tensor, "batch seq tokens dim"],      # backward-motion tokens
    ]:
        """Run self-attention blocks on paired tokens to produce forward and backward motion tokens.

        Returns:
            fwd_tokens: Forward-motion tokens [B, S, N, D].
            bwd_tokens: Backward-motion tokens [B, S, N, D].
        """
        B, S, N, D = patch_tokens.shape
        C = num_cameras
        F = S // C  # frames per camera
        P = F - 1   # consecutive frame pairs

        # For each of the P consecutive pairs, concatenate curr and nxt frame tokens
        # across all cameras and attend jointly: [(B*P), 2*C*N, D]
        tokens = patch_tokens.view(B, C, F, N, D)
        curr = rearrange(tokens[:, :, :P], 'b c p n d -> (b p) (c n) d') + self.temporal_embed[0]
        nxt  = rearrange(tokens[:, :, 1:], 'b c p n d -> (b p) (c n) d') + self.temporal_embed[1]
        pairs = torch.cat([curr, nxt], dim=1)

        for block in self.layer_blocks[layer_idx]:
            if self.use_checkpoint:
                pairs = checkpoint(block, pairs, use_reentrant=False)
            else:
                pairs = block(pairs)
        assert pairs is not None, "Motion blocks returned None"

        # Split processed pairs back into curr/nxt halves
        curr_proc, nxt_proc = pairs.split(C * N, dim=1)

        # fwd[t] = curr tokens from pair (t, t+1); last frame has no next
        # bwd[t] = nxt  tokens from pair (t-1, t); first frame has no prev
        fwd = torch.zeros(B, C, F, N, D, device=patch_tokens.device, dtype=patch_tokens.dtype)
        bwd = torch.zeros(B, C, F, N, D, device=patch_tokens.device, dtype=patch_tokens.dtype)
        fwd[:, :, :P] = rearrange(curr_proc, '(b p) (c n) d -> b c p n d', b=B, c=C)
        bwd[:, :, 1:] = rearrange(nxt_proc,  '(b p) (c n) d -> b c p n d', b=B, c=C)

        return fwd.view(B, S, N, D), bwd.view(B, S, N, D)
