
"""
CUDA-backed Gaussian splatting decoder.
Wraps gsplat.rasterization to render Gaussian splats into RGB+D images.

Supports both 3D and 4D Gaussians:
- 3D: Standard Gaussian splatting with static Gaussians
- 4D: Temporal Gaussians with motion model, allowing dynamic scenes to be rendered at arbitrary timestamps
"""

from math import sqrt

import torch
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float
from gsplat import rasterization, spherical_harmonics

from ..types import Gaussians
from .config import Decoder4DGSCfg
from .types import Decoder, DecoderOutput

DEFAULT_NEAR_PLANE = 0.001
DEFAULT_RADIUS_CLIP = 0.1


class Decoder4DGS(Decoder[Decoder4DGSCfg]):
    """CUDA-accelerated splatting decoder for Gaussian rendering."""
    background_color: Float[Tensor, "3"]

    def __init__(self, cfg: Decoder4DGSCfg) -> None:
        super().__init__(cfg)
        self.chunk_size = cfg.chunk_size
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _apply_opacity_sh(
        op_chunk: Float[Tensor, "views gaussian"],
        xyz_chunk: Float[Tensor, "views gaussian 3"],
        opacity_sh: Float[Tensor, "gaussian d_sh"],
        chunk_viewmats: Float[Tensor, "views 4 4"],
        sh_degree: int,
        num_views: int,
    ) -> Float[Tensor, "views gaussian"]:
        """Apply view-dependent opacity correction via spherical harmonics."""
        # Compute camera positions in world space
        R = chunk_viewmats[:, :3, :3]
        t_vec = chunk_viewmats[:, :3, 3:]
        cam_pos = -(R.mT @ t_vec).squeeze(-1)                          # (views, 3)

        # Compute unit direction from each camera to each Gaussian center
        dirs = F.normalize(xyz_chunk - cam_pos.unsqueeze(1), dim=-1)   # (views, G, 3)

        # Evaluate opacity SH (broadcast scalar SH to 3-channel format required by gsplat)
        osh_chunk_3 = opacity_sh.unsqueeze(0).expand(num_views, -1, -1, 3)
        sh_val = spherical_harmonics(sh_degree, dirs, osh_chunk_3)[..., 0]  # (views, G)
        return (op_chunk + sh_val).clamp_(0.0, 1.0)

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, 'batch view 4 4'],
        intrinsics: Float[Tensor, 'batch view 3 3'],
        image_shape: tuple[int, int],
        ts: Float[Tensor, 'batch view'] | None = None,
    ) -> DecoderOutput:
        """
        Render images and depths from Gaussian splats using CUDA rasterization.
        Args:
            gaussians: Input Gaussians
            extrinsics: [B, V, 4, 4] camera extrinsics
            intrinsics: [B, V, 3, 3] camera intrinsics (normalized)
            image_shape: (H, W)
            ts: [B, V] Optional timestamps for temporal Gaussians
        Returns:
            DecoderOutput with color, depth, alpha
        """
        B, V, _, _ = intrinsics.shape
        H, W = image_shape
        device = gaussians.means.device
        is_4d = gaussians.is_4d()

        # Prepare static Gaussian attributes shared across views
        scales = gaussians.scales.float().clamp(min=0, max=0.05)
        features = gaussians.harmonics.permute(0, 1, 3, 2).contiguous().float()
        sh_degree = int(sqrt(features.shape[-2])) - 1
        background = self.background_color.to(device).unsqueeze(0)

        # Prepare opacity SH if available
        opacity_sh_features = None
        if gaussians.opacity_sh is not None:
            opacity_sh_features = gaussians.opacity_sh.unsqueeze(-1).float()

        # Prepare camera matrices
        viewmats = extrinsics.float().inverse().unsqueeze(2).contiguous()
        Ks = intrinsics.float().unsqueeze(2).contiguous()

        # Allocate output buffers
        rendered_imgs   = torch.zeros(B, V, 3, H, W, device=device, dtype=torch.float32)
        rendered_depths = torch.zeros(B, V, H, W, device=device, dtype=torch.float32)
        rendered_alphas = torch.zeros(B, V, H, W, device=device, dtype=torch.float32)

        # Main render loop
        for i in range(B):
            scale_chunk = scales[i].unsqueeze(0).expand(self.chunk_size, -1, -1).contiguous().float()
            color_chunk = features[i].unsqueeze(0).expand(self.chunk_size, -1, -1, -1).contiguous().float() # memory intensive operation -> use a small sh_degree or small chunk_size
            background_chunk = background.expand(self.chunk_size, -1, -1).contiguous().float()

            # Chunked rasterization
            for start_idx in range(0, V, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, V)
                current_chunk_size = end_idx - start_idx

                if is_4d and ts is not None:
                    # Dynamic: transform Gaussians to target time for each view
                    transformed = gaussians.transform_to_target_time(
                        batch_idx=i,
                        ts=ts[i,start_idx:end_idx].unsqueeze(0).to(device),
                    )
                    xyz_chunk, rot_chunk, op_chunk = transformed["means"], transformed["rotations"], transformed["opacities"]
                else:
                    # Static: use original Gaussians
                    xyz_chunk = gaussians.means[i].unsqueeze(0).expand(current_chunk_size, -1, -1)
                    rot_chunk = gaussians.rotations[i].unsqueeze(0).expand(current_chunk_size, -1, -1)
                    op_chunk  = gaussians.opacities[i].unsqueeze(0).expand(current_chunk_size, -1)

                # Apply view-dependent opacity SH
                if opacity_sh_features is not None:
                    op_chunk = self._apply_opacity_sh(
                        op_chunk, xyz_chunk, opacity_sh_features[i],
                        viewmats[i, start_idx:end_idx, 0], sh_degree, current_chunk_size,
                    )

                # Rasterize the current chunk
                rendering, alpha, _ = rasterization(
                    means=xyz_chunk.float(),
                    quats=rot_chunk.float(),
                    scales=scale_chunk[:current_chunk_size],
                    opacities=op_chunk.float(),
                    colors=color_chunk[:current_chunk_size],
                    viewmats=viewmats[i, start_idx:end_idx],
                    Ks=Ks[i, start_idx:end_idx],
                    width=W,
                    height=H,
                    sh_degree=sh_degree,
                    render_mode="RGB+D",
                    packed=False,
                    near_plane=DEFAULT_NEAR_PLANE,
                    backgrounds=background_chunk[:current_chunk_size],
                    radius_clip=DEFAULT_RADIUS_CLIP,
                    rasterize_mode="classic",
                )

                # Store results into the main buffers
                rgb, depth = rendering.split([3, 1], dim=-1)
                rendered_imgs[i, start_idx:end_idx] = rgb.squeeze(1).permute(0, 3, 1, 2).clamp(0.0, 1.0)
                rendered_depths[i, start_idx:end_idx] = depth.view(current_chunk_size, H, W)
                rendered_alphas[i, start_idx:end_idx] = alpha.view(current_chunk_size, H, W)

        return DecoderOutput(
            color=rendered_imgs,
            depth=rendered_depths,
            alpha=rendered_alphas,
        )
