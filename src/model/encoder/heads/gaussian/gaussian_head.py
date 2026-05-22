"""Gaussian parameter heads for NoPo4D.

GaussianAdapter: activates raw network outputs into a Gaussians dataclass.
GaussianHead: runs the DPT-based parameter head, optionally injects motion encoder
    velocity and assembles the final Gaussians.
"""
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor

from src.model.encoder.heads.gaussian.dpt_gs_head import DPT_GS_Head
from src.model.encoder.heads.gaussian.voxelization import VoxelizationProcessor, TORCH_SCATTER_AVAILABLE
from src.model.types import Gaussians
from src.utils.geometry import batchify_unproject_depth_map_to_point_map, normalize_intrinsics
from src.utils.misc import pad_tensor_list, scaled_sigmoid
from src.utils.temporal import dt_to_cov_t

from src.model.encoder.backbone.types import LayerTokens
from src.model.encoder.types import CameraPoseDict, DepthDict, SceneInput


@dataclass
class GaussianHeadCfg:
    dim_in: int = 2048
    patch_size: int = 14
    feature_dim: int = 256
    use_voxelization: bool = False
    voxel_size: float = 0.002


@dataclass
class GaussianAdapterCfg:
    """Configuration for the Gaussian adapter."""
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int
    is_4d: bool
    max_dxyz: float
    min_dxyz: float
    dxyz_bias: float
    use_opacity_sh: bool = True


class GaussianAdapter(nn.Module):
    """Convert raw network outputs into a Gaussians dataclass."""

    cfg: GaussianAdapterCfg
    sh_mask: Tensor
    opacity_sh_mask: Tensor
    VEL_PARAMS_SPLIT: ClassVar[list[int]] = [1, 3, 3, 3, 3]

    @staticmethod
    def get_no_vel_split(sh_degree: int) -> list[int]:
        """Channel split for non-velocity params: scale, rotation, harmonics, dxyz, opacity_sh."""
        d_sh = (sh_degree + 1) ** 2
        return [3, 4, 3 * d_sh, 3, d_sh]

    def __init__(self, cfg: GaussianAdapterCfg) -> None:
        super().__init__()
        self.cfg = cfg

        # SH masks: DC-biased init, higher-degree bands attenuated
        self.register_buffer("sh_mask", torch.ones(self.d_sh, dtype=torch.float32), persistent=False)
        for deg in range(1, cfg.sh_degree + 1):
            self.sh_mask[deg**2 : (deg + 1)**2] = 0.1 * 0.25 ** deg

        self.register_buffer("opacity_sh_mask", torch.zeros(self.d_sh, dtype=torch.float32), persistent=False)
        self.opacity_sh_mask[0] = 0.1
        for deg in range(1, cfg.sh_degree + 1):
            self.opacity_sh_mask[deg**2 : (deg + 1)**2] = 0.1 * 0.25 ** deg

        vel_split = GaussianAdapter.VEL_PARAMS_SPLIT if cfg.is_4d else []
        self.gs_params_split = GaussianAdapter.get_no_vel_split(cfg.sh_degree) + vel_split

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        """Total input dim including velocity channels."""
        return sum(self.gs_params_split)

    @property
    def d_in_no_vel(self) -> int:
        """Input dim without velocity: opacity + gaussian params + confidence."""
        return 1 + sum(GaussianAdapter.get_no_vel_split(self.cfg.sh_degree)) + 1

    def forward(
        self,
        means: Float[Tensor, "batch gaussian 3"],
        raw_gaussians: Float[Tensor, "batch gaussian _"],
        eps: float = 1e-8,
        ts: Float[Tensor, "batch view h w"] | None = None,
    ) -> Gaussians:
        """Activate raw Gaussian parameters and assemble a Gaussians dataclass.

        Args:
            means:         3D Gaussian centers in world space.
            raw_gaussians: Flat parameter vector (opacity, scale, rotation, SH, dxyz,
                           opacity_sh, and optionally cov_t, ms3_fwd/bwd, omega_fwd/bwd).
            eps:           Small epsilon added to rotation norm for numerical stability.
            ts:            Per-Gaussian timestamps; None for static (3D) scenes.

        Returns:
            Gaussians dataclass with all attributes activated and assembled.
        """
        # Split into opacity and per-attribute groups
        opacities = raw_gaussians[..., 0].sigmoid()
        params = raw_gaussians[..., 1:].split(self.gs_params_split, dim=-1)
        scales, rotations, sh, dxyz, opacity_sh = params[:5]

        # 4D temporal parameters (None in static mode)
        if self.cfg.is_4d:
            cov_t, ms3_fwd, ms3_bwd, omega_fwd, omega_bwd = params[5:]
            cov_t = cov_t.squeeze(-1)
        else:
            cov_t = ms3_fwd = ms3_bwd = omega_fwd = omega_bwd = None

        # Position residual
        if dxyz is not None:
            dxyz = scaled_sigmoid(dxyz, self.cfg.dxyz_bias, self.cfg.min_dxyz, self.cfg.max_dxyz)
            means = means + dxyz

        # Temporal covariance activation
        if cov_t is not None:
            cov_t = dt_to_cov_t(cov_t)

        # Normalize rotations and apply SH masks
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3) * self.sh_mask

        return Gaussians(
            means=means.float(),
            harmonics=sh.float(),
            opacities=opacities.float(),
            opacity_sh=(opacity_sh * self.opacity_sh_mask).float() if self.cfg.use_opacity_sh else None,
            scales=scales.float(),
            rotations=rotations.float(),
            t=ts,
            ms3_fwd=ms3_fwd,
            ms3_bwd=ms3_bwd,
            omega_fwd=omega_fwd,
            omega_bwd=omega_bwd,
            cov_t=cov_t,
        )

    def rescale_scales(
        self,
        raw_scales: Float[Tensor, "batch view 3 height width"],
        depth_map: Float[Tensor, "batch view height width"],
        intrinsics: Float[Tensor, "batch view 3 3"],
    ) -> Float[Tensor, "batch view 3 height width"]:
        """Activate raw scale channels and convert to metric world-space values."""
        _, _, h, w = depth_map.shape
        pixel_size = 1 / torch.tensor((w, h), dtype=raw_scales.dtype, device=raw_scales.device)
        intr_inv   = normalize_intrinsics(intrinsics, w, h)[..., :2, :2].inverse()
        multiplier = (0.1 * einsum(intr_inv, pixel_size, "... i j, j -> ... i")).sum(dim=-1)

        scale_min, scale_max = self.cfg.gaussian_scale_min, self.cfg.gaussian_scale_max
        return (
            (scale_min + (scale_max - scale_min) * raw_scales.sigmoid())
            * rearrange(depth_map,  "b v h w -> b v 1 h w")
            * rearrange(multiplier, "b v -> b v 1 1 1")
        )


class GaussianHead(nn.Module):
    """Predict per-pixel Gaussian parameters from multi-scale backbone tokens.

    Runs a DPT-based head to produce raw parameters, optionally injects motion
    encoder velocity, then flattens (or voxelizes) pixels and delegates activation to GaussianAdapter.
    """

    def __init__(self, cfg: GaussianHeadCfg, head_out_dim: int, gaussian_adapter: GaussianAdapter):
        super().__init__()
        self.cfg = cfg
        self.gaussian_adapter = gaussian_adapter
        self.gaussian_param_head = DPT_GS_Head(
            dim_in=cfg.dim_in,
            patch_size=cfg.patch_size,
            output_dim=head_out_dim,
            activation="norm_exp",
            conf_activation="expp1",
            features=cfg.feature_dim,
        )

        use_voxelization = cfg.use_voxelization and TORCH_SCATTER_AVAILABLE
        self.voxelizer = VoxelizationProcessor(voxel_size=cfg.voxel_size) if use_voxelization else None

    def forward(
        self,
        scene_input: SceneInput,
        layers_tokens: list[LayerTokens],
        camera_pose: CameraPoseDict,
        depth: DepthDict,
        velocity: Float[Tensor, "batch view vel_dim height width"] | None = None,
    ) -> Gaussians:
        """Run the Gaussian parameter head and return assembled Gaussians.

        Args:
            scene_input:   Input images, timestamps, and camera layout.
            layers_tokens: Multi-scale backbone token list, one entry per exported layer.
            camera_pose:   Extrinsics (w2c) and intrinsics for each view.
            depth:         Depth map from the depth head.
            velocity:      Optional motion encoder output; injected before the confidence channel.

        Returns:
            Gaussians assembled from the predicted and activated parameters.
        """
        h, w = scene_input.image.shape[-2:]
        point_map = batchify_unproject_depth_map_to_point_map(
            depth["depth"], camera_pose["extrinsic_w2c"], camera_pose["intrinsic"]
        )

        raw_params = self.gaussian_param_head(
            [feat[0] for feat in layers_tokens], # patch tokens from each extracted layer
            point_map.flatten(0, 1).permute(0, 3, 1, 2),
            scene_input.image,
            patch_start_idx=0,
            image_size=(h, w),
        )

        # Inject velocity between static params and confidence channel
        if velocity is not None:
            raw_params = torch.cat([raw_params[:, :, :-1], velocity, raw_params[:, :, -1:]], dim=2)

        # rescaling scales before voxelization
        world_scales = self.gaussian_adapter.rescale_scales(raw_params[:, :, 1:4], depth["depth"], camera_pose["intrinsic"])
        rescaled_params = torch.cat([raw_params[:, :, :1], world_scales, raw_params[:, :, 4:]], dim=2)

        # voxelize or flatten into Gaussian features, means, and timestamps
        gs_feats, gs_means, gs_times = self._prepare_gaussians(
            rescaled_params[:, :, :-1], point_map,
            rescaled_params[:, :, -1], scene_input.timestamps, scene_input.num_cameras,
        )

        return self.gaussian_adapter(means=gs_means, raw_gaussians=gs_feats, ts=gs_times)

    def _prepare_gaussians(
        self,
        anchor_feats: Float[Tensor, "batch view raw_gs_dim height width"],
        point_map: Float[Tensor, "batch view height width 3"],
        conf: Float[Tensor, "batch view height width"],
        timestamps: Float[Tensor, "batch view"] | None = None,
        num_cameras: int = 1,
    ) -> tuple[
        Float[Tensor, "batch num_gaussians raw_gs_dim"],
        Float[Tensor, "batch num_gaussians 3"],
        Float[Tensor, "batch num_gaussians"] | None,
    ]:
        """Flatten all pixels (or voxelize) into a padded Gaussian set.

        Returns times as [batch, num_gaussians] or None when timestamps are absent.
        """
        b, v, h, w, _ = point_map.shape
        num_gs = v * h * w

        if self.voxelizer is not None:
            out = self.voxelizer.voxelize(anchor_feats, point_map, conf, timestamps, num_cameras)
            feats_list, pts_list, times_list = out["feats"], out["pts"], out["times"]

            max_gs = max(f.shape[0] for f in feats_list)
            return (
                pad_tensor_list(feats_list, (max_gs,), value=-1e10),
                pad_tensor_list(pts_list,   (max_gs,), value=-1e4),
                pad_tensor_list(times_list, (max_gs,), value=-1e4).squeeze(-1) if timestamps is not None else None
            )

        feats_out = anchor_feats.permute(0, 1, 3, 4, 2).reshape(b, num_gs, -1)
        pts_out   = point_map.reshape(b, num_gs, 3)
        times_out = None

        # Broadcast per-view timestamps to all pixels, then flatten
        if timestamps is not None:
            times_out = timestamps.view(b, v, 1, 1).expand(b, v, h, w).reshape(b, num_gs)

        return feats_out, pts_out, times_out

