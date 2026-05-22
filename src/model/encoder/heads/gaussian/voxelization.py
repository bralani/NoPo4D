"""Voxelization processing utilities for confidence-weighted Gaussian fusion."""

import torch
from torch import Tensor
from typing import TypedDict
from jaxtyping import Float
try:
    from torch_scatter import scatter_add, scatter_softmax
    TORCH_SCATTER_AVAILABLE = True
except ImportError:
    TORCH_SCATTER_AVAILABLE = False


class VoxelizedBatch(TypedDict):
    feats:  list[Float[Tensor, "num_voxels channels"]]   # fused features per batch element
    pts:    list[Float[Tensor, "num_voxels 3"]]          # fused 3D positions per batch element
    times:  list[Float[Tensor, "num_voxels 1"]]          # fused timestamps per batch element


class VoxelizationProcessor:
    """Confidence-weighted voxel fusion: merges all pixels that fall into the
    same (x, y, z, t) voxel into a single Gaussian, weighted by depth confidence."""

    def __init__(self, voxel_size: float) -> None:
        self.voxel_size = voxel_size

    def voxelize(
        self,
        anchor_feats: Float[Tensor, "batch view raw_gs_dim height width"],
        point_map: Float[Tensor, "batch view height width 3"],
        conf: Float[Tensor, "batch view height width"],
        timestamps: Float[Tensor, "batch view"] | None,
        num_cameras: int,
    ) -> VoxelizedBatch:
        """Voxelize each batch element independently.

        Args:
            anchor_feats: raw Gaussian features,  shape (B, V, C, H, W)
            point_map:      unprojected 3D points,   shape (B, V, H, W, 3)
            conf:         depth confidence scores, shape (B, V, H, W)
            timestamps:   per-view timestamps,     shape (B, V), or None for static scenes
            num_cameras:  number of distinct cameras (used to compute frame count)

        Returns:
            VoxelizedBatch with one list entry per batch element:
                feats  — fused features,     each (N_vox, C)
                pts    — fused 3D positions, each (N_vox, 3)
                times  — fused timestamps,   each (N_vox, 1)
        """
        B = len(anchor_feats)
        result: VoxelizedBatch = {"feats": [], "pts": [], "times": []}

        for i in range(B):
            ts_i  = timestamps[i] if timestamps is not None else None
            pts, t, feats = self._fuse_single_batch(anchor_feats[i], point_map[i], conf[i], ts_i, num_cameras)

            result["pts"].append(pts)
            result["times"].append(t)
            result["feats"].append(feats)

        return result

    def _fuse_single_batch(
        self,
        feats: Float[Tensor, "views channels height width"],
        pts3d: Float[Tensor, "views height width 3"],
        conf: Float[Tensor, "views height width"],
        timestamps: Float[Tensor, "views"] | None,
        num_cameras: int,
    ) -> tuple[
        Float[Tensor, "num_voxels 3"],
        Float[Tensor, "num_voxels 1"],
        Float[Tensor, "num_voxels channels"],
    ]:
        """Fuse one batch: assign every pixel to a voxel, then scatter-aggregate."""
        V, _, H, W = feats.shape
        N = V * H * W
        total_frames = V // num_cameras

        # flatten to pixel list
        pts_flat   = pts3d.reshape(N, 3)
        feats_flat = feats.permute(0, 2, 3, 1).reshape(N, -1)
        conf_flat  = conf.reshape(N)

        # per-pixel timestamps and integer time bins for voxel keying
        if timestamps is not None:
            time_bins      = (timestamps * total_frames).round().long()
            time_bins_flat = time_bins.repeat_interleave(H * W).unsqueeze(-1)
            times_flat     = timestamps.repeat_interleave(H * W).to(pts_flat.dtype).unsqueeze(-1)
        else:
            time_bins_flat = pts_flat.new_zeros(N, 1, dtype=torch.long)
            times_flat     = pts_flat.new_zeros(N, 1)

        # assign each pixel to a (x, y, z, t) voxel
        spatial_bins = (pts_flat / self.voxel_size).round().long()
        _, pixel_to_voxel = torch.unique(
            torch.cat([spatial_bins, time_bins_flat], dim=-1), dim=0, return_inverse=True
        )

        # confidence-weighted average of each pixel's contribution to its voxel
        weights = scatter_softmax(conf_flat, pixel_to_voxel).unsqueeze(-1)

        # aggregate positions/timestamps/features per voxel
        return (
            scatter_add(pts_flat   * weights, pixel_to_voxel, dim=0),
            scatter_add(times_flat * weights, pixel_to_voxel, dim=0),
            scatter_add(feats_flat * weights, pixel_to_voxel, dim=0),
        )
