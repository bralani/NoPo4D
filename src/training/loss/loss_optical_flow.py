"""Optical Flow Consistency Loss for 4D Gaussian velocity supervision."""

import os
import sys
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from dotdict import dotdict
from gsplat import rasterization as gsplat_rasterization
from jaxtyping import Bool, Float
from torch import Tensor
from torchvision.utils import flow_to_image

from depth_anything_3.utils.geometry import affine_inverse
from src.dataset.types import BatchedExample
from src.model.decoder.types import DecoderOutput
from src.model.encoder.types import EncoderOutput
from src.model.types import Gaussians
from src.utils.geometry import homogenize_points, project_camera_space, transform_rigid
from .loss import Loss


@dataclass
class LossOpticalFlowCfg:
    weight: float = 1.0
    loss_type: Literal["flow", "rendering"] = "flow"  # "flow": supervise du/dv; "rendering": rasterize flow

    flow_threshold: float = 2.0    # Ignore flow vectors smaller than this (pixels)
    max_flow: float = 150.0        # Clamp extreme flow magnitudes
    max_pairs: int = 0             # 0 = use all pairs
    warmup_steps: int = 0

    motion_weight: float = 20.0    # Extra weight for moving pixels
    only_moving_pixels: bool = True
    smooth_l1_beta: float = 1.0

    occlusion_aware: bool = True
    occlusion_threshold: float = 30.0  # Cycle error (px) above which pixel is occluded

    weight_sparse_velocities: float = 0.0

    raft_pretrained: str = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"


@dataclass
class LossOpticalFlowCfgWrapper:
    optical_flow: LossOpticalFlowCfg


class LossOpticalFlow(Loss[LossOpticalFlowCfg, LossOpticalFlowCfgWrapper]):
    """Supervises 3D velocity to produce 2D motion matching RAFT optical flow."""

    MIN_DEPTH = 0.05
    MAX_SCALE = 0.05
    MIN_SCALE_CLIPPED = 1e-5
    SUBPIXEL_NOISE = 1.0   # Flow magnitude below this (px) is treated as noise
    RADIUS_CLIP = 0.1      # Minimum Gaussian screen-space radius for rasterization

    def __init__(self, cfg: LossOpticalFlowCfgWrapper) -> None:
        super().__init__(cfg)
        self.flow_net = None
        self._flow_net_device = None

    def _ensure_flow_net(self, device: torch.device) -> None:
        """Lazily load SEA-RAFT onto *device* (no-op if already loaded)."""
        if self.flow_net is not None and self._flow_net_device == device:
            return
        raft_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../sea_raft/core"))
        if raft_path not in sys.path:
            sys.path.insert(0, raft_path)
        from raft import RAFT
        args = dotdict(dict(
            dim=128, radius=4, iters=12, block_dims=[64, 128, 256],
            initial_dim=64, num_blocks=2, pretrain="resnet34",
            use_var=True, var_min=0, var_max=10,
        ))
        self.flow_net = RAFT.from_pretrained(self.cfg.raft_pretrained, args=args)
        self.flow_net.to(device).eval().requires_grad_(False)
        self._flow_net_device = device

    def _collect_pairs(self, B: int, V: int, views_per_cam: int) -> list[tuple[int, int, int]]:
        """Return all consecutive same-camera (b, v1, v2) pairs in both directions.

        Only consecutive views that belong to the same camera are paired — this avoids
        comparing views from different cameras which share no overlapping scene content.
        Both (v, v+1) and (v+1, v) are included so the loop supervises fwd and bwd flow.
        """
        pairs = []
        for b in range(B):
            for v in range(V - 1):
                if v // views_per_cam == (v + 1) // views_per_cam:
                    pairs += [(b, v, v + 1), (b, v + 1, v)]
        return pairs

    @torch.no_grad()
    def _compute_gt_flow(
        self,
        imgs: Float[Tensor, "B V 3 H W"],
        pairs: list[tuple[int, int, int]],
    ) -> Float[Tensor, "P H W 2"]:
        """Run SEA-RAFT on all pairs and return pseudo-GT optical flow.

        Sub-pixel noise (flow magnitude < 1 px) is zeroed out so it does not
        dilute the moving-pixel mask.
        """
        _, _, _, H, W = imgs.shape
        flow = self.compute_optical_flow(
            torch.stack([imgs[b, v1] for b, v1, v2 in pairs]),
            torch.stack([imgs[b, v2] for b, v1, v2 in pairs]),
        ).permute(0, 2, 3, 1)  # [P, H, W, 2]
        flow[flow.norm(dim=-1) < self.SUBPIXEL_NOISE] = 0.0
        return flow

    def _subsample_pairs(
        self,
        pairs: list[tuple[int, int, int]],
        flow_gt: Float[Tensor, "P H W 2"],
    ) -> tuple[list[tuple[int, int, int]], Float[Tensor, "Q H W 2"]]:
        """Subsample pairs proportionally to average motion magnitude.

        Pairs with more motion are more likely to be selected, focusing the
        training budget on frames where velocities are actually visible.
        """
        scores = flow_gt.norm(dim=-1).mean(dim=(-1, -2))
        selected = torch.multinomial(
            scores if scores.sum() > 0 else torch.ones_like(scores),
            self.cfg.max_pairs, replacement=False,
        ).sort().values.tolist()
        return [pairs[i] for i in selected], flow_gt[selected]

    def _build_occlusion_mask(
        self,
        target_flow: Float[Tensor, "H W 2"],
        flow_gt: Float[Tensor, "P H W 2"],
        pair_to_idx: dict[tuple[int, int, int], int],
        b: int, v1: int, v2: int,
    ) -> Bool[Tensor, "H W"] | None:
        """Compute forward-backward cycle-consistency occlusion mask.

        Returns a boolean mask (True = visible) or None if the reverse pair
        is not available. A pixel is marked occluded when the cycle error —
        forward flow warped back by the backward flow — exceeds the threshold.
        """
        rev_i = pair_to_idx.get((b, v2, v1))
        if rev_i is None:
            return None
        cycle_err = (target_flow + self._warp_flow(flow_gt[rev_i], target_flow)).norm(dim=-1)
        return cycle_err <= self.cfg.occlusion_threshold

    def _write_vis_cache(
        self,
        visualization_cache: dict,
        vis_bufs: dict[str, list[Tensor]],
        valid_pairs: int,
    ) -> None:
        """Flush per-view flow visualizations into *visualization_cache*."""
        if valid_pairs == 0:
            return
        for k, suffix in (
            ("gt_fwd",   "gt_vis_fwd"),
            ("pred_fwd", "pred_vis_fwd"),
            ("gt_bwd",   "gt_vis_bwd"),
            ("pred_bwd", "pred_vis_bwd"),
        ):
            visualization_cache[f"optical_flow_{suffix}"] = torch.stack(vis_bufs[k])

    @torch.no_grad()
    def compute_optical_flow(
        self,
        img1: Float[Tensor, "B 3 H W"],
        img2: Float[Tensor, "B 3 H W"],
    ) -> Float[Tensor, "B 2 H W"]:
        """Compute SEA-RAFT optical flow between a batch of image pairs."""
        img1, img2 = img1 * 255.0, img2 * 255.0
        B, _, H, W = img1.shape
        self._ensure_flow_net(img1.device)
        pad_h, pad_w = (8 - H % 8) % 8, (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            img1 = F.pad(img1, (0, pad_w, 0, pad_h), mode="replicate")
            img2 = F.pad(img2, (0, pad_w, 0, pad_h), mode="replicate")
        flow = self.flow_net(img1, img2)["flow"][-1]
        return flow[:, :, :H, :W]

    def _project_to_pixels(self, means: Tensor, w2c: Tensor, K: Tensor) -> Tensor:
        """Project 3D world-space means to 2D pixel coordinates."""
        cam_pts = transform_rigid(homogenize_points(means[None]), w2c[None])
        return project_camera_space(cam_pts[..., :-1], K[None], epsilon=self.MIN_DEPTH).squeeze(0)

    def render_optical_flow(
        self,
        gaussians: Gaussians,
        b_idx: int,
        w2c_t0: Tensor, w2c_t1: Tensor,
        K_t0: Tensor,
        t0: float, t1: float,
        H: int, W: int,
    ) -> Tensor:
        """Rasterize 2D optical flow from Gaussian positions at t0 and t1."""
        device = gaussians.means.device
        tr0 = gaussians.transform_to_target_time(torch.tensor([t0], device=device), batch_idx=b_idx)
        tr1 = gaussians.transform_to_target_time(torch.tensor([t1], device=device), batch_idx=b_idx)
        means_t0, means_t1 = tr0["means"].squeeze(0), tr1["means"].squeeze(0)
        rot_t0, op_t0       = tr0["rotations"].squeeze(0), tr0["opacities"].squeeze(0)

        # Keep only Gaussians in front of both cameras.
        p0 = transform_rigid(homogenize_points(means_t0[None]), w2c_t0[None]).squeeze(0)
        p1 = transform_rigid(homogenize_points(means_t1[None]), w2c_t1[None]).squeeze(0)
        valid = (p0[..., 2] > self.MIN_DEPTH) & (p1[..., 2] > self.MIN_DEPTH)

        means_t0, means_t1 = means_t0[valid], means_t1[valid]
        rot_t0, op_t0 = rot_t0[valid], op_t0[valid]
        scales = gaussians.scales[b_idx][valid].unsqueeze(0).clamp(self.MIN_SCALE_CLIPPED, self.MAX_SCALE)

        uv_t0 = self._project_to_pixels(means_t0, w2c_t0, K_t0)
        uv_t1 = self._project_to_pixels(means_t1, w2c_t0, K_t0)
        flow  = (uv_t1 - uv_t0).clamp(-self.cfg.max_flow, self.cfg.max_flow)

        rendering, alpha, _ = gsplat_rasterization(
            means=means_t0.unsqueeze(0), quats=rot_t0.unsqueeze(0),
            scales=scales, opacities=op_t0.unsqueeze(0), colors=flow.unsqueeze(0),
            viewmats=w2c_t0[None, None], Ks=K_t0[None, None], width=W, height=H,
            sh_degree=None, packed=False, near_plane=self.MIN_DEPTH,
            radius_clip=self.RADIUS_CLIP, rasterize_mode="classic",
        )
        return (rendering / (alpha + 1e-10)).view(H, W, 2)

    def _warp_flow(
        self,
        flow: Float[Tensor, "H W 2"],
        displacement: Float[Tensor, "H W 2"],
    ) -> Float[Tensor, "H W 2"]:
        """Warp *flow* by *displacement* using bilinear sampling."""
        H, W = flow.shape[:2]
        gy, gx = torch.meshgrid(
            torch.arange(H, device=flow.device, dtype=torch.float32),
            torch.arange(W, device=flow.device, dtype=torch.float32),
            indexing="ij",
        )
        grid = torch.stack([
            2.0 * (gx + displacement[..., 0]) / (W - 1) - 1.0,
            2.0 * (gy + displacement[..., 1]) / (H - 1) - 1.0,
        ], dim=-1).unsqueeze(0)
        warped = F.grid_sample(
            flow.permute(2, 0, 1).unsqueeze(0), grid,
            mode="bilinear", padding_mode="border", align_corners=True,
        )
        return warped.squeeze(0).permute(1, 2, 0)

    def _compute_pixel_loss(
        self, pred: Tensor, target: Tensor, mask: Bool[Tensor, "H W"],
    ) -> Tensor:
        """Smooth-L1 flow loss with motion-weighted masking."""
        pixel_loss = F.smooth_l1_loss(pred, target, reduction="none", beta=self.cfg.smooth_l1_beta).sum(-1)

        if self.cfg.only_moving_pixels:
            # Average loss only over pixels with significant motion.
            moving = pixel_loss[mask]
            return moving.mean() if moving.numel() > 0 else pixel_loss.sum() * 0.0
        else:
            # Upweight moving pixels but still average over all pixels.
            weights = torch.ones_like(pixel_loss)
            weights[mask] = self.cfg.motion_weight
            return (weights * pixel_loss).mean()

    def forward(
        self,
        prediction_context: DecoderOutput,
        prediction_target: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int,
        warmup_steps: int = 0,
        encoder_output_context: EncoderOutput | None = None,
        distill_infos=None,
        visualization_cache: dict | None = None,
    ) -> Float[Tensor, ""]:

        zero = gaussians.means.sum() * 0.0  # differentiable zero on the right device
        if global_step < self.cfg.warmup_steps:
            return zero
        if not (gaussians.is_4d() and gaussians.ms3_fwd is not None):
            return zero

        imgs       = batch["context"]["image"]   # [B, V, 3, H, W]
        timestamps = batch["context"].get("timestamp")
        B, V, _, H, W = imgs.shape
        if V < 2 or timestamps is None:
            return zero

        # Build pairs, compute pseudo-GT flow, optionally subsample.
        views_per_cam = V // int(batch["num_cameras"][0])
        pairs = self._collect_pairs(B, V, views_per_cam)
        if not pairs:
            return zero

        flow_gt = self._compute_gt_flow(imgs, pairs)

        if self.cfg.max_pairs > 0 and len(pairs) > self.cfg.max_pairs:
            pairs, flow_gt = self._subsample_pairs(pairs, flow_gt)

        # Reverse-pair lookup for forward-backward occlusion masking.
        pair_to_idx = {(b, v1, v2): j for j, (b, v1, v2) in enumerate(pairs)}

        # Pre-compute rendering inputs if needed.
        if self.cfg.loss_type == "rendering":
            w2c = affine_inverse(encoder_output_context.camera_pose["extrinsic_c2w"])
            Ks  = encoder_output_context.camera_pose["intrinsic"]
        else:
            if encoder_output_context is None or encoder_output_context.optical_flow is None:
                raise RuntimeError('loss_type == "flow" requires encoder_output_context.optical_flow')
            flow_fwd = encoder_output_context.optical_flow.get("motion_flow_fwd")
            flow_bwd = encoder_output_context.optical_flow.get("motion_flow_bwd")
            if flow_fwd is None or flow_bwd is None:
                raise RuntimeError('loss_type == "flow" requires motion_flow_fwd and motion_flow_bwd')

        # Per-pair loss accumulation.
        total_loss, valid_pairs = zero, 0
        do_vis = visualization_cache is not None
        vis_bufs = {k: [torch.zeros(3, H, W, device=imgs.device) for _ in range(V)]
                    for k in ("gt_fwd", "pred_fwd", "gt_bwd", "pred_bwd")} if do_vis else None

        for i, (b, v1, v2) in enumerate(pairs):
            t0, t1      = timestamps[b, v1].item(), timestamps[b, v2].item()
            target_flow = flow_gt[i].clamp(-self.cfg.max_flow, self.cfg.max_flow)
            mask        = target_flow.norm(dim=-1) > self.cfg.flow_threshold

            if self.cfg.occlusion_aware:
                visible = self._build_occlusion_mask(target_flow, flow_gt, pair_to_idx, b, v1, v2)
                if visible is not None:
                    mask = mask & visible

            if not mask.any():
                continue

            if self.cfg.loss_type == "flow":
                pred_flow = (flow_fwd[b, v1] if v1 < v2 else flow_bwd[b, v1]).permute(1, 2, 0)
            else:
                pred_flow = self.render_optical_flow(gaussians, b, w2c[b, v1], w2c[b, v2], Ks[b, v1], t0, t1, H, W)

            total_loss += self._compute_pixel_loss(pred_flow, target_flow, mask)
            valid_pairs += 1

            if do_vis:
                gt_img   = flow_to_image(target_flow.permute(2, 0, 1)).float() / 255.0
                pred_img = flow_to_image(pred_flow.permute(2, 0, 1)).float() / 255.0
                (vis_bufs["gt_fwd"]   if v1 < v2 else vis_bufs["gt_bwd"]  )[v1] = gt_img
                (vis_bufs["pred_fwd"] if v1 < v2 else vis_bufs["pred_bwd"])[v1] = pred_img

        if do_vis:
            self._write_vis_cache(visualization_cache, vis_bufs, valid_pairs)

        # L1 regularization on Gaussian velocities to encourage sparsity.
        sparse_vel_loss = zero
        if self.cfg.weight_sparse_velocities > 0.0:
            vels = [v.norm(dim=-1) for v in (gaussians.ms3_fwd, gaussians.ms3_bwd) if v is not None]
            if vels:
                sparse_vel_loss = self.cfg.weight_sparse_velocities * torch.stack(vels).mean()

        if valid_pairs == 0:
            return sparse_vel_loss
        return self.cfg.weight * (total_loss / valid_pairs) + sparse_vel_loss
