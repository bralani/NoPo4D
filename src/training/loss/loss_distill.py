from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from src.utils.geometry import mat_to_quat
from src.model.encoder.types import EncoderOutput
from src.training.distillation.types import DistillationOutput
from src.dataset.types import AnyExample
from .utils import _normal_map_loss

# Exponential decay factor applied per pose refinement iteration.
_POSE_LOSS_DECAY_GAMMA: float = 0.6


def extri_intri_to_pose_encoding(
    extrinsics,
    intrinsics,
    image_size_hw=None,  # e.g., (256, 512)
    pose_encoding_type="absT_quaR_FoV",
):
    """Convert camera extrinsics and intrinsics to a compact pose encoding.

    Args:
        extrinsics: Camera extrinsic parameters [B, S, 3, 4] (w2c, OpenCV convention).
        intrinsics: Camera intrinsic parameters [B, S, 3, 3] in pixels.
        image_size_hw: (height, width) used for FoV computation.
        pose_encoding_type: Encoding format; only "absT_quaR_FoV" is supported.
    Returns:
        Pose encoding tensor [B, S, 9]: (translation[3], quaternion[4], fov[2]).
    """
    if pose_encoding_type != "absT_quaR_FoV":
        raise NotImplementedError

    R = extrinsics[:, :, :3, :3]
    T = extrinsics[:, :, :3, 3]
    quat  = mat_to_quat(R)
    fov_h = 2 * torch.atan(0.5 / intrinsics[..., 1, 1])
    fov_w = 2 * torch.atan(0.5 / intrinsics[..., 0, 0])
    return torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()


def huber_loss(x: Tensor, y: Tensor, delta: float = 1.0) -> Tensor:
    """Element-wise Huber loss between x and y."""
    diff = x - y
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).to(diff.dtype)
    return flag * 0.5 * diff**2 + (1 - flag) * delta * (abs_diff - 0.5 * delta)


@dataclass
class LossDistillCfg:
    delta: float = 1.0
    weight_pose: float = 1.0
    weight_depth: float = 1.0
    weight_normal: float = 1.0


@dataclass
class LossDistillCfgWrapper:
    distill: LossDistillCfg


class DistillLoss(nn.Module):
    def __init__(self, cfg: LossDistillCfgWrapper):
        super().__init__()
        self.cfg = cfg.distill
        self.gamma = _POSE_LOSS_DECAY_GAMMA

    def camera_loss_single(
        self,
        cur_pred_pose_enc: Tensor,
        gt_pose_encoding: Tensor,
        loss_type: str = "l1",
    ) -> tuple[Tensor, Tensor, Tensor]:
        if loss_type == "l1":
            loss_T  = (cur_pred_pose_enc[..., :3]  - gt_pose_encoding[..., :3]).abs()
            loss_R  = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).abs()
            loss_fl = (cur_pred_pose_enc[..., 7:]  - gt_pose_encoding[..., 7:]).abs()
        elif loss_type == "l2":
            loss_T  = (cur_pred_pose_enc[..., :3]  - gt_pose_encoding[..., :3]).norm(dim=-1, keepdim=True)
            loss_R  = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).norm(dim=-1)
            loss_fl = (cur_pred_pose_enc[..., 7:]  - gt_pose_encoding[..., 7:]).norm(dim=-1)
        elif loss_type == "huber":
            loss_T  = huber_loss(cur_pred_pose_enc[..., :3],  gt_pose_encoding[..., :3])
            loss_R  = huber_loss(cur_pred_pose_enc[..., 3:7], gt_pose_encoding[..., 3:7])
            loss_fl = huber_loss(cur_pred_pose_enc[..., 7:],  gt_pose_encoding[..., 7:])
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Clamp to finite range and reduce.
        loss_T  = loss_T.nan_to_num(0.0).clamp(-100, 100).mean()
        loss_R  = loss_R.nan_to_num(0.0).clamp(-100, 100).mean()
        loss_fl = loss_fl.nan_to_num(0.0).clamp(-100, 100).mean()
        return loss_T, loss_R, loss_fl

    def forward(
        self,
        distill_infos: DistillationOutput,
        encoder_output_context: EncoderOutput,
        pred_depth: Float[Tensor, "b v h w"],
        batch: AnyExample,
    ) -> dict[str, Tensor]:

        device = pred_depth.device
        if self.cfg.weight_pose == 0.0 and self.cfg.weight_depth == 0.0 and self.cfg.weight_normal == 0.0:
            zero = torch.tensor(0.0, device=device)
            return {"loss_distill": zero, "loss_pose": zero, "loss_depth": zero, "loss_normal": zero}

        loss_pose = pred_depth.new_tensor(0.0)
        loss_depth = pred_depth.new_tensor(0.0)
        loss_normal = pred_depth.new_tensor(0.0)

        pred_pose_enc_list = (
            encoder_output_context.camera_pose["encodings"]
            if encoder_output_context.camera_pose is not None else None
        )
        if self.cfg.weight_pose > 0.0 and pred_pose_enc_list is not None:
            pesudo_gt_pose_enc = distill_infos["pred_pose_enc_list"]
            num_predictions = len(pred_pose_enc_list)
            for i, (cur_pred, cur_gt) in enumerate(zip(pred_pose_enc_list, pesudo_gt_pose_enc)):
                i_weight = self.gamma ** (num_predictions - i - 1)
                loss_pose = loss_pose + i_weight * huber_loss(cur_pred, cur_gt).mean()
            loss_pose = torch.nan_to_num(loss_pose / num_predictions, nan=0.0, posinf=0.0, neginf=0.0)

        pred_depth      = encoder_output_context.depth["depth"].flatten(0, 1).squeeze(-1)
        pesudo_gt_depth = distill_infos["depth_map"].flatten(0, 1).squeeze(-1)

        if self.cfg.weight_depth > 0.0:
            loss_depth = F.mse_loss(pred_depth, pesudo_gt_depth)

        if self.cfg.weight_normal > 0.0:
            loss_normal = _normal_map_loss(
                pred_depth, pesudo_gt_depth,
                batch["context"]["intrinsics"].flatten(0, 1),
            )

        loss_distill = torch.nan_to_num(
            loss_pose * self.cfg.weight_pose
            + loss_depth * self.cfg.weight_depth
            + loss_normal * self.cfg.weight_normal,
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        return {
            "loss_distill": loss_distill,
            "loss_pose":    loss_pose   * self.cfg.weight_pose,
            "loss_depth":   loss_depth  * self.cfg.weight_depth,
            "loss_normal":  loss_normal * self.cfg.weight_normal,
        }
