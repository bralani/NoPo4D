from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from src.dataset.types import BatchedExample
from src.model.decoder.types import DecoderOutput
from src.model.encoder.types import EncoderOutput
from src.model.types import Gaussians
from src.training.distillation.types import DistillationOutput
from .loss import Loss
from .loss_ssim import ssim
from .utils import _normal_map_loss


@dataclass
class LossDepthConsisCfg:
    weight: float
    loss_type: Literal["MSE", "SSIM"] = "MSE"
    detach: bool = False
    apply_after_step: int = 0
    init_weight: float | None = None
    decay_steps: int | None = None
    weight_normal: float = 0.0
    alpha_penalty_weight: float = 0.0


@dataclass
class LossDepthConsisCfgWrapper:
    depth_consis: LossDepthConsisCfg


def _loss_mse(r: Tensor, p: Tensor, batch: BatchedExample) -> Tensor:
    return F.mse_loss(r, p)


def _loss_ssim_depth(r: Tensor, p: Tensor, batch: BatchedExample) -> Tensor:
    ssim_val, _, _, _ = ssim(r.unsqueeze(1), p.unsqueeze(1), data_range=1.0, size_average=True)
    return 1.0 - ssim_val


_DEPTH_LOSS_FNS = {
    "MSE":  _loss_mse,
    "SSIM": _loss_ssim_depth,
}


class LossDepthConsis(Loss[LossDepthConsisCfg, LossDepthConsisCfgWrapper]):

    def _compute_decayed_weight(self, global_step: int) -> float:
        """Linearly interpolate weight from init_weight to weight over decay_steps."""
        if self.cfg.init_weight is None or self.cfg.decay_steps is None:
            return self.cfg.weight
        effective_step = max(0, global_step - self.cfg.apply_after_step)
        if effective_step >= self.cfg.decay_steps:
            return self.cfg.weight
        t = effective_step / self.cfg.decay_steps
        return self.cfg.init_weight + (self.cfg.weight - self.cfg.init_weight) * t

    def forward(
        self,
        prediction_context: DecoderOutput,
        prediction_target: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict,
        global_step: int,
        warmup_steps: int = 0,
        encoder_output_context: EncoderOutput | None = None,
        distill_infos: DistillationOutput | None = None,
        visualization_cache: dict | None = None,
    ) -> Float[Tensor, ""]:

        device = prediction_context.depth.device
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0.0, device=device)

        current_weight = self._compute_decayed_weight(global_step)
        rendered_depth = prediction_context.depth.flatten(0, 1).squeeze(-1)
        pred_depth = depth_dict["depth"].flatten(0, 1).squeeze(-1)
        if self.cfg.detach:
            pred_depth = pred_depth.detach()

        depth_loss = _DEPTH_LOSS_FNS[self.cfg.loss_type](rendered_depth, pred_depth, batch)

        alpha_penalty = rendered_depth.new_tensor(0.0)
        if self.cfg.alpha_penalty_weight > 0.0 and prediction_context.alpha is not None:
            alpha_flat = prediction_context.alpha.flatten(0, 1)
            alpha_penalty = torch.nan_to_num((1.0 - alpha_flat.clamp(0.0, 1.0)).mean(), nan=0.0)

        loss_normal = rendered_depth.new_tensor(0.0)
        if self.cfg.weight_normal > 0.0:
            loss_normal = _normal_map_loss(
                rendered_depth, pred_depth,
                batch["context"]["intrinsics"].flatten(0, 1),
            )

        return (
            current_weight * torch.nan_to_num(depth_loss, nan=0.0)
            + self.cfg.alpha_penalty_weight * alpha_penalty
            + self.cfg.weight_normal * loss_normal
        )

