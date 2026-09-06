from dataclasses import dataclass
from typing import Callable, Literal

from jaxtyping import Float
from torch import Tensor
import torch
from src.dataset.types import BatchedExample
from src.model.decoder.types import DecoderOutput
from src.model.encoder.types import EncoderOutput
from src.model.types import Gaussians
from .loss import Loss

_OPACITY_FNS: dict[str, Callable[[Tensor], Tensor]] = {
    "exp":      lambda o: torch.exp(-(o - 0.5) ** 2 / 0.05).mean(),
    "mean":     lambda o: o.mean(),
    "exp+mean": lambda o: 0.5 * torch.exp(-(o - 0.5) ** 2 / 0.05).mean() + o.mean()
}


@dataclass
class LossOpacityCfg:
    weight: float
    type: Literal["exp", "mean", "exp+mean"] = "exp+mean"
    sh_weight: float = 0.0  # regularizer weight to shrink opacity SH coefficients toward 0


@dataclass
class LossOpacityCfgWrapper:
    opacity: LossOpacityCfg


class LossOpacity(Loss[LossOpacityCfg, LossOpacityCfgWrapper]):
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

        warmup_ratio = min(1.0, global_step / max(1, warmup_steps))
        opacities = gaussians.opacities

        opacity_loss = _OPACITY_FNS[self.cfg.type](opacities)

        total_loss = self.cfg.weight * opacity_loss
        if self.cfg.sh_weight > 0.0 and gaussians.opacity_sh is not None:
            sh_loss = gaussians.opacity_sh.abs().mean()
            total_loss = total_loss + self.cfg.sh_weight * sh_loss

        return warmup_ratio * torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
