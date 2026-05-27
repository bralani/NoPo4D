from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
import torch
from src.dataset.types import BatchedExample
from src.model.decoder.types import DecoderOutput
from src.model.encoder.types import EncoderOutput
from src.model.types import Gaussians
from .loss import Loss


@dataclass
class LossLifeSpanCfg:
    weight: float


@dataclass
class LossLifeSpanCfgWrapper:
    life_span: LossLifeSpanCfg


class LossLifeSpan(Loss[LossLifeSpanCfg, LossLifeSpanCfgWrapper]):
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

        if gaussians.cov_t is None:
            return torch.tensor(0.0, device=prediction_target.color.device)

        loss = 1.0 / (gaussians.cov_t.mean() + 1e-8)

        return warmup_ratio * self.cfg.weight * loss
    
