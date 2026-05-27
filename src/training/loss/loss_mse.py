from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from src.dataset.types import BatchedExample
from src.model.decoder.types import DecoderOutput
from src.model.encoder.types import EncoderOutput
from src.model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
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
        loss = torch.nn.functional.mse_loss(prediction_target.color, batch["target"]["image"])
        return self.cfg.weight * torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
