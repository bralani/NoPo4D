from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from src.dataset.types import BatchedExample
from src.utils.nn import convert_to_buffer
from src.model.decoder.types import DecoderOutput
from src.model.encoder.types import EncoderOutput
from src.model.types import Gaussians
from .loss import Loss


@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int
    batch_size: int = 15


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)
        self.lpips = LPIPS(net="alex")
        self.lpips.eval()
        for param in self.lpips.parameters():
            param.requires_grad = False
        convert_to_buffer(self.lpips, persistent=False)

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
        image = batch["target"]["image"]
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=image.device)

        b, v, *_ = prediction_target.color.shape
        pred_flat = rearrange(prediction_target.color, "b v c h w -> (b v) c h w")
        img_flat  = rearrange(image,                  "b v c h w -> (b v) c h w")

        # Compute LPIPS in chunks to avoid OOM.
        chunk_size = self.cfg.batch_size
        losses = [
            self.lpips(pred_flat[i : i + chunk_size], img_flat[i : i + chunk_size], normalize=True)
            for i in range(0, b * v, chunk_size)
        ]
        loss = torch.cat(losses, dim=0)
        return self.cfg.weight * torch.nan_to_num(loss.mean(), nan=0.0, posinf=0.0, neginf=0.0)
