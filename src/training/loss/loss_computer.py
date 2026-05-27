"""Loss computation utilities for model training."""

import torch
from torch import nn, Tensor

from src.model.decoder.types import DecoderOutput
from src.model.encoder.types import EncoderOutput
from src.training.distillation.types import DistillationOutput
from src.model.types import Gaussians
from src.dataset.types import BatchedExample
from src.training.config import TrainCfg
from src.training.loss.loss_distill import DistillLoss


class LossComputer:
    """Handles computation of all training losses."""

    def __init__(
        self,
        losses: nn.ModuleList,
        train_cfg: TrainCfg,
    ) -> None:
        self.losses = losses
        self.train_cfg = train_cfg

    def compute_total_loss(
        self,
        output_context: DecoderOutput,
        output_target: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict,
        distill_infos: DistillationOutput | None,
        encoder_output_context: EncoderOutput | None,
        logger_fn,
        global_step: int,
    ) -> tuple[Tensor, dict]:
        """Compute and log all training losses.

        Returns:
            Tuple of (total loss tensor, visualization cache dict).
        """
        device = output_target.color.device
        total_loss = torch.tensor(0.0, device=device)
        visualization_cache: dict = {}

        with torch.autocast("cuda", enabled=False):
            for loss_fn in self.losses:
                if isinstance(loss_fn, DistillLoss):
                    if distill_infos is not None and output_context.depth is not None and encoder_output_context is not None:
                        loss_dict = loss_fn(distill_infos, encoder_output_context, output_context.depth, batch)
                        # DistillLoss returns a dict of sub-losses; log each and accumulate the combined one.
                        for suffix, key in (("distill", "loss_distill"), ("pose", "loss_pose"),
                                            ("depth", "loss_depth"), ("normal", "loss_normal")):
                            logger_fn(f"loss/distill_{suffix}" if suffix != "distill" else "loss/distill", loss_dict[key])
                        total_loss = total_loss + loss_dict["loss_distill"]
                else:
                    loss = loss_fn(
                        output_context, output_target, batch, gaussians, depth_dict,
                        global_step, warmup_steps=self.train_cfg.warmup_steps_regularization,
                        encoder_output_context=encoder_output_context,
                        distill_infos=distill_infos,
                        visualization_cache=visualization_cache,
                    )
                    logger_fn(f"loss/{loss_fn.name}", loss)
                    total_loss = total_loss + loss

        logger_fn("loss/total", total_loss)
        return total_loss, visualization_cache
