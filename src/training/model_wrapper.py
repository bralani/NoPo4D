"""Lightning wrapper that wires together the model, losses, optimizer, and metrics."""

import re
from typing import cast

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn

from src.dataset.data_module import get_data_shim
from src.utils.step_tracker import StepTracker
from src.model.nopo4d import NoPo4D
from src.training.config import TrainCfg
from src.training.distillation.distillation import DistillationManager
from src.training.loss import Loss
from src.training.loss.loss_computer import LossComputer
from src.training.loss.loss_distill import DistillLoss
from src.training.logger.metrics_logger import MetricsLogger
from src.training.optimizer import build_optimizer_and_scheduler, OptimizerCfg
from src.training.training_wrapper import TrainingMixin
from src.training.validation_wrapper import ValidationMixin


def _apply_selective_freezing(module: nn.Module, freeze_module_regex: list[str] | None) -> None:
    """Freeze encoder params matching any regex, but always keep distillation params trainable."""
    if not freeze_module_regex:
        return
    patterns = [re.compile(r) for r in freeze_module_regex]
    for name, param in module.named_parameters():
        if "distill" not in name and any(p.search(name) for p in patterns):
            param.requires_grad = False


class ModelWrapper(TrainingMixin, ValidationMixin, LightningModule):
    logger: WandbLogger | None
    trainer: Trainer
    model: NoPo4D
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        train_cfg: TrainCfg,
        model: NoPo4D,
        losses: list[nn.Module],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.model = model
        self.data_shim = get_data_shim(self.model.encoder)

        # Freeze encoder params as configured
        _apply_selective_freezing(self.model.encoder, self.train_cfg.freeze_module_regex)

        # Build losses and loss computer
        self.losses = nn.ModuleList(losses)
        self.loss_computer = LossComputer(self.losses, self.train_cfg)

        # Build distillation manager
        has_distill = any(isinstance(l, DistillLoss) for l in self.losses)
        self.distill_manager = DistillationManager(self.model.encoder.cfg) if has_distill else None

        # Build metrics logger for logging validation metrics and visualizations
        self.metrics_logger = MetricsLogger()

        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
        print("Trainable parameters:\n" + "\n".join(f"  {n}" for n in trainable))

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return cast(OptimizerLRScheduler, build_optimizer_and_scheduler(self, self.optimizer_cfg))
