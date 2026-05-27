"""Shared base class for TrainingMixin and ValidationMixin."""

from typing import Callable

from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger

from src.dataset.data_module import combine_batches
from src.dataset.types import BatchedExample
from src.utils.image import prep_image
from src.model.decoder.types import DecoderOutput
from src.model.nopo4d import NoPo4D
from src.training.config import TrainCfg
from src.training.distillation.distillation import DistillationManager
from src.training.distillation.types import DistillationOutput
from src.training.logger.metrics_logger import MetricsLogger
from src.training.logger.visualizer import Visualizer
from src.training.logger.visualization.layout import add_border
from src.training.logger.visualization.video_render import render_video_interpolation


class ModelWrapperBase:
    """Declares the attributes that ModelWrapper provides to both mixins and the shared batch helpers."""

    trainer: Trainer
    current_epoch: int
    global_step: int
    global_rank: int
    data_shim: Callable[[BatchedExample], BatchedExample]
    model: NoPo4D
    train_cfg: TrainCfg
    metrics_logger: MetricsLogger
    logger: WandbLogger | None
    distill_manager: DistillationManager | None

    def _prepare_batch(self, batch) -> BatchedExample:
        if isinstance(batch, list):
            batch = combine_batches(batch)
        return self.data_shim(batch)

    def _log_comparison_and_video(
        self,
        batch: BatchedExample,
        output: DecoderOutput,
        depth_dict: dict,
        distill_infos: DistillationOutput | None,
        log_key: str,
        visualization_cache: dict | None = None,
        key: str = "target",
    ) -> None:
        comparison = Visualizer.create_comparison_image(
            batch=batch,
            output=output,
            depth_dict=depth_dict,
            distill_infos=distill_infos,
            visualization_cache=visualization_cache,
            key=key,
        )
        # Derive the video log prefix from log_key ("comparison_train" -> "train/", "comparison_val" -> "val/")
        stage = log_key.split("_")[-1] + "/"
        render_video_interpolation(wrapper=self, batch=batch, stage=stage, num_frames=120)
        if self.logger is not None:
            self.logger.log_image(
                log_key,
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=batch["scene"],
            )
