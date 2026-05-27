"""Training entry point for NoPo4D. Configures logging, checkpointing, and the Lightning Trainer, then launches training."""

import os
import sys
import warnings

# Make project root importable when running this script directly.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from pathlib import Path

import hydra
import torch
import wandb
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

from src.model import get_model
from src.utils.nn import recursive_load_weights

# Enables beartype runtime type-checking for all src.* imports inside this block.
with install_import_hook(("src",), ("beartype", "beartype")):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.cfg import set_cfg
    from src.training.loss import get_losses
    from src.training.logger.LocalLogger import LocalLogger
    from src.utils.step_tracker import StepTracker
    from src.training.logger.wandb_tools import update_checkpoint_path
    from src.training.model_wrapper import ModelWrapper


def _setup_logger(cfg_dict: DictConfig, output_dir: Path):
    """Returns (logger, callbacks). Callbacks includes LRMonitor when wandb is active."""
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))
        if wandb.run is not None:  # wandb.run is None on ranks != 0
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()
    return logger, callbacks


def _setup_checkpointing(cfg, output_dir: Path) -> ModelCheckpoint:
    ckpt_callback = ModelCheckpoint(
        output_dir / "checkpoints",
        every_n_train_steps=cfg.checkpointing.every_n_train_steps,
        save_top_k=cfg.checkpointing.save_top_k,
        save_weights_only=cfg.checkpointing.save_weights_only,
        monitor="info/global_step",
        mode="max",  # keep the most recent checkpoints
    )
    ckpt_callback.CHECKPOINT_EQUALS_CHAR = '_'  # avoid '=' in filenames
    return ckpt_callback


def _load_model(cfg, checkpoint_path):
    model = get_model(cfg.model.encoder, cfg.model.decoder)
    if cfg.checkpointing.load_only_weights:
        print(f"Loading weights only from checkpoint: {checkpoint_path}")
        ckpt = torch.load(str(checkpoint_path), map_location='cpu')
        recursive_load_weights(model, ckpt.get("state_dict", ckpt))
        del ckpt
        checkpoint_path = None  # prevent trainer.fit from also restoring optimizer/scheduler state
    return model, checkpoint_path


@hydra.main(version_base=None, config_path="../config", config_name="main")
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to {output_dir}.")
    cfg.train.output_path = output_dir

    logger, callbacks = _setup_logger(cfg_dict, output_dir)
    callbacks.append(_setup_checkpointing(cfg, output_dir))
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        devices="auto",
        num_nodes=cfg.trainer.num_nodes,
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        logger=logger,
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,  # required to use val_check_interval in steps
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        inference_mode=True,
    )
    # Offset seed per rank so each GPU samples different data.
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    model, checkpoint_path = _load_model(cfg, checkpoint_path)
    model_wrapper = ModelWrapper(
        cfg.optimizer, cfg.train, model, get_losses(cfg.loss), step_tracker
    )
    data_module = DataModule(
        cfg.dataset, cfg.data_loader, step_tracker, global_rank=trainer.global_rank
    )

    trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    train()
