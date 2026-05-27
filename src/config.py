from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .dataset import DatasetCfgWrapper
from .dataset.data_module import DataLoaderCfg
from .model.decoder import DecoderCfg
from .model.encoder import EncoderCfg
from .training.loss import LossCfgWrapper
from .training.model_wrapper import OptimizerCfg, TrainCfg


@dataclass
class CheckpointingCfg:
    load: Optional[str]           # checkpoint URI to resume from; str not Path because it may be "wandb://..."
    load_only_weights: bool       # if True, only model weights are restored (optimizer/scheduler state is dropped)
    every_n_train_steps: int      # save a checkpoint every N optimizer steps
    save_top_k: int               # keep only the top-K checkpoints by validation metric (-1 = keep all)
    save_weights_only: bool       # if True, omit optimizer state from saved checkpoints (smaller files)


@dataclass
class ModelCfg:
    encoder: EncoderCfg           # backbone + feature extraction config
    decoder: DecoderCfg           # head / output projection config


@dataclass
class TrainerCfg:
    max_steps: int                                              # total number of optimizer steps before training stops
    val_check_interval: int | float | None                      # run validation every N steps (int) or every fraction of an epoch (float)
    gradient_clip_val: int | float | None                       # max gradient norm for clipping; None disables clipping
    num_nodes: int = 1                                          # number of machines for distributed training
    accumulate_grad_batches: int = 1                            # gradient accumulation steps before an optimizer update
    precision: Literal["32", "16-mixed", "bf16-mixed"] = "32"   # floating-point precision used during training


@dataclass
class RootCfg:
    seed: int                               # global random seed for reproducibility
    base_path: str                          # root directory for outputs (checkpoints, logs, etc.)
    wandb: dict                             # wandb.init() kwargs (project, entity, name, tags, …)
    dataset: list[DatasetCfgWrapper]        # one or more dataset configs (train/val splits)
    data_loader: DataLoaderCfg              # batch size, num_workers, and related DataLoader settings
    model: ModelCfg                         # encoder + decoder architecture config
    optimizer: OptimizerCfg                 # optimizer type and hyperparameters (lr, weight_decay, …)
    loss: list[LossCfgWrapper]              # one or more loss terms with their weights
    train: TrainCfg                         # training-loop settings
    trainer: TrainerCfg                     # PyTorch Lightning Trainer settings
    checkpointing: CheckpointingCfg         # checkpoint save/load behaviour


# dacite type hook: keeps Path fields as Path objects after OmegaConf resolution
TYPE_HOOKS = {Path: Path}

T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg, resolve=True),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )


def _separate_wrappers(joined: dict, wrapper_type) -> list:
    """Convert a flat {name: cfg} dict into a list of typed wrapper objects.

    dacite cannot directly resolve union types inside a plain list, so each
    entry is wrapped in a temporary Dummy dataclass first.
    """
    @dataclass
    class Dummy:
        dummy: wrapper_type  # type: ignore[valid-type]

    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]


def separate_loss_cfg_wrappers(joined: dict) -> list[LossCfgWrapper]:
    return _separate_wrappers(joined, LossCfgWrapper)


def separate_dataset_cfg_wrappers(joined: dict) -> list[DatasetCfgWrapper]:
    return _separate_wrappers(joined, DatasetCfgWrapper)


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
        {
            list[LossCfgWrapper]: separate_loss_cfg_wrappers,
            list[DatasetCfgWrapper]: separate_dataset_cfg_wrappers,
        },
    )
