"""Dataset registry and factory for constructing dataset instances."""

from dataclasses import fields
from typing import Type
from torch import Generator

from ..utils.step_tracker import StepTracker
from .types import Stage
from .view_sampler import get_view_sampler
from .dataset import BaseDataset
from .dataset_exo4d import DatasetExo4D, DatasetExo4DCfgWrapper

# To add a new dataset:
#   1. Create src/dataset/dataset_<name>.py with a BaseDataset subclass and its Cfg/CfgWrapper dataclasses
#   2. Add an entry to DATASETS: {"<name>": DatasetName}
#   3. Update DatasetCfgWrapper to a Union: DatasetExo4DCfgWrapper | DatasetNameCfgWrapper
DATASETS: dict[str, Type[BaseDataset]] = {
    "exo4d": DatasetExo4D,
}

DatasetCfgWrapper = DatasetExo4DCfgWrapper


def get_dataset(
    cfgs: list[DatasetCfgWrapper],
    stage: Stage,
    step_tracker: StepTracker | None,
    generator: Generator | None = None,
) -> BaseDataset:
    assert len(cfgs) == 1, f"Expected exactly one dataset config, got {len(cfgs)}"
    (cfg_wrapper,) = cfgs
    (field,) = fields(type(cfg_wrapper))
    cfg = getattr(cfg_wrapper, field.name)
    view_sampler = get_view_sampler(cfg.view_sampler, step_tracker, generator=generator)
    return DATASETS[cfg.name](cfg, stage, view_sampler)
