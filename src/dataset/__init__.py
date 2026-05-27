from dataclasses import fields
from typing import Type
from torch.utils.data import ConcatDataset
from torch import Generator
import bisect

from ..utils.step_tracker import StepTracker
from .types import Stage
from .view_sampler import get_view_sampler
from torch.utils.data import Dataset
from .dataset import BaseDataset, DatasetShim
from .dataset_exo4d import DatasetExo4D, DatasetExo4DCfgWrapper

DATASETS: dict[str, Type[BaseDataset]] = {
    "exo4d": DatasetExo4D,
}

DatasetCfgWrapper = DatasetExo4DCfgWrapper

class TestDatasetWrapper(Dataset):
    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset

    def __getitem__(self, idx):

        return self.dataset[(idx, self.dataset.view_sampler.num_context_views, self.dataset.cfg.input_image_shape[1] // 14)] # fake parameters here, to fit the input of dataset
    
    def __len__(self):
        return len(self.dataset)

        
    
class CustomConcatDataset(ConcatDataset):

    def __getitem__(self, idx_tuple):

        batch_size = 1
        while isinstance(idx_tuple, list):
            if len(idx_tuple) == 0:
                raise ValueError("Received empty index list in CustomConcatDataset")
            batch_size *= len(idx_tuple)
            idx_tuple = idx_tuple[0]

        if not isinstance(idx_tuple, tuple):
            raise TypeError(f"Expected tuple index, got {type(idx_tuple)}")

        idx = idx_tuple[0]
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        extra = idx_tuple[1:]
        if len(extra) >= 1:
            request = (sample_idx,) + tuple(extra)
        else:
            request = sample_idx

        return [self.datasets[dataset_idx][request] for _ in range(batch_size)]


def get_dataset(
    cfgs: list[DatasetCfgWrapper],
    stage: Stage,
    step_tracker: StepTracker | None,
    dataset_shim: DatasetShim,
    generator: Generator | None = None,
    batch_size: int = 1,
) -> tuple[CustomConcatDataset, list[BaseDataset]] | TestDatasetWrapper:
    datasets = []
    if stage != "test":
        if stage == "val":
            cfgs = [cfgs[0]]
        for cfg in cfgs:
            (field,) = fields(type(cfg))
            cfg = getattr(cfg, field.name)
            
            view_sampler = get_view_sampler(
                cfg.view_sampler,
                stage,
                cfg.overfit_to_scene is not None,
                cfg.cameras_are_circular,
                step_tracker,
                generator=generator,
                batch_size=batch_size,
            )
            dataset = DATASETS[cfg.name](cfg, stage, view_sampler)
            dataset = dataset_shim(dataset, stage)
            datasets.append(dataset)

        return CustomConcatDataset(datasets), datasets
    elif stage == "test":
        assert len(cfgs) == 1
        cfg = cfgs[0]
        (field,) = fields(type(cfg))
        cfg = getattr(cfg, field.name)
        
        view_sampler = get_view_sampler(
            cfg.view_sampler,
            stage,
            cfg.overfit_to_scene is not None,
            cfg.cameras_are_circular,
            step_tracker,
        )
        dataset = DATASETS[cfg.name](cfg, stage, view_sampler)
        dataset = dataset_shim(dataset, stage)

        return TestDatasetWrapper(dataset)