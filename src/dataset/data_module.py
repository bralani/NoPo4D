import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import default_collate
from lightning.pytorch import LightningDataModule
from torch import Generator, nn
from torch.utils.data import DataLoader
from typing import cast
from src.cfg import get_cfg


from ..utils.step_tracker import StepTracker
from ..utils.distributed import get_world_size, get_rank
from . import DatasetCfgWrapper, get_dataset
from .types import DataShim, BatchedExample
from .dataset import DatasetShim
from .data_sampler import HomogeneousBatchSampler

def custom_collate_fn(batch):
    if len(batch) == 1:
        return batch[0]
    return default_collate(batch)


def get_data_shim(encoder: nn.Module) -> DataShim:
    """Get functions that modify the batch. It's sometimes necessary to modify batches
    outside the data loader because GPU computations are required to modify the batch or
    because the modification depends on something outside the data loader.
    """

    shims: list[DataShim] = []
    if hasattr(encoder, "get_data_shim"):
        shims.append(encoder.get_data_shim())

    def combined_shim(batch):
        for shim in shims:
            batch = shim(batch)
        return batch

    return combined_shim


def combine_batches(batch_list: list[dict]) -> BatchedExample:
    """Combine a list of per-dataloader batches into a single batch.

    Args:
        batch_list: list of batch dicts from different dataloaders.

    Returns:
        A single combined batch conforming to BatchedExample.
    """
    batch_combined = None
    for batch_per_dl in batch_list:
        if batch_combined is None:
            # start from a shallow copy to avoid aliasing
            batch_combined = {k: v for k, v in batch_per_dl.items()}
        else:
            for k in batch_combined.keys():
                if isinstance(batch_combined[k], list):
                    batch_combined[k] += batch_per_dl[k]
                elif isinstance(batch_combined[k], dict):
                    for kk in batch_combined[k].keys():
                        batch_combined[k][kk] = torch.cat([batch_combined[k][kk], batch_per_dl[k][kk]], dim=0)
                else:
                    raise NotImplementedError(f"combine_batches does not support key='{k}' with type {type(batch_combined[k])}")

    assert batch_combined is not None, "Batch must include at least one sample"
    if batch_combined["context"]["image"].ndim == 4:
        for k in batch_combined.keys():
            if isinstance(batch_combined[k], dict):
                for kk in batch_combined[k].keys():
                    batch_combined[k][kk] = batch_combined[k][kk].unsqueeze(0)
            elif not isinstance(batch_combined[k], list):
                batch_combined[k] = [batch_combined[k]]

        
    return cast(BatchedExample, batch_combined)


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    seed: int | None


@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg
    test: DataLoaderStageCfg
    val: DataLoaderStageCfg

def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class DataModule(LightningDataModule):
    dataset_cfgs: list[DatasetCfgWrapper]
    data_loader_cfg: DataLoaderCfg
    step_tracker: StepTracker | None
    dataset_shim: DatasetShim
    global_rank: int
    
    def __init__(
        self,
        dataset_cfgs: list[DatasetCfgWrapper],
        data_loader_cfg: DataLoaderCfg,
        step_tracker: StepTracker | None = None,
        dataset_shim: DatasetShim = lambda dataset, _: dataset,
        global_rank: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_cfgs = dataset_cfgs
        self.data_loader_cfg = data_loader_cfg
        self.step_tracker = step_tracker
        self.dataset_shim = dataset_shim
        self.global_rank = global_rank
        self.train_generator = self.init_generator(self.data_loader_cfg.train)
        self.val_generator = self.init_generator(self.data_loader_cfg.val)
        self.test_generator = self.init_generator(self.data_loader_cfg.test)
        
    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def init_generator(self, loader_cfg: DataLoaderStageCfg) -> Generator | None:
        if loader_cfg.seed is None:
            return None
        
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator
        
    def train_dataloader(self):
        dataset, datasets_ls = get_dataset(self.dataset_cfgs, "train", self.step_tracker, self.dataset_shim,
                                           generator=self.train_generator,
                                           batch_size=self.data_loader_cfg.train.batch_size)
        world_size = get_world_size()
        rank = get_rank()
        prob_ls = [dataset.cfg.sampling_weight for dataset in datasets_ls]
        # we assume all the dataset share the same num_context_views
        
        if len(datasets_ls) > 1:
            prob = prob_ls
            context_num_views = [dataset.cfg.view_sampler.num_context_views for dataset in datasets_ls]
        else:
            prob = None
            dataset_key = next(iter(get_cfg()["dataset"]))
            dataset_cfg = get_cfg()["dataset"][dataset_key]
            vs_cfg = dataset_cfg['view_sampler']
            context_num_views = vs_cfg.get('max_context_views', vs_cfg.get('num_context_views'))
            
        sampler = HomogeneousBatchSampler(datasets_ls,
                                    batch_size=self.data_loader_cfg.train.batch_size,
                                    num_context_views=context_num_views, 
                                    world_size=world_size, 
                                    rank=rank,
                                    prob=prob,
                                    generator=self.train_generator)
        sampler.set_epoch(0)
        self.train_loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self.data_loader_cfg.train.num_workers,
            generator=self.train_generator,
            worker_init_fn=worker_init_fn,
            collate_fn=custom_collate_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.train),
        )
        # Set epoch for train and validation loaders (if applicable)
        if hasattr(self.train_loader, "dataset") and hasattr(self.train_loader.dataset, "set_epoch"):
            print("Training: Set Epoch in DataModule")
            self.train_loader.dataset.set_epoch(0)
        if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
            print("Training: Set Epoch in DataModule")
            self.train_loader.sampler.set_epoch(0)
        
        return self.train_loader

    def val_dataloader(self):
        dataset, datasets_ls = get_dataset(self.dataset_cfgs, "val", self.step_tracker,
                                           self.dataset_shim, generator=self.val_generator,
                                           batch_size=self.data_loader_cfg.val.batch_size)
        world_size = get_world_size()
        rank = get_rank()
        # here, we random select one dataset for val
        dataset_key = next(iter(get_cfg()["dataset"]))
        dataset_cfg = get_cfg()["dataset"][dataset_key]
        if len(datasets_ls) > 1:
             prob = [0.5] * len(datasets_ls)
        else:
            prob = None
        val_vs_cfg = dataset_cfg['view_sampler']
        val_context_num_views = val_vs_cfg.get('max_context_views', val_vs_cfg.get('num_context_views'))
        sampler = HomogeneousBatchSampler(datasets_ls,
                                    batch_size=self.data_loader_cfg.val.batch_size,
                                    num_context_views=val_context_num_views,
                                    world_size=world_size,
                                    rank=rank,
                                    prob=prob,
                                    generator=self.val_generator)
        sampler.set_epoch(0)
        self.val_loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self.data_loader_cfg.val.num_workers,
            generator=self.val_generator,
            worker_init_fn=worker_init_fn,
            collate_fn=custom_collate_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.val),
        )
        if hasattr(self.val_loader, "dataset") and hasattr(self.val_loader.dataset, "set_epoch"):
            print("Validation: Set Epoch in DataModule")
            self.val_loader.dataset.set_epoch(0)
        if hasattr(self.val_loader, "sampler") and hasattr(self.val_loader.sampler, "set_epoch"):
            print("Validation: Set Epoch in DataModule")
            self.val_loader.sampler.set_epoch(0)
        return self.val_loader

    def test_dataloader(self):
        dataset = get_dataset(self.dataset_cfgs, "test", self.step_tracker, self.dataset_shim, generator=self.test_generator, batch_size=self.data_loader_cfg.test.batch_size)
        data_loader = DataLoader(
            dataset,
            self.data_loader_cfg.test.batch_size,
            num_workers=self.data_loader_cfg.test.num_workers,
            generator=self.test_generator,
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.test),
        )
            
        return data_loader
