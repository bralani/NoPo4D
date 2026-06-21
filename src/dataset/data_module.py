"""LightningDataModule wiring together datasets, samplers, and dataloaders."""

import itertools
import random
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch import Generator
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler
from lightning.pytorch import LightningDataModule

from ..utils.step_tracker import StepTracker
from ..utils.distributed import get_world_size, get_rank
from . import DatasetCfgWrapper, get_dataset
from .types import BatchedExample, UnbatchedExample


def worker_init_fn(_: int) -> None:
    # Each worker gets a unique seed to avoid identical augmentations across workers.
    seed = int(torch.utils.data.get_worker_info().seed) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)


def collate_fn(batch: list[UnbatchedExample]) -> BatchedExample:
    result = {}
    for k in batch[0]:
        if isinstance(batch[0][k], dict):
            result[k] = {kk: torch.stack([b[k][kk] for b in batch]) for kk in batch[0][k]}
        else:
            result[k] = [b[k] for b in batch]
    return cast(BatchedExample, result)


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    seed: int | None  # None: non-deterministic


@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg
    val: DataLoaderStageCfg


class DataSampler(BatchSampler):
    """
    Batch sampler that synchronizes sampling across distributed workers.

    Randomly draws the number of cameras, context views, and target views using
    a shared seeded RNG. Yields batches of index tuples:
    (dataset_index, num_context, num_target, num_cameras).
    """

    SEED_OFFSET_MULTIPLIER = 100

    def __init__(self, dataset, batch_size: int, world_size: int = 1, rank: int = 0, seed: int = 42, sampler=None):
        self.batch_size = batch_size
        self.drop_last = True  # read by DataLoader when batch_sampler is provided
        self.rng = random.Random()
        self.view_sampler = dataset.view_sampler

        vs_cfg = dataset.cfg.view_sampler
        self.min_cam = vs_cfg.min_cameras
        self.max_cam = vs_cfg.max_cameras
        self.min_context = vs_cfg.min_context_views
        self.max_context = vs_cfg.max_context_views
        self.min_target = vs_cfg.min_target_views
        self.max_target = vs_cfg.max_target_views

        self.excluded_cameras = set(getattr(vs_cfg, "exclude_cameras", []) or [])

        self._dist_sampler = sampler if sampler is not None else DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=False
        )
        self.set_epoch(seed)

    def set_epoch(self, epoch: int) -> None:
        """Tie the RNG seed to the epoch so all distributed workers agree."""
        if hasattr(self._dist_sampler, "set_epoch"):
            self._dist_sampler.set_epoch(epoch)
        self.epoch = epoch
        self.rng.seed(epoch * self.SEED_OFFSET_MULTIPLIER)

    @property
    def current_max_cameras(self) -> int:
        if hasattr(self.view_sampler, "get_current_max_cameras"):
            return self.view_sampler.get_current_max_cameras()
        return self.max_cam

    @property
    def current_max_context_views(self) -> int:
        if hasattr(self.view_sampler, "get_current_max_context_views"):
            return self.view_sampler.get_current_max_context_views()
        return self.max_context

    @property
    def current_max_target_views(self) -> int:
        if hasattr(self.view_sampler, "get_current_max_target_views"):
            return self.view_sampler.get_current_max_target_views()
        return self.max_target

    def _sample_num_cameras(self) -> int:
        if self.max_cam <= 1:
            return 1
        eligible_cams = [
            cam for cam in range(self.min_cam, self.current_max_cameras + 1)
            if cam not in self.excluded_cameras
        ]
        return self.rng.choice(eligible_cams) if eligible_cams else self.min_cam

    def _sample_view_count(self, min_views: int, max_views: int, scale: float) -> int:
        """Scales the max views and samples a random view count."""
        scaled_max = max(min_views, round(max_views * scale))
        return self.rng.randint(min_views, scaled_max)

    def __iter__(self):
        """
        Yield batches of index tuples. Total frames per batch are kept
        roughly constant by scaling view counts inversely to camera counts.
        """
        it = iter(self._dist_sampler)
        for batch_indices in iter(lambda: list(itertools.islice(it, self.batch_size)), []):
            num_cameras = self._sample_num_cameras()

            # Scale view counts inversely so total frames stay relatively stable.
            view_scale = max(1, self.max_cam) / num_cameras

            num_context = self._sample_view_count(self.min_context, self.current_max_context_views, view_scale)
            num_target = self._sample_view_count(self.min_target, self.current_max_target_views, view_scale)

            yield [(idx, num_context, num_target, num_cameras) for idx in batch_indices]

    def __len__(self) -> int:
        return len(self._dist_sampler)


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset_cfgs: list[DatasetCfgWrapper],
        data_loader_cfg: DataLoaderCfg,
        step_tracker: StepTracker | None = None,
        global_rank: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_cfgs = dataset_cfgs
        self.data_loader_cfg = data_loader_cfg
        self.step_tracker = step_tracker
        self.global_rank = global_rank
        self.train_generator = self._init_generator(data_loader_cfg.train)
        self.val_generator = self._init_generator(data_loader_cfg.val)

    def _init_generator(self, cfg: DataLoaderStageCfg) -> Generator | None:
        if cfg.seed is None:
            return None
        g = Generator()
        g.manual_seed(cfg.seed + self.global_rank) # each gpu gets a different random augmentation
        return g

    def _make_loader(self, stage: str, cfg: DataLoaderStageCfg, generator: Generator | None) -> DataLoader:
        dataset = get_dataset(
            self.dataset_cfgs, stage, self.step_tracker,
            generator=generator,
        )
        sampler = DataSampler(dataset, batch_size=cfg.batch_size, world_size=get_world_size(), rank=get_rank())
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=cfg.num_workers,
            generator=generator,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
            persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        )
        sampler.set_epoch(0)
        return loader

    def train_dataloader(self) -> DataLoader:
        self.train_loader = self._make_loader("train", self.data_loader_cfg.train, self.train_generator)
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        self.val_loader = self._make_loader("val", self.data_loader_cfg.val, self.val_generator)
        return self.val_loader
