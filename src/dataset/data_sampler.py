# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Random sampling under a constraint
# --------------------------------------------------------
import numpy as np
import torch
from typing import Optional
from torch.utils.data import DistributedSampler, Sampler, BatchSampler
import random


class DynamicBatchSampler(Sampler):
    """
    A custom batch sampler that dynamically adjusts batch size, aspect ratio, and image number
    for each sample. Batches within a sample share the same aspect ratio and image number.
    """
    def __init__(self,
                 sampler,
                 image_num_range,
                 h_range,
                 epoch=0,
                 seed=42,
                 max_img_per_gpu=48,
                 target_image_num_range=None,
                 camera_range=None,
                 view_sampler=None):
        """
        Initializes the dynamic batch sampler.

        Args:
            sampler: Instance of DynamicDistributedSampler.
            image_num_range: List containing [min_context_views, max_context_views] per sample.
                These are the base values (valid when camera_range[1] cameras are used).
            h_range: List containing [min_height, max_height].
            epoch: Current epoch number.
            seed: Random seed for reproducibility.
            max_img_per_gpu: Maximum number of images to fit in GPU memory.
            target_image_num_range: List containing [min_target_views, max_target_views]. If None,
                target count is not randomized (sampler yields None for target_num).
            camera_range: List containing [min_cameras, max_cameras]. If None or [1, 1],
                no camera randomization occurs and num_cameras is always 1.
                max_context/target_views are treated as base values for max_cameras cameras;
                when fewer cameras are selected, the max views scale proportionally.
            view_sampler: Optional ViewSampler instance. If it implements
                get_current_max_cameras(), the camera range is warmed up dynamically.
        """
        self.sampler = sampler
        self.image_num_range = image_num_range
        self.h_range = h_range
        self.rng = random.Random()

        # Uniformly sample from the range of possible image numbers.
        self.image_num_weights = {num_images: 1.0 for num_images in range(image_num_range[0], image_num_range[1] + 1)}

        # Possible image numbers, e.g., [2, 3, 4, ..., 24]
        self.possible_nums = list(range(image_num_range[0], image_num_range[1] + 1))

        # Normalize weights for sampling
        weights = [self.image_num_weights[n] for n in self.possible_nums]
        total = sum(weights)
        self.normalized_weights = [w / total for w in weights]

        # Target view count range (optional)
        self.target_image_num_range = target_image_num_range
        if target_image_num_range is not None:
            tgt_weights = {n: 1.0 for n in range(target_image_num_range[0], target_image_num_range[1] + 1)}
            self.possible_target_nums = list(range(target_image_num_range[0], target_image_num_range[1] + 1))
            tgt_w = [tgt_weights[n] for n in self.possible_target_nums]
            tgt_total = sum(tgt_w)
            self.normalized_target_weights = [w / tgt_total for w in tgt_w]
        else:
            self.possible_target_nums = None
            self.normalized_target_weights = None

        # Camera randomization: possible_cameras is None when disabled
        if camera_range is not None and camera_range[1] > 1:
            self.possible_cameras = list(range(camera_range[0], camera_range[1] + 1))
            self.base_cameras = camera_range[1]  # max_cameras = base for scaling
            self.camera_range_min = camera_range[0]
        else:
            self.possible_cameras = None
            self.base_cameras = 1
            self.camera_range_min = 1
        self.view_sampler = view_sampler

        # Maximum image number per GPU
        self.max_img_per_gpu = max_img_per_gpu

        # Set the epoch for the sampler
        self.set_epoch(epoch + seed)

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler, affecting the random sequence.

        Args:
            epoch: The epoch number.
        """
        self.sampler.set_epoch(epoch)
        self.epoch = epoch
        self.rng.seed(epoch * 100)

    def __iter__(self):
        """
        Yields batches of samples with synchronized dynamic parameters.
        All GPUs seeded identically (same epoch → same self.rng state) will pick
        the same random_num_cameras, random_image_num and random_target_num each iteration.

        Camera count is sampled FIRST so all GPUs agree before scaling context/target ranges.
        When fewer than base_cameras are used, max context/target views scale proportionally
        so that total views-per-GPU stays roughly constant.

        Returns:
            Iterator yielding batches of indices with associated parameters.
        """
        sampler_iterator = iter(self.sampler)

        while True:
            try:
                # 1. Sample num_cameras first (all GPUs agree via seeded rng).
                if self.possible_cameras is not None:
                    if self.view_sampler is not None and hasattr(self.view_sampler, 'get_current_max_cameras'):
                        current_max_cam = self.view_sampler.get_current_max_cameras()
                        effective_cameras = list(range(self.camera_range_min, current_max_cam + 1))
                    else:
                        effective_cameras = self.possible_cameras
                    if (
                        self.view_sampler is not None
                        and hasattr(self.view_sampler, 'cfg')
                        and hasattr(self.view_sampler.cfg, 'exclude_cameras')
                        and self.view_sampler.cfg.exclude_cameras
                    ):
                        excluded = set(self.view_sampler.cfg.exclude_cameras)
                        effective_cameras = [c for c in effective_cameras if c not in excluded]
                        if not effective_cameras:
                            effective_cameras = [self.camera_range_min]
                    random_num_cameras = self.rng.choices(effective_cameras, k=1)[0]
                else:
                    random_num_cameras = 1

                # 2. Compute scaled max context/target views for this camera count.
                #    total_budget = base_cameras * max_views; effective_max = total_budget / num_cameras
                scale = self.base_cameras / random_num_cameras
                if self.view_sampler is not None and hasattr(self.view_sampler, 'get_current_max_context_views'):
                    current_max_ctx = self.view_sampler.get_current_max_context_views()
                else:
                    current_max_ctx = self.image_num_range[1]
                scaled_max_ctx = max(self.image_num_range[0], round(current_max_ctx * scale))

                # 3. Use self.rng (seeded per epoch) so all GPUs pick the same counts.
                random_image_num = self.rng.randint(self.image_num_range[0], scaled_max_ctx)
                random_ps_h = self.rng.randint(self.h_range[0] // 14, self.h_range[1] // 14)

                if self.possible_target_nums is not None:
                    if self.view_sampler is not None and hasattr(self.view_sampler, 'get_current_max_target_views'):
                        current_max_tgt = self.view_sampler.get_current_max_target_views()
                    else:
                        current_max_tgt = self.target_image_num_range[1]
                    scaled_max_tgt = max(self.target_image_num_range[0], round(current_max_tgt * scale))
                    random_target_num = self.rng.randint(self.target_image_num_range[0], scaled_max_tgt)
                else:
                    random_target_num = None

                # Update sampler parameters
                self.sampler.update_parameters(
                    image_num=random_image_num,
                    target_num=random_target_num,
                    ps_h=random_ps_h,
                    num_cameras=random_num_cameras,
                )
                
                batch_size = 1

                # Collect samples for the current batch
                current_batch = []
                for _ in range(batch_size):
                    try:
                        item = next(sampler_iterator)  # item is (idx, aspect_ratio, image_num)
                        current_batch.append(item)
                    except StopIteration:
                        break  # No more samples

                if not current_batch:
                    break  # No more data to yield

                yield current_batch

            except StopIteration:
                break  # End of sampler's iterator

    def __len__(self):
        # Return a large dummy length
        return 1000000


class DynamicDistributedSampler(DistributedSampler):
    """
    Extends PyTorch's DistributedSampler to include dynamic aspect_ratio and image_num
    parameters, which can be passed into the dataset's __getitem__ method.
    """
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        self.image_num = None
        self.target_num = None
        self.ps_h = None
        self.num_cameras = None

    def __iter__(self):
        """
        Yields a sequence of (index, image_num, target_num, ps_h, num_cameras).
        Relies on the parent class's logic for shuffling/distributing
        the indices across replicas, then attaches extra parameters.
        """
        indices_iter = super().__iter__()

        for idx in indices_iter:
            yield (idx, self.image_num, self.target_num, self.ps_h, self.num_cameras)

    def update_parameters(self, image_num, target_num, ps_h, num_cameras=None):
        """
        Updates dynamic parameters for each new epoch or iteration.

        Args:
            image_num: The number of context images to set.
            target_num: The number of target images to set (or None if not randomized).
            ps_h: The patch size height.
            num_cameras: The number of cameras to use (or None if not randomized).
        """
        self.image_num = image_num
        self.target_num = target_num
        self.ps_h = ps_h
        self.num_cameras = num_cameras

class HomogeneousBatchSampler(BatchSampler):
    """Sample one batch from a selected dataset with given probability.
    Compatible with datasets at different resolution
    """

    def __init__(
        self, src_dataset_ls, batch_size, num_context_views, world_size=1, rank=0, prob=None, sampler=None, generator=None
    ):
        self.base_sampler = None
        self.batch_size = batch_size
        self.num_context_views = num_context_views
        self.world_size = world_size
        self.rank = rank
        self.drop_last = True
        self.generator = generator

        self.src_dataset_ls = src_dataset_ls
        self.n_dataset = len(self.src_dataset_ls)
        
        # Dataset length
        self.dataset_length = [len(ds) for ds in self.src_dataset_ls]
        self.cum_dataset_length = [
            sum(self.dataset_length[:i]) for i in range(self.n_dataset)
        ]  # cumulative dataset length
        
        # BatchSamplers for each source dataset
        self.src_batch_samplers = []
        for ds in self.src_dataset_ls:
            sampler = DynamicDistributedSampler(ds, num_replicas=self.world_size,
                                                rank=self.rank, seed=42, shuffle=True)
            sampler.set_epoch(0)

            if hasattr(ds, "epoch"):
                ds.epoch = 0
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(0)
            vs = ds.cfg.view_sampler
            if hasattr(vs, 'min_context_views'):
                min_ctx = vs.min_context_views
                max_ctx = vs.max_context_views
            else:
                min_ctx = max_ctx = vs.num_context_views
            if hasattr(vs, 'min_target_views'):
                min_tgt = vs.min_target_views
                max_tgt = vs.max_target_views
            elif hasattr(vs, 'num_target_views'):
                min_tgt = max_tgt = vs.num_target_views
            else:
                min_tgt = max_tgt = min_ctx
            min_cam = getattr(vs, 'min_cameras', 1)
            max_cam = getattr(vs, 'max_cameras', 1)
            batch_sampler = DynamicBatchSampler(
                    sampler,
                    [min_ctx, max_ctx],
                    ds.cfg.input_image_shape,
                    seed=42,
                    max_img_per_gpu=vs.max_img_per_gpu,
                    target_image_num_range=[min_tgt, max_tgt],
                    camera_range=[min_cam, max_cam] if max_cam > 1 else None,
                    view_sampler=ds.view_sampler,
            )
            self.src_batch_samplers.append(batch_sampler)

        for sampler in self.src_batch_samplers:
            sampler.epoch = 0
        self.batch_iterators = [iter(bs) for bs in self.src_batch_samplers]
        self.n_batches = [len(bs.sampler) for bs in self.src_batch_samplers]
        self.n_total_batch = sum(self.n_batches)

        # sampling probability
        if prob is None:
            # if not given, decide by dataset length
            self.prob = torch.ones(self.n_dataset) / self.n_dataset
        else:
            self.prob = torch.as_tensor(prob)
    
    def __iter__(self):
        """Yields batches of indices in the format of (sample_idx, feat_idx) tuples,
        where indices correspond to ConcatDataset of src_dataset_ls
        """
        idx_ds = torch.multinomial(
            self.prob, 1, generator=self.generator
        ).item()
        
        try:
            batch_raw = next(self.batch_iterators[idx_ds])
        except StopIteration:
            self.batch_iterators[idx_ds] = iter(self.src_batch_samplers[idx_ds])
            batch_raw = next(self.batch_iterators[idx_ds])

        # shift only the sample_idx by cumulative dataset length, keep feat_idx unchanged
        shift = self.cum_dataset_length[idx_ds]
        processed_sample = []
        for item in batch_raw:
            # Preserve all dynamic fields emitted by DynamicDistributedSampler:
            # (idx, num_context_views, num_target_views, ps_h).
            processed_item = (item[0] + shift, *item[1:])
            processed_sample.append(processed_item)

        yield [processed_sample] * self.batch_size
        
    def set_epoch(self, epoch):
        """Set epoch for all underlying BatchSamplers"""
        for sampler in self.src_batch_samplers:
            sampler.set_epoch(epoch)
        self.batch_iterators = [iter(bs) for bs in self.src_batch_samplers]

    def __len__(self):
        return self.n_total_batch