"""Video view sampler with bounded temporal gaps and warm-up schedules."""

from dataclasses import dataclass, field
from typing import Literal

import torch
from jaxtyping import Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerBoundedVideoCfg:
    name: Literal["bounded_video"]
    min_context_views: int
    max_context_views: int
    min_target_views: int  # Only used if sparse=True
    max_target_views: int  # Only used if sparse=True
    gap: int  # Maximum gap between two consecutive sampled frames
    min_gap: int  # Minimum gap between consecutive frames
    warm_up_steps: int
    sparse: bool = False  # If True, sample num_target_views random targets instead of all frames
    min_cameras: int = 1  # Minimum number of cameras to sample per batch (1 = no randomization)
    max_cameras: int = 1  # Maximum (base) cameras; when < max, more time frames are sampled proportionally
    exclude_cameras: list[int] = field(default_factory=list)  # Camera counts to never sample


class ViewSamplerBoundedVideo(ViewSampler[ViewSamplerBoundedVideoCfg]):
    """View sampler for video datasets with bounded gaps between context views."""

    def get_current_gap(self) -> int:
        return self._warmup_schedule(self.cfg.min_gap, self.cfg.gap, self.cfg.warm_up_steps)

    def get_current_max_cameras(self) -> int:
        return self._warmup_schedule(self.cfg.min_cameras, self.cfg.max_cameras, self.cfg.warm_up_steps)

    def get_current_max_context_views(self) -> int:
        return self._warmup_schedule(self.cfg.min_context_views, self.cfg.max_context_views, self.cfg.warm_up_steps)

    def get_current_max_target_views(self) -> int:
        return self._warmup_schedule(self.cfg.min_target_views, self.cfg.max_target_views, self.cfg.warm_up_steps)

    def _sample_context(
        self, num_context_views: int, total_frames: int, current_gap: int, max_span: int, device: torch.device
    ) -> Int64[Tensor, " context_view"]:
        """Sample context frame indices given a pre-validated gap and span."""

        # Sample a start index that guarantees the max-gap span fits in the video.
        start_idx = torch.randint(
            0, total_frames - max_span, size=(), device=device, generator=self.generator
        ).item()

        # Sample gaps between consecutive context frames, ensuring they are between min_gap and current_gap.
        gaps = torch.randint(
            self.cfg.min_gap, current_gap + 1, size=(num_context_views,), device=device, generator=self.generator
        )
        gaps[0] = 0 # First gap is 0 to include the start_idx as the first context view

        return start_idx + torch.cumsum(gaps, dim=0)

    def _sample_targets(
        self, index_context: Int64[Tensor, " view"], num_target_views: int, device: torch.device
    ) -> Int64[Tensor, " target_view"]:
        """Sample target indices between the first and last context frame."""
        first_idx = index_context[0].item()
        last_idx = index_context[-1].item()
        span = (last_idx - first_idx) + 1

        # Dense: Return all frames from fist to last context index.
        if not self.cfg.sparse:
            return torch.arange(first_idx, last_idx + 1, device=device)

        # Sparse: Pick a random, temporally-ordered subset
        desired_targets = num_target_views or self.get_current_max_target_views()
        num_targets = min(desired_targets, span)

        offsets = torch.randperm(span, device=device, generator=self.generator)[:num_targets]
        sorted_offsets = offsets.sort().values
        
        return first_idx + sorted_offsets

    def sample(
        self,
        scene: str,
        num_context_views: int,
        num_frames: int,
        device: torch.device = torch.device("cpu"),
        num_target_views: int = 0,
    ) -> tuple[
        Int64[Tensor, " context_view"],
        Int64[Tensor, " target_view"],
    ]:
        current_gap = self.get_current_gap()
        max_span = (num_context_views - 1) * current_gap

        if num_frames < max_span + 1:
            raise ValueError(
                f"Not enough frames for scene={scene}: need {max_span + 1} "
                f"to support current_gap={current_gap}, got {num_frames} "
                f"(num_context_views={num_context_views})"
            )

        index_context = self._sample_context(num_context_views, num_frames, current_gap, max_span, device)
        index_target = self._sample_targets(index_context, num_target_views, device)
        return index_context, index_target
