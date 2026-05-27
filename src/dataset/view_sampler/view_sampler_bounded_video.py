from dataclasses import dataclass, field
from typing import Literal

import torch
from jaxtyping import Float, Int64
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
    initial_gap: int  # Gap to use during initial warm-up steps
    max_img_per_gpu: int
    sparse: bool = False  # If True, sample num_target_views random targets instead of all frames
    min_cameras: int = 1  # Minimum number of cameras to sample per batch (1 = no randomization)
    max_cameras: int = 1  # Maximum (base) cameras; when < max, more time frames are sampled proportionally
    exclude_cameras: list[int] = field(default_factory=list)  # Camera counts to never sample

class ViewSamplerBoundedVideo(ViewSampler[ViewSamplerBoundedVideoCfg]):
    """View sampler for video datasets with bounded gaps between context views."""

    def get_current_gap(self) -> int:
        """Compute the current gap based on warm-up schedule.

        Returns:
            int: Current maximum gap between consecutive frames.
        """
        if self.global_step >= self.cfg.warm_up_steps:
            return self.cfg.gap

        fraction = self.global_step / self.cfg.warm_up_steps
        current_gap = self.cfg.initial_gap + int(
            (self.cfg.gap - self.cfg.initial_gap) * fraction
        )
        return current_gap

    def get_current_max_cameras(self) -> int:
        """Compute the current max cameras based on warm-up schedule."""
        if self.cfg.warm_up_steps == 0 or self.global_step >= self.cfg.warm_up_steps:
            return self.cfg.max_cameras
        fraction = self.global_step / self.cfg.warm_up_steps
        current_max = self.cfg.min_cameras + int(
            (self.cfg.max_cameras - self.cfg.min_cameras) * fraction
        )
        return max(current_max, self.cfg.min_cameras)

    def get_current_max_context_views(self) -> int:
        """Compute the current max context views based on warm-up schedule."""
        if self.cfg.warm_up_steps == 0 or self.global_step >= self.cfg.warm_up_steps:
            return self.cfg.max_context_views
        fraction = self.global_step / self.cfg.warm_up_steps
        current_max = self.cfg.min_context_views + int(
            (self.cfg.max_context_views - self.cfg.min_context_views) * fraction
        )
        return max(current_max, self.cfg.min_context_views)

    def get_current_max_target_views(self) -> int:
        """Compute the current max target views based on warm-up schedule."""
        if self.cfg.warm_up_steps == 0 or self.global_step >= self.cfg.warm_up_steps:
            return self.cfg.max_target_views
        fraction = self.global_step / self.cfg.warm_up_steps
        current_max = self.cfg.min_target_views + int(
            (self.cfg.max_target_views - self.cfg.min_target_views) * fraction
        )
        return max(current_max, self.cfg.min_target_views)

    def sample(
        self,
        scene: str,
        num_context_views: int,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        num_target_views: int = 0,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
        Float[Tensor, " overlap"],  # overlap
    ]:
        num_views, _, _ = extrinsics.shape


        if self.cameras_are_circular:
            raise NotImplementedError(
                "Circular datasets are not supported for ViewSamplerBoundedVideo."
            )

        # Get the current gap based on warm-up schedule
        current_gap = self.get_current_gap()
        if current_gap < self.cfg.min_gap:
            current_gap = self.cfg.min_gap
        
        # Build context indices without post-hoc clamping.
        num_gaps = max(0, num_context_views - 1)
        min_required_span = num_gaps * self.cfg.min_gap
        if num_views < min_required_span + 1:
            raise ValueError(
                f"Example does not have enough frames! "
                f"Need at least {min_required_span + 1} frames for context views with "
                f"num_context_views={num_context_views}, min_gap={self.cfg.min_gap}, got {num_views}"
            )

        # Sample a start that guarantees at least the minimum span fits.
        max_start_idx = max(0, num_views - 1 - min_required_span)
        start_idx = torch.randint(
            low=0,
            high=max(1, max_start_idx + 1),
            size=tuple(),
            device=device,
            generator=self.generator,
        ).item()

        if num_gaps == 0:
            index_context = torch.tensor([start_idx], dtype=torch.int64, device=device)
        else:
            # Allocate extra gap budget while respecting both per-gap max and video bounds.
            available_span = (num_views - 1) - start_idx
            base_span = num_gaps * self.cfg.min_gap
            gap_extra_cap = max(0, current_gap - self.cfg.min_gap)
            total_extra_cap = min(num_gaps * gap_extra_cap, available_span - base_span)

            if total_extra_cap < 0:
                raise ValueError(
                    f"Invalid span budget for scene={scene}: available_span={available_span}, base_span={base_span}"
                )

            context_gaps = torch.full(
                (num_gaps,),
                self.cfg.min_gap,
                dtype=torch.int64,
                device=device,
            )

            if total_extra_cap > 0 and gap_extra_cap > 0:
                total_extra = torch.randint(
                    low=0,
                    high=total_extra_cap + 1,
                    size=tuple(),
                    device=device,
                    generator=self.generator,
                ).item()

                extras = torch.zeros((num_gaps,), dtype=torch.int64, device=device)
                while total_extra > 0:
                    gap_idx = torch.randint(
                        low=0,
                        high=num_gaps,
                        size=tuple(),
                        device=device,
                        generator=self.generator,
                    ).item()
                    if extras[gap_idx].item() < gap_extra_cap:
                        extras[gap_idx] += 1
                        total_extra -= 1

                context_gaps = context_gaps + extras

            cumulative_gaps = torch.cumsum(context_gaps, dim=0)
            index_context = torch.cat(
                [
                    torch.tensor([start_idx], dtype=torch.int64, device=device),
                    start_idx + cumulative_gaps,
                ]
            )

        if (index_context[1:] <= index_context[:-1]).any():
            raise ValueError(f"Non-increasing context indices sampled: {index_context.tolist()}")
        if (index_context < 0).any() or (index_context >= num_views).any():
            raise ValueError(
                f"Out-of-bounds context indices sampled: {index_context.tolist()} for num_views={num_views}"
            )
        
        # Sample target views
        if self.cfg.sparse:
            # Sparse mode: sample num_target_views random targets between first and last context
            first_ctx = index_context[0].item()
            last_ctx = index_context[-1].item()
            all_indices = torch.arange(first_ctx, last_ctx + 1, device=device)
            n_targets = num_target_views if num_target_views > 0 else self.cfg.max_target_views
            if n_targets <= 0:
                n_targets = 1
            n_targets = min(n_targets, len(all_indices))
            perm = torch.randperm(len(all_indices), device=device, generator=self.generator)[:n_targets]
            index_target = all_indices[perm].sort().values
        else:
            # Dense mode: target views are all frames from the first to the last context view
            index_target = torch.arange(
                start=index_context[0].item(),
                end=(index_context[-1] + 1).item(),
                device=device,
            )

        overlap = torch.tensor([0.5], dtype=torch.float32, device=device)  # dummy

        return (index_context, index_target, overlap)

    @property
    def num_context_views(self) -> int:
        return self.cfg.max_context_views
    
    @property
    def num_target_views(self) -> int:
        return 0 # Dummy, not used in this sampler
