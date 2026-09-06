"""Abstract ViewSampler interface."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
from jaxtyping import Int64
from torch import Tensor, Generator

from ...utils.step_tracker import StepTracker

T = TypeVar("T")


class ViewSampler(ABC, Generic[T]):
    cfg: T
    step_tracker: StepTracker | None

    def __init__(
        self,
        cfg: T,
        step_tracker: StepTracker | None,
        generator: Generator | None = None,
    ) -> None:
        self.cfg = cfg
        self.step_tracker = step_tracker
        self.generator = generator

    def _warmup_schedule(self, start_val: int, end_val: int, warm_up_steps: int) -> int:
        """Linearly interpolate from start_val to end_val over warm_up_steps."""
        if warm_up_steps == 0 or self.global_step >= warm_up_steps:
            return end_val
        fraction = self.global_step / warm_up_steps
        return start_val + int((end_val - start_val) * fraction)

    @abstractmethod
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
        pass

    @property
    def global_step(self) -> int:
        return 0 if self.step_tracker is None else self.step_tracker.get_step()
