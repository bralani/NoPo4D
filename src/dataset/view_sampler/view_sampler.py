from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
from jaxtyping import Float, Int64
from torch import Tensor, Generator

from ...utils.step_tracker import StepTracker
from ..types import Stage

T = TypeVar("T")


class ViewSampler(ABC, Generic[T]):
    cfg: T
    stage: Stage
    is_overfitting: bool
    cameras_are_circular: bool
    step_tracker: StepTracker | None

    def __init__(
        self,
        cfg: T,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
        generator: Generator | None = None,
        batch_size: int = 1,
    ) -> None:
        self.cfg = cfg
        self.stage = stage
        self.is_overfitting = is_overfitting
        self.cameras_are_circular = cameras_are_circular
        self.step_tracker = step_tracker
        self.generator = generator
        self.batch_size = batch_size

    @abstractmethod
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
        pass

    @property
    @abstractmethod
    def num_target_views(self) -> int:
        pass

    @property
    @abstractmethod
    def num_context_views(self) -> int:
        pass

    @property
    def global_step(self) -> int:
        return 0 if self.step_tracker is None else self.step_tracker.get_step()
