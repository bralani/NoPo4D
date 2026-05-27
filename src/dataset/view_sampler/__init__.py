from typing import Any
from torch import Generator

from ...utils.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler
from .view_sampler_bounded_video import ViewSamplerBoundedVideo, ViewSamplerBoundedVideoCfg

VIEW_SAMPLERS: dict[str, ViewSampler[Any]] = {
    "bounded_video": ViewSamplerBoundedVideo,
}

ViewSamplerCfg = ViewSamplerBoundedVideoCfg


def get_view_sampler(
    cfg: ViewSamplerCfg,
    stage: Stage,
    overfit: bool,
    cameras_are_circular: bool,
    step_tracker: StepTracker | None,
    generator: Generator | None = None,
    batch_size: int = 1,
) -> ViewSampler[Any]:
    return VIEW_SAMPLERS[cfg.name](
        cfg,
        stage,
        overfit,
        cameras_are_circular,
        step_tracker,
        generator=generator,
        batch_size=batch_size,
    )
