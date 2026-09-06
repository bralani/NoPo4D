"""View sampler registry and factory."""

from typing import Any
from torch import Generator

from ...utils.step_tracker import StepTracker
from .view_sampler import ViewSampler
from .view_sampler_bounded_video import ViewSamplerBoundedVideo, ViewSamplerBoundedVideoCfg

VIEW_SAMPLERS: dict[str, ViewSampler[Any]] = {
    "bounded_video": ViewSamplerBoundedVideo,
}

ViewSamplerCfg = ViewSamplerBoundedVideoCfg


def get_view_sampler(
    cfg: ViewSamplerCfg,
    step_tracker: StepTracker | None,
    generator: Generator | None = None,
) -> ViewSampler[Any]:
    return VIEW_SAMPLERS[cfg.name](cfg, step_tracker, generator=generator)
