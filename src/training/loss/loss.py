from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Any, Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from src.dataset.types import BatchedExample
from src.model.decoder.types import DecoderOutput
from src.model.encoder.types import EncoderOutput
from src.model.types import Gaussians

T_cfg = TypeVar("T_cfg")
T_wrapper = TypeVar("T_wrapper")


class Loss(nn.Module, ABC, Generic[T_cfg, T_wrapper]):
    cfg: T_cfg
    name: str

    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__()
        
        # Extract the configuration from the wrapper.
        (field,) = fields(type(cfg))
        self.cfg = getattr(cfg, field.name)
        self.name = field.name

    @abstractmethod
    def forward(
        self,
        prediction_context: DecoderOutput,
        prediction_target: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict,
        global_step: int,
        warmup_steps: int = 0,
        encoder_output_context: EncoderOutput | None = None,
        distill_infos: Any | None = None,
        visualization_cache: dict | None = None,
    ) -> Float[Tensor, ""]: ...
