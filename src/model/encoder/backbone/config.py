"""Backbone configuration base class."""

from abc import ABC
from dataclasses import dataclass


@dataclass
class BackboneCfg(ABC):
    backbone_checkpoint_name: str                    # pretrained model identifier
    intermediate_layer_idx: list[int]                # which transformer blocks to export tokens from
    backbone_temporal_encoding: bool                 # whether to inject sinusoidal time embeddings
    input_mean: tuple[float, float, float]           # per-channel normalization mean (RGB)
    input_std:  tuple[float, float, float]           # per-channel normalization std  (RGB)
