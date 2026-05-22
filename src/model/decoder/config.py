"""Decoder configuration dataclass."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Decoder4DGSCfg:
    """Configuration for Decoder4DGS."""
    name: Literal["4dgs"]
    background_color: list[float]
    chunk_size: int
