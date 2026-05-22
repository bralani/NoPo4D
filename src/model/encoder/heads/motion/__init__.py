"""Velocity branch for motion encoding.

This package contains the motion encoder that uses self-attention
to capture motion between consecutive frames for velocity prediction.
"""

from .motion_encoder import MotionEncoder, MotionEncoderCfg

__all__ = [
    "MotionEncoder",
    "MotionEncoderCfg",
]
