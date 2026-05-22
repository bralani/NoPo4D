import os
import sys

_da3_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Depth-Anything-3", "src")
if _da3_src not in sys.path:
    sys.path.insert(0, _da3_src)

from typing import Union

from .backbone_da3 import BackboneDA3, BackboneDA3Cfg
from .types import Backbone

BACKBONES = {
    "da3": BackboneDA3,
}

BACKBONE_CFGS = {
    "da3": BackboneDA3Cfg,
}

BackboneCfgUnion = Union[BackboneDA3Cfg]


def get_backbone(cfg: BackboneCfgUnion) -> Backbone:
    return BACKBONES[cfg.name](cfg)
