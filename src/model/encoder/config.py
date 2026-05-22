"""EncoderNoPo4DCfg: top-level encoder configuration dataclass."""

from typing import Any, Literal
from dataclasses import dataclass, field

from .heads.gaussian import GaussianAdapterCfg, GaussianHeadCfg
from .heads.motion import MotionEncoderCfg
from .backbone import BackboneCfgUnion, BACKBONE_CFGS


@dataclass
class EncoderNoPo4DCfg:
    # Model identity
    name: Literal["nopo4d"]

    # Backbone config
    backbone: BackboneCfgUnion

    # Sub-module configs
    gaussian_adapter: GaussianAdapterCfg
    gaussian_head: GaussianHeadCfg = field(default_factory=GaussianHeadCfg)
    motion_encoder: MotionEncoderCfg = field(default_factory=MotionEncoderCfg)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EncoderNoPo4DCfg":
        """Convert a plain dict to EncoderNoPo4DCfg."""
        d = dict(d)
        backbone_d = d.pop("backbone")
        if isinstance(backbone_d, dict):
            backbone_cls = BACKBONE_CFGS[backbone_d["name"]]
            backbone_d = backbone_cls(**backbone_d)
        d["backbone"] = backbone_d
        for key, cls_ in (
            ("gaussian_adapter", GaussianAdapterCfg),
            ("gaussian_head", GaussianHeadCfg),
            ("motion_encoder", MotionEncoderCfg),
        ):
            if key in d and isinstance(d[key], dict):
                d[key] = cls_(**d[key])
        return cls(**d)
