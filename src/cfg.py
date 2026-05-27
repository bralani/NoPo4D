from typing import Optional
from omegaconf import DictConfig

_cfg: Optional[DictConfig] = None


def set_cfg(new_cfg: DictConfig) -> None:
    global _cfg
    _cfg = new_cfg


def get_cfg() -> DictConfig:
    global _cfg
    return _cfg
