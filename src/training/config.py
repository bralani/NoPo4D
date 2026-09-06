from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainCfg:
    """Training-specific options and weights.

    Contains visualization flags, distillation/pose weights and other
    hyper-parameters referenced by the training loop.
    """
    output_path: Path
    print_log_every_n_steps: int
    warmup_steps_regularization: int = 0
    pose_free: bool = True
    freeze_module_regex: list[str] = field(default_factory=list)


__all__ = ["TrainCfg"]
