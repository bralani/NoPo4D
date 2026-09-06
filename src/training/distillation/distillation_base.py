from typing import Iterable

import torch
from torch import nn


class DistillationBase:
    """Base class for distillation utilities.

    Provides common utilities for freezing modules and moving them between
    devices (CPU/GPU) during teacher inference.
    """

    @staticmethod
    def _freeze_and_move_to_cpu(modules: Iterable[nn.Module]) -> None:
        """Freeze parameters and move to CPU."""
        for module in modules:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False
                param.data = param.data.cpu()

    @staticmethod
    def _move_to_device(device: torch.device, modules: Iterable[nn.Module]) -> None:
        """Move modules to specified device."""
        for module in modules:
            if module is None:
                continue
            module.to(device, non_blocking=True)

    @staticmethod
    def _move_to_cpu(modules: Iterable[nn.Module]) -> None:
        """Move modules to CPU."""
        for module in modules:
            if module is None:
                continue
            for param in module.parameters():
                param.data = param.data.cpu()
