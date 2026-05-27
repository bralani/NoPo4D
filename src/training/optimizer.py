"""AdamW optimizer with per-parameter learning rates and a linear warm-up followed by cosine decay."""

from dataclasses import dataclass, field
from typing import TypedDict
import re

import torch

from src.cfg import get_cfg


@dataclass
class OptimizerCfg:
    lr: float                                        # base learning rate; used as default and as cosine decay anchor
    warm_up_steps: int                               # steps to ramp lr from near-zero before cosine decay starts
    param_lr_rules: list[list[str | float]] = field(default_factory=list)  # [[regex, lr], ...] — first match wins
    verbose: bool = False                            # print the per-parameter LR assignment table at startup


class LRSchedulerConfig(TypedDict):
    """Lightning-expected format for the lr_scheduler entry in configure_optimizers."""
    scheduler: torch.optim.lr_scheduler.LRScheduler
    interval: str   # "step" or "epoch" — we use step-level updates
    frequency: int  # how often Lightning calls scheduler.step()


class OptimizerConfig(TypedDict):
    """Return type of configure_optimizers, as expected by Lightning."""
    optimizer: torch.optim.Optimizer
    lr_scheduler: LRSchedulerConfig


class Optimizer:
    """Builds and owns the AdamW optimizer and its LR scheduler for a given model."""

    def __init__(self, model: torch.nn.Module, cfg: OptimizerCfg) -> None:
        self.cfg = cfg
        param_groups = self._assign_lr_groups(model)
        self.optimizer = self._build_optimizer(param_groups)
        self.scheduler = self._build_scheduler()

    def _assign_lr_groups(
        self, model: torch.nn.Module
    ) -> dict[float, list[torch.nn.Parameter]]:
        """Match each trainable param against param_lr_rules; fall back to cfg.lr."""
        groups: dict[float, list[torch.nn.Parameter]] = {}
        assignments: list[tuple[str, float, str]] = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            lr, matched_rule = self.cfg.lr, "default"
            for pattern, rule_lr in self.cfg.param_lr_rules:
                if re.search(pattern, name):
                    lr, matched_rule = float(rule_lr), pattern
                    break
            groups.setdefault(lr, []).append(param)
            assignments.append((name, lr, matched_rule))

        if self.cfg.verbose:
            print("Parameter LR assignments:")
            for name, lr, rule in assignments:
                print(f"  {name:<60} {lr:.2e}  [{rule}]")
            print()

        return groups

    def _build_optimizer(
        self, param_groups: dict[float, list[torch.nn.Parameter]]
    ) -> torch.optim.AdamW:
        """Build AdamW with one param group per distinct learning rate."""
        return torch.optim.AdamW(
            [{"params": params, "lr": lr} for lr, params in param_groups.items()],
            lr=self.cfg.lr,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.SequentialLR:
        """Warm-up from lr/warm_up_steps to lr, then cosine-decay to 10% of lr."""
        warm_up = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1 / self.cfg.warm_up_steps,
            end_factor=1.0,
            total_iters=self.cfg.warm_up_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=get_cfg()["trainer"]["max_steps"],
            eta_min=self.cfg.lr * 0.1,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warm_up, cosine],
            milestones=[self.cfg.warm_up_steps],
        )

    def to_lightning(self) -> OptimizerConfig:
        """Return the Lightning-compatible dict for configure_optimizers."""
        return OptimizerConfig(
            optimizer=self.optimizer,
            lr_scheduler=LRSchedulerConfig(
                scheduler=self.scheduler, interval="step", frequency=1
            ),
        )


def build_optimizer_and_scheduler(model: torch.nn.Module, cfg: OptimizerCfg) -> OptimizerConfig:
    return Optimizer(model, cfg).to_lightning()
