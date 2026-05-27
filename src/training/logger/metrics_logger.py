"""Metrics logging utilities for training and validation."""

from functools import cache
from pathlib import Path
from typing import Any, Callable

import torch
import torchvision
from einops import rearrange, reduce
from jaxtyping import Bool, Float
from lpips import LPIPS
from skimage.metrics import structural_similarity
from tabulate import tabulate
from torch import Tensor

from src.dataset.types import BatchedExample
from src.model.decoder.types import DecoderOutput
from src.model.types import Gaussians


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@cache
def _get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    value = _get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)


@torch.no_grad()
def abs_relative_difference(
    output: Float[Tensor, "batch height width"],
    target: Float[Tensor, "batch height width"],
    valid_mask: Bool[Tensor, "batch height width"] | None = None,
) -> Float[Tensor, ""]:
    diff = torch.abs(output - target) / target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    return (torch.sum(diff, (-1, -2)) / n).mean()


@torch.no_grad()
def _threshold_percentage(output, target, threshold_val, valid_mask=None):
    max_ratio = torch.max(output / target, target / output)
    bit_mat = (max_ratio < threshold_val).float()
    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    return (torch.sum(bit_mat, (-1, -2)) / n).mean()


@torch.no_grad()
def delta1_acc(
    pred: Float[Tensor, "batch height width"],
    gt: Float[Tensor, "batch height width"],
    valid_mask: Bool[Tensor, "batch height width"] | None = None,
) -> Float[Tensor, ""]:
    return _threshold_percentage(pred, gt, 1.25, valid_mask)


class MetricsLogger:
    """Handles computation and logging of training/validation metrics."""

    def __init__(self) -> None:
        """Initialize metrics logger with empty metric accumulators."""
        self.running_metrics: Optional[dict[str, float | Tensor]] = None
        self.running_metric_steps: int = 0
        self.running_metrics_sub: Optional[dict[str, dict[str, float | Tensor]]] = None
        self.running_metric_steps_sub: Optional[dict[str, int]] = None
        self._debug_image_counter: int = 0

    def log_training_metrics(
        self,
        output: DecoderOutput,
        target_output: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict[str, Float[Tensor, "batch view_context height width ..."]],
        distill_infos: dict[str, Any] | None,
        logger_fn: Callable[[str, float | Tensor], None]
    ) -> None:
        """Log training metrics including RGB quality, depth consistency, and Gaussian stats."""
        context_gt: Float[Tensor, "batch view channel height width"] = batch["context"]["image"]

        # Recompute scene scale as mean L2 norm of Gaussian means
        scene_scale = gaussians.means.flatten(0, 1).norm(dim=-1).mean().clip(min=1e-8)
        self._log_scalar(logger_fn, "train/scene_scale", scene_scale)

        # Log RGB metrics
        self._log_rgb_metrics(logger_fn, context_gt, output.color)
        self._log_target_rgb_metrics(logger_fn, batch["target"]["image"], target_output.color)

        # Log depth consistency metrics if available
        if self._has_depth_info(distill_infos, output):
            self._log_depth_metrics(
                logger_fn, output, depth_dict, distill_infos
            )

        # Log Gaussian statistics
        self._log_gaussian_stats(logger_fn, gaussians)

    def log_validation_metrics(
        self,
        rgb_gt: Float[Tensor, "batch_view channel height width"],
        rgb_pred: Float[Tensor, "batch_view channel height width"],
        output: DecoderOutput,
        depth_dict: dict[str, Float[Tensor, "batch view_context height width ..."]],
        distill_infos: dict[str, Any] | None,
        logger_fn: Callable[[str, float | Tensor], None],
    ) -> None:
        """Log validation metrics including RGB quality and depth consistency.

        Args:
            rgb_gt: Ground truth RGB images. Shape: (batch*view, channel, height, width).
            rgb_pred: Predicted RGB images. Shape: (batch*view, channel, height, width).
            output: Model output containing predicted depth.
            depth_dict: Dictionary with depth tensors.
            distill_infos: Distillation info with optional confidence masks.
            logger_fn: Logging function that takes (metric_name, value) pairs.
        """
        # Log RGB quality metrics
        self._log_scalar(logger_fn, "val/psnr", compute_psnr(rgb_gt, rgb_pred).mean())
        self._log_scalar(logger_fn, "val/lpips", compute_lpips(rgb_gt, rgb_pred).mean())
        self._log_scalar(logger_fn, "val/ssim", compute_ssim(rgb_gt, rgb_pred).mean())

        # Log depth consistency metrics
        self._log_validation_depth_metrics(logger_fn, output, depth_dict, distill_infos)
    
    @staticmethod
    def _log_scalar(
        logger_fn: Callable[[str, float | Tensor], None],
        name: str,
        value: float | Tensor
    ) -> None:
        """Log a single scalar metric."""
        logger_fn(name, value)
    
    @staticmethod
    def _has_depth_info(distill_infos: dict[str, Any] | None, output: DecoderOutput) -> bool:
        """Check if depth information is available for logging."""
        return distill_infos is not None and len(distill_infos.keys()) > 0 and output.depth is not None
    
    def _log_rgb_metrics(
        self,
        logger_fn: Callable[[str, float | Tensor], None],
        target_gt: Float[Tensor, "batch view channel height width"],
        predicted_color: Float[Tensor, "batch view channel height width"]
    ) -> None:
        """Log RGB quality metrics (PSNR)."""
        # Reshape from (batch, view, channel, height, width) to (batch*view, channel, height, width)
        target_flat = rearrange(target_gt, "b v c h w -> (b v) c h w")
        pred_flat = rearrange(predicted_color, "b v c h w -> (b v) c h w")
        
        psnr_probabilistic = compute_psnr(target_flat, pred_flat)
        self._log_scalar(logger_fn, "train/psnr_probabilistic", psnr_probabilistic.mean())

    def _log_target_rgb_metrics(
        self,
        logger_fn: Callable[[str, float | Tensor], None],
        target_gt: Float[Tensor, "batch view channel height width"],
        predicted_color: Float[Tensor, "batch view channel height width"]
    ) -> None:
        """Log RGB quality metrics for target views."""
        target_flat = rearrange(target_gt, "b v c h w -> (b v) c h w")
        pred_flat = rearrange(predicted_color, "b v c h w -> (b v) c h w")

        psnr_target = compute_psnr(target_flat, pred_flat)
        self._log_scalar(logger_fn, "train/psnr_target", psnr_target.mean())
    
    def _log_depth_metrics(
        self,
        logger_fn: Callable[[str, float | Tensor], None],
        output: DecoderOutput,
        depth_dict: dict[str, Float[Tensor, "batch view_context height width ..."]],
        distill_infos: dict[str, Any] | None
    ) -> None:
        """Log depth consistency metrics."""
        assert output.depth is not None, "Depth predictions must be available"
        
        # Extract depth predictions for context views
        pred_depth = output.depth  # Shape: (1, num_context, height, width)
        pred_depth_flat = rearrange(pred_depth, "b v h w -> (b v) h w")
        
        # Extract reference depth
        ref_depth = depth_dict['depth'].squeeze(-1)  # Shape: (batch, view, height, width)
        ref_depth_flat = rearrange(ref_depth, "b v h w -> (b v) h w")
        
        # Extract confidence mask
        conf_mask_flat = rearrange(distill_infos['conf_mask'], "b v h w -> (b v) h w")
        
        # Compute and log metrics
        consis_absrel = abs_relative_difference(pred_depth_flat, ref_depth_flat, conf_mask_flat)
        self._log_scalar(logger_fn, "train/consis_absrel", consis_absrel.mean())
        
        consis_delta1 = delta1_acc(pred_depth_flat, ref_depth_flat, conf_mask_flat)
        self._log_scalar(logger_fn, "train/consis_delta1", consis_delta1.mean())
    
    def _log_gaussian_stats(
        self,
        logger_fn: Callable[[str, float | Tensor], None],
        gaussians: Gaussians
    ) -> None:
        """Log Gaussian statistics including scales and quantiles."""
        # Log mean scale
        mean_scale = gaussians.scales[0].mean()
        self._log_scalar(logger_fn, "train/scales_mean", mean_scale)
        self._log_scalar(logger_fn, "train/num_gaussians", float(gaussians.means.shape[1]))
        
        # Log quantile statistics
        self._log_quantile_stats(gaussians, logger_fn)
    
    def _log_quantile_stats(
        self,
        gaussians: Gaussians,
        logger_fn: Callable[[str, float | Tensor], None]
    ) -> None:
        """Log quantile statistics for Gaussian properties (opacities, time, motion)."""
        quantiles_str = ["0.25", "0.5", "0.75", "0.9", "0.99"]
        quantiles = torch.tensor(
            [0.25, 0.5, 0.75, 0.9, 0.99],
            device=gaussians.means.device
        )

        # Log opacity quantiles
        opacities_quantiles = torch.quantile(gaussians.opacities[0], quantiles)
        for quantile_str, quantile_val in zip(quantiles_str, opacities_quantiles):
            self._log_scalar(
                logger_fn,
                f"train/opacities_quantile_{quantile_str}",
                quantile_val
            )

        # Log time-related quantiles (if available)
        if gaussians.t is not None:
            self._log_time_quantiles(gaussians, quantiles, quantiles_str, logger_fn)
    
    def _log_time_quantiles(
        self,
        gaussians: Gaussians,
        quantiles: Float[Tensor, "num_quantiles"],
        quantiles_str: list[str],
        logger_fn: Callable[[str, float | Tensor], None]
    ) -> None:
        """Log time-related quantile statistics for 4D Gaussians."""
        assert gaussians.t is not None, "Time attribute must be available"
        assert gaussians.cov_t is not None, "Time covariance must be available"
        assert gaussians.ms3_fwd is not None, "Forward motion speed attribute must be available"
        assert gaussians.ms3_bwd is not None, "Backward motion speed attribute must be available"
        assert gaussians.omega_fwd is not None, "Forward angular velocity attribute must be available"
        assert gaussians.omega_bwd is not None, "Backward angular velocity attribute must be available"

        # Log time quantiles
        t_quantiles = torch.quantile(gaussians.t[0].to(torch.float32), quantiles)
        for quantile_str, quantile_val in zip(quantiles_str, t_quantiles):
            self._log_scalar(logger_fn, f"train/t_quantile_{quantile_str}", quantile_val)

        # Log time covariance quantiles
        cov_t_quantiles = torch.quantile(gaussians.cov_t[0], quantiles)
        for quantile_str, quantile_val in zip(quantiles_str, cov_t_quantiles):
            self._log_scalar(logger_fn, f"train/cov_t_quantile_{quantile_str}", quantile_val)

        # Log forward/backward velocity quantiles
        for name, tensor in [
            ("ms3_fwd", gaussians.ms3_fwd),
            ("ms3_bwd", gaussians.ms3_bwd),
            ("omega_fwd", gaussians.omega_fwd),
            ("omega_bwd", gaussians.omega_bwd),
        ]:
            q = torch.quantile(tensor[0].norm(dim=-1), quantiles)
            for quantile_str, quantile_val in zip(quantiles_str, q):
                self._log_scalar(logger_fn, f"train/{name}_quantile_{quantile_str}", quantile_val)
    
    def _log_validation_depth_metrics(
        self,
        logger_fn: Callable[[str, float | Tensor], None],
        output: DecoderOutput,
        depth_dict: dict[str, Float[Tensor, "batch view_context height width ..."]],
        distill_infos: dict[str, Any] | None
    ) -> None:
        """Log validation depth consistency metrics.
        
        Args:
            logger_fn: Logging function.
            output: Model output with depth predictions.
            depth_dict: Dictionary with reference depth tensors.
            distill_infos: Distillation info with optional confidence masks.
        """
        assert output.depth is not None, "Depth predictions must be available"
        
        # Reshape depth tensors
        pred_depth_flat = rearrange(output.depth, "b v h w -> (b v) h w")
        ref_depth_flat = rearrange(depth_dict['depth'].squeeze(-1), "b v h w -> (b v) h w")
        
        # Create valid mask (all ones)
        valid_mask = rearrange(
            torch.ones_like(output.depth, device=output.depth.device, dtype=torch.bool),
            "b v h w -> (b v) h w"
        )
        
        # Compute and log metrics
        consis_absrel = abs_relative_difference(pred_depth_flat, ref_depth_flat)
        self._log_scalar(logger_fn, "val/consis_absrel", consis_absrel.mean())
        
        consis_delta1 = delta1_acc(pred_depth_flat, ref_depth_flat, valid_mask=valid_mask)
        self._log_scalar(logger_fn, "val/consis_delta1", consis_delta1.mean())
        
        # Log depth MSE if confidence mask is available
        if distill_infos is not None and 'conf_mask' in distill_infos:
            diff_map = torch.abs(output.depth - depth_dict['depth'].squeeze(-1))
            masked_diff = diff_map[distill_infos['conf_mask']]
            self._log_scalar(logger_fn, "val/consis_mse", masked_diff.mean())
    
    def print_preview_metrics(
        self,
        metrics: dict[str, float | Tensor],
        methods: list[str] | None = None,
        overlap_tag: str | None = None,
    ) -> None:
        """Print and accumulate preview metrics with running averages.
        
        Maintains running averages of metrics across multiple calls, both overall
        and grouped by overlap categories. Prints formatted tables of results.

        Args:
            metrics: Dictionary of metric values (e.g., {'psnr_ours': 28.5, ...}).
            methods: List of method names to display in table. Defaults to ['ours'].
            overlap_tag: Optional category tag for grouping metrics (e.g., 'high_overlap').
        """
        # Update overall running metrics
        self._update_running_metrics(metrics)

        # Update sub-metrics by category if tag provided
        if overlap_tag is not None:
            self._update_sub_metrics(metrics, overlap_tag)

        # Print formatted metric tables
        self._print_metric_tables(methods, overlap_tag)
    
    def _update_running_metrics(self, metrics: dict[str, float | Tensor]) -> None:
        """Update running average of overall metrics.
        
        Args:
            metrics: New metric values to incorporate.
        """
        if self.running_metrics is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            steps = self.running_metric_steps
            self.running_metrics = {
                key: ((steps * value) + metrics[key]) / (steps + 1)
                for key, value in self.running_metrics.items()
            }
            self.running_metric_steps += 1
    
    def _update_sub_metrics(self, metrics: dict[str, float | Tensor], tag: str) -> None:
        """Update running average of metrics grouped by category tag.
        
        Args:
            metrics: New metric values to incorporate.
            tag: Category tag for grouping (e.g., 'high_overlap', 'low_overlap').
        """
        # Initialize sub-metrics if needed
        if self.running_metrics_sub is None:
            self.running_metrics_sub = {}
        if self.running_metric_steps_sub is None:
            self.running_metric_steps_sub = {}
        
        # Initialize or update metrics for this tag
        if tag not in self.running_metrics_sub:
            self.running_metrics_sub[tag] = metrics
            self.running_metric_steps_sub[tag] = 1
        else:
            steps = self.running_metric_steps_sub[tag]
            self.running_metrics_sub[tag] = {
                key: ((steps * value) + metrics[key]) / (steps + 1)
                for key, value in self.running_metrics_sub[tag].items()
            }
            self.running_metric_steps_sub[tag] += 1
    
    def _print_metric_tables(
        self,
        methods: list[str] | None,
        overlap_tag: str | None
    ) -> None:
        """Print formatted tables of accumulated metrics.
        
        Args:
            methods: List of method names to display.
            overlap_tag: If provided, also print per-category metrics.
        """
        if self.running_metrics is not None:
            print("All Pairs:")
            self._print_metrics_table(self.running_metrics, methods)

        if overlap_tag is not None and self.running_metrics_sub is not None:
            for category, category_metrics in self.running_metrics_sub.items():
                print(f"\nOverlap: {category}")
                self._print_metrics_table(category_metrics, methods)
    
    @staticmethod
    def _print_metrics_table(
        metrics: dict[str, float | Tensor],
        methods: list[str] | None = None
    ) -> None:
        """Print a formatted table of metrics for specified methods.
        
        Args:
            metrics: Dictionary of metric values.
            methods: List of method names. Defaults to ['ours'].
        """
        if methods is None:
            methods = ['ours']

        metric_names = ["psnr", "lpips", "ssim"]
        
        # Build table rows
        table_rows = []
        for method in methods:
            row = [method]
            for metric_name in metric_names:
                metric_key = f"{metric_name}_{method}"
                metric_value = metrics.get(metric_key, 0.0)
                row.append(f"{metric_value:.3f}")
            table_rows.append(row)

        # Print formatted table
        headers = ["Method"] + metric_names
        table_str = tabulate(table_rows, headers=headers)
        print(table_str)
