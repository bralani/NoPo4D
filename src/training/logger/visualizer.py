"""Validation visualization utilities."""

from typing import Any
import torch
from src.utils.image import inverse_normalize, vis_depth_map
from src.utils.geometry import get_normal_map
from src.training.logger.visualization.annotation import add_label
from src.training.logger.visualization.layout import hcat, vcat


class Visualizer:
    """Handles creation of validation comparison images."""

    @staticmethod
    def create_comparison_image(
        batch: Any,
        output: Any,
        depth_dict: dict,
        distill_infos: Any,
        visualization_cache: dict | None = None,
        key="target"
    ) -> torch.Tensor:
        """Create comparison image for validation.

        Args:
            batch: Batched example with context images.
            output: Model output with color and depth.
            depth_dict: Dictionary with depth tensors.
            distill_infos: Distillation info with confidence masks.
            key: Key to access target data in batch.

        Returns:
            comparison: Combined comparison image tensor.
        """
        rgb_pred = output.color.flatten(0, 1).float()
        depth_pred = vis_depth_map(output.depth.flatten(0, 1))

        # Compute depth difference map
        diff_map = torch.abs(output.depth - depth_dict['depth'].squeeze(-1))
        colored_diff_map = vis_depth_map(
            diff_map.flatten(0, 1),
            near=torch.tensor(1e-4, device=diff_map.device),
            far=torch.tensor(1.0, device=diff_map.device)
        )

        # Model depth prediction
        model_depth_pred = depth_dict["depth"].flatten(0, 1)
        model_depth_pred = vis_depth_map(model_depth_pred)

        pseudo_gt_depth = None
        if distill_infos is not None and 'depth_map' in distill_infos and distill_infos['depth_map'] is not None:
            pseudo_gt_depth = distill_infos['depth_map'].flatten(0, 1).squeeze(-1)
            pseudo_gt_depth = vis_depth_map(pseudo_gt_depth)

        # Ground truth
        rgb_gt = batch[key]["image"].flatten(0, 1).float()

        # Normal maps
        render_normal = (
            get_normal_map(
                output.depth.flatten(0, 1),
                batch[key]["intrinsics"].flatten(0, 1)
            ).permute(0, 3, 1, 2) + 1
        ) / 2.

        pred_normal = (
            get_normal_map(
                depth_dict['depth'].flatten(0, 1).squeeze(-1),
                batch[key]["intrinsics"].flatten(0, 1)
            ).permute(0, 3, 1, 2) + 1
        ) / 2.

        # Combine into comparison image
        comparison_items = [
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_pred), "Target (Prediction)"),
            add_label(vcat(*depth_pred), "Depth (Prediction)"),
            add_label(vcat(*model_depth_pred), "Depth (head Prediction)"),
        ]
        if pseudo_gt_depth is not None:
            comparison_items.append(add_label(vcat(*pseudo_gt_depth), "Depth (Pseudo GT)"))
        comparison_items.extend([
            add_label(vcat(*render_normal), "Normal (Prediction)"),
            add_label(vcat(*pred_normal), "Normal (head Prediction)"),
            add_label(vcat(*colored_diff_map), "Diff Map"),
        ])
        
        # Add optical flow visualizations if available
        if visualization_cache is not None:
            if "optical_flow_gt_vis_fwd" in visualization_cache and "optical_flow_pred_vis_fwd" in visualization_cache:
                comparison_items.extend([
                    add_label(vcat(*visualization_cache["optical_flow_gt_vis_fwd"]), "Optical Flow Fwd (GT)"),
                    add_label(vcat(*visualization_cache["optical_flow_pred_vis_fwd"]), "Optical Flow Fwd (Pred)"),
                ])
            if "optical_flow_gt_vis_bwd" in visualization_cache and "optical_flow_pred_vis_bwd" in visualization_cache:
                comparison_items.extend([
                    add_label(vcat(*visualization_cache["optical_flow_gt_vis_bwd"]), "Optical Flow Bwd (GT)"),
                    add_label(vcat(*visualization_cache["optical_flow_pred_vis_bwd"]), "Optical Flow Bwd (Pred)"),
                ])
        
        comparison = hcat(*comparison_items)

        # Downsample for logging
        comparison = torch.nn.functional.interpolate(
            comparison.unsqueeze(0),
            scale_factor=0.5,
            mode='bicubic',
            align_corners=False
        ).squeeze(0)

        return comparison

    @staticmethod
    def create_test_comparison_image(
        batch: Any,
        rgb_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Create comparison image for test output.

        Args:
            batch: Batched example with context images.
            rgb_pred: Predicted RGB images.

        Returns:
            comparison: Combined comparison image tensor.
        """
        context_img = inverse_normalize(batch["context"]["image"][0])
        rgb_gt = batch["target"]["image"][0]

        comparison = hcat(
            add_label(vcat(*context_img), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_pred), "Target (Prediction)"),
        )

        return comparison
