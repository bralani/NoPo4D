"""Training-specific functionality for model wrapper."""

import gc
import time

import torch
from jaxtyping import Float
from torch import Tensor

from src.dataset.types import BatchedExample, BatchedViews
from src.cfg import get_cfg
from src.utils.step_tracker import StepTracker
from src.utils.geometry import affine_inverse
from src.utils.pose import apply_sim3_to_c2w, batch_align_poses_umeyama
from src.model.encoder.types import EncoderOutput
from src.training.distillation.types import DistillationOutput
from src.model.decoder.types import DecoderOutput
from src.training.base_wrapper import ModelWrapperBase
from src.training.loss.loss_computer import LossComputer


class TrainingMixin(ModelWrapperBase):
    """Mixin providing training step and visualization logic."""

    loss_computer: LossComputer
    step_tracker: StepTracker | None

    def on_train_epoch_start(self) -> None:
        """Propagate epoch index to dataset and sampler for deterministic shuffling."""
        if hasattr(self.trainer.datamodule.train_loader.dataset, "set_epoch"):
            self.trainer.datamodule.train_loader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.datamodule.train_loader.sampler, "set_epoch"):
            self.trainer.datamodule.train_loader.sampler.set_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):
        """Run one training iteration: distillation, encoder, decoder, loss, logging."""
        batch = self._prepare_batch(batch)
        num_cameras = int(batch["num_cameras"][0])
        context: BatchedViews = batch["context"]
        target: BatchedViews  = batch["target"]
        _, _, _, h, w = target["image"].shape

        # Distillation: optional teacher signal for depth/poses supervision.
        distill_infos = None
        if self.distill_manager is not None:
            distill_infos = self.distill_manager.run(
                image=context["image"],
                num_cameras=num_cameras,
                extrinsics=context["extrinsics"] if not self.train_cfg.pose_free else None,
                intrinsics=context["intrinsics"] if not self.train_cfg.pose_free else None,
            )

        # Encoder: predict Gaussians and camera poses from context views.
        encoder_out: EncoderOutput = self.model.encoder(
            images=context["image"],
            timestamps=context.get("timestamp"),
            num_cameras=num_cameras,
            input_extrinsics=context["extrinsics"] if not self.train_cfg.pose_free else None,
            input_intrinsics=context["intrinsics"] if not self.train_cfg.pose_free else None,
        )

        # For multi-camera scenes, run the encoder on context+target jointly and Umeyama-align to reference poses.
        target_extrinsics, target_intrinsics = self._get_target_poses(context, target, encoder_out, num_cameras)
        target_for_render = {**target, "extrinsics": target_extrinsics, "intrinsics": target_intrinsics}

        # Decoder: render context and target views in a single batched pass.
        output_context, output_target = self._render_all_views(encoder_out, context, target_for_render, (h, w))

        # Loss computation
        total_loss, visualization_cache = self.loss_computer.compute_total_loss(
            output_context=output_context,
            output_target=output_target,
            batch=batch,
            gaussians=encoder_out.gaussians,
            depth_dict=encoder_out.depth,
            distill_infos=distill_infos,
            encoder_output_context=encoder_out,
            logger_fn=self.log,
            global_step=self.global_step,
        )

        with torch.no_grad():
            self._log_and_visualize(
                batch=batch,
                output_context=output_context,
                output_target=output_target,
                encoder_out=encoder_out,
                distill_infos=distill_infos,
                total_loss=total_loss,
                visualization_cache=visualization_cache,
            )

        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        if self.global_step % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        return total_loss

    def _get_target_poses(
        self,
        context: BatchedViews,
        target: BatchedViews,
        encoder_out: EncoderOutput,
        num_cameras: int,
    ) -> tuple[
        Float[Tensor, "batch v_target 4 4"],  # extrinsics c2w
        Float[Tensor, "batch v_target 3 3"],  # intrinsics
    ]:
        """Resolve target poses: tile context poses for single-camera, or run the encoder
        on context+target jointly and Umeyama-align to reference poses for multi-camera."""
        ctx_extr   = encoder_out.camera_pose["extrinsic_c2w"]
        ctx_intr   = encoder_out.camera_pose["intrinsic"]
        v_target   = target["image"].shape[1]
        v_context  = ctx_extr.shape[1]

        if num_cameras == 1:
            repeats = (v_target + v_context - 1) // v_context
            return (
                ctx_extr.repeat(1, repeats, 1, 1)[:, :v_target],
                ctx_intr.repeat(1, repeats, 1, 1)[:, :v_target],
            )

        ctx_ts, tgt_ts = context.get("timestamp"), target.get("timestamp")
        combined_ts = (
            torch.cat([ctx_ts, tgt_ts], dim=1)
            if ctx_ts is not None and tgt_ts is not None
            else None
        )
        with torch.no_grad():
            combined_out: EncoderOutput = self.model.encoder(
                torch.cat([context["image"], target["image"]], dim=1),
                timestamps=combined_ts,
                run_gaussian_head=False,
                num_cameras=num_cameras,
                average_poses=False,
            )

        poses      = combined_out.camera_pose["extrinsic_c2w"]
        intrinsics = combined_out.camera_pose["intrinsic"]
        rots, trans, scales = batch_align_poses_umeyama(
            affine_inverse(ctx_extr).detach(),
            affine_inverse(poses[:, :v_context]).detach(),
        )
        return apply_sim3_to_c2w(rots, trans, scales, poses[:, v_context:]), intrinsics[:, v_context:]

    def _render_all_views(
        self,
        encoder_out: EncoderOutput,
        context: BatchedViews,
        target: BatchedViews,
        image_shape: tuple[int, int],
    ) -> tuple[DecoderOutput, DecoderOutput]:
        """Render context and target views in a single batched decoder pass."""
        ctx_extr = encoder_out.camera_pose["extrinsic_c2w"]
        ctx_intr = encoder_out.camera_pose["intrinsic"]
        ctx_ts, tgt_ts = context.get("timestamp"), target.get("timestamp")

        # Concatenate context and target along the view dimension for a single forward pass.
        out = self.model.decoder.forward(
            gaussians=encoder_out.gaussians,
            extrinsics=torch.cat([ctx_extr, target["extrinsics"]], dim=1),
            intrinsics=torch.cat([ctx_intr, target["intrinsics"]], dim=1),
            image_shape=image_shape,
            ts=torch.cat([ctx_ts, tgt_ts], dim=1) if ctx_ts is not None and tgt_ts is not None else None,
        )

        # Split the joint output back into context and target views.
        v = ctx_extr.shape[1]
        output_context = DecoderOutput(
            color=out.color[:, :v],
            depth=out.depth[:, :v] if out.depth is not None else None,
            alpha=out.alpha[:, :v] if out.alpha is not None else None,
        )
        output_target = DecoderOutput(
            color=out.color[:, v:],
            depth=out.depth[:, v:] if out.depth is not None else None,
            alpha=out.alpha[:, v:] if out.alpha is not None else None,
        )
        return output_context, output_target

    def _log_and_visualize(
        self,
        batch: BatchedExample,
        output_context: DecoderOutput,
        output_target: DecoderOutput,
        encoder_out: EncoderOutput,
        distill_infos: DistillationOutput | None,
        total_loss: Float[Tensor, ""],
        visualization_cache: dict | None = None,
    ) -> None:
        """Log metrics and, at validation frequency, a comparison image and rendered video."""
        self.log("info/global_step", self.global_step)
        self.metrics_logger.log_training_metrics(
            output=output_context,
            target_output=output_target,
            batch=batch,
            gaussians=encoder_out.gaussians,
            depth_dict=encoder_out.depth,
            distill_infos=distill_infos,
            logger_fn=self.log,
        )

        if self.global_rank == 0 and self.global_step % self.train_cfg.print_log_every_n_steps == 0:
            print(f"train step {self.global_step}; scene = {batch['scene'][0]}; loss = {total_loss:.6f}")

        if self.global_step % get_cfg()["trainer"]["val_check_interval"] == 0:
            self._log_comparison_and_video(
                batch=batch,
                output=output_context,
                depth_dict=encoder_out.depth,
                distill_infos=distill_infos,
                log_key="comparison_train",
                visualization_cache=visualization_cache,
                key="context",
            )

    def on_after_backward(self) -> None:
        """Log per-parameter RMS gradient norm."""
        grad_norms_sq = [
            p.grad.detach().norm(2).item() ** 2
            for p in self.parameters()
            if p.grad is not None
        ]
        if grad_norms_sq:
            self.log("loss/grad_norm", (sum(grad_norms_sq) / len(grad_norms_sq)) ** 0.5)
