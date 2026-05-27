"""Validation-specific functionality for model wrapper."""

from lightning.pytorch.utilities import rank_zero_only

from src.dataset.types import BatchedExample
from src.training.base_wrapper import ModelWrapperBase


class ValidationMixin(ModelWrapperBase):
    """Mixin providing validation step logic."""

    def on_validation_epoch_start(self) -> None:
        print(f"Validation epoch start on rank {self.trainer.global_rank}")
        # Propagate epoch index so the validation dataset/sampler can shuffle deterministically.
        if hasattr(self.trainer.datamodule.val_loader.dataset, "set_epoch"):
            self.trainer.datamodule.val_loader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.datamodule.val_loader.sampler, "set_epoch"):
            self.trainer.datamodule.val_loader.sampler.set_epoch(self.current_epoch)

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch: BatchedExample = self._prepare_batch(batch)

        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1, "validation expects batch size 1"
        num_cameras = int(batch.get("num_cameras", [1])[0])
        target_image = batch["target"]["image"]
        timestamps = batch["target"].get("timestamp")
        target_extrinsics = batch["target"].get("extrinsics")
        target_intrinsics = batch["target"].get("intrinsics")

        print(
            f"validation step {self.global_step}; "
            f"scene = {batch['scene']}; "
            f"context = {batch['target']['index'].tolist()}"
        )

        # Run teacher model to get pseudo-GT depth for visualization.
        distill_infos = None
        if self.distill_manager is not None:
            distill_infos = self.distill_manager.run(
                target_image,
                num_cameras=num_cameras,
                extrinsics=target_extrinsics,
                intrinsics=target_intrinsics,
            )

        # Predict Gaussians and camera poses from the target views.
        encoder_output = self.model(
            target_image,
            timestamps=timestamps,
            num_cameras=num_cameras,
            input_extrinsics=target_extrinsics if not self.train_cfg.pose_free else None,
            input_intrinsics=target_intrinsics if not self.train_cfg.pose_free else None,
        )

        # Render target views from the predicted Gaussians.
        output = self.model.render(
            encoder_output.gaussians,
            extrinsics=encoder_output.camera_pose["extrinsic_c2w"],
            intrinsics=encoder_output.camera_pose["intrinsic"],
            image_shape=(h, w),
            timestamps=timestamps,
        )

        # Log image quality and depth metrics.
        self.log("val/GS_num", h * w * v)
        self.metrics_logger.log_validation_metrics(
            rgb_gt=target_image[0].float(),
            rgb_pred=output.color[0].float(),
            output=output,
            depth_dict=encoder_output.depth,
            distill_infos=distill_infos,
            logger_fn=self.log,
        )

        # Log side-by-side comparison image and interpolated video.
        self._log_comparison_and_video(
            batch=batch,
            output=output,
            depth_dict=encoder_output.depth,
            distill_infos=distill_infos,
            log_key="comparison_val",
            key="target",
        )
