"""NoPo4D encoder: lifts multi-view images into 3D/4D Gaussians."""

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from src.utils.geometry import (
    affine_inverse,
    normalize_extrinsics,
    extri_intri_to_pose_encoding,
    pose_encoding_to_extri_intri,
)

from .heads.gaussian import GaussianAdapter, GaussianHead
from .heads.motion import MotionEncoder
from .backbone.types import Backbone, LayerTokens
from .config import EncoderNoPo4DCfg
from .types import CameraPoseDict, DepthDict, Encoder, EncoderOutput, SceneInput


class EncoderNoPo4D(Encoder[EncoderNoPo4DCfg]):
    """Encodes multi-view images into 3D/4D Gaussians with predicted camera poses."""

    def __init__(
        self,
        cfg: EncoderNoPo4DCfg,
        backbone: Backbone,
    ) -> None:
        super().__init__(cfg)
        self.backbone = backbone
        self._dtype = backbone.dtype

        # Gaussian head
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)
        self.gaussian_head = GaussianHead(
            cfg=cfg.gaussian_head,
            head_out_dim=self.gaussian_adapter.d_in_no_vel,
            gaussian_adapter=self.gaussian_adapter,
        )

        # Motion encoder for predicting 4D Gaussian params
        if cfg.gaussian_adapter.is_4d:
            self.motion_encoder = MotionEncoder(
                cfg=cfg.motion_encoder,
                embed_dim=cfg.gaussian_head.dim_in,
                num_export_layers=backbone.num_export_layers,
                patch_size=cfg.gaussian_head.patch_size,
                vit_block=backbone.vit_block,
            )

    def _run_backbone(self, scene_input: SceneInput) -> list[LayerTokens]:
        """Run the backbone and return per-layer (patch, camera) token pairs."""
        h, w = scene_input.image.shape[-2:]

        # Optionally condition the backbone camera encoder with known extrinsics and intrinsics.
        cam_token = None
        if scene_input.input_extrinsics is not None and scene_input.input_intrinsics is not None:
            with torch.no_grad(), torch.autocast(device_type=scene_input.image.device.type, enabled=False):
                c2w = normalize_extrinsics(affine_inverse(scene_input.input_extrinsics.clone()))
                cam_token = self.backbone.camera_enc(c2w, scene_input.input_intrinsics, (h, w))

        # Run the backbone, extracting intermediate tokens from specified layers.
        layers_tokens = self.backbone.aggregator(
            scene_input.image,
            cam_token=cam_token,
            timestamps=scene_input.timestamps,
            num_cameras=scene_input.num_cameras,
        )
        return layers_tokens

    def _predict_camera_poses(
        self,
        scene_input: SceneInput,
        layers_tokens: list[LayerTokens],
        average_poses: bool = True,
    ) -> CameraPoseDict:
        """Predict camera poses and return a fully populated CameraPoseDict."""
        _, v, _, h, w = scene_input.image.shape
        image_size = (h, w)

        last_cam_tokens = layers_tokens[-1][1].to(torch.float32)
        raw_pose_enc = self.backbone.camera_head(last_cam_tokens)

        if average_poses:
            # average over temporal views per camera to enforce static cameras.
            views_per_cam = v // scene_input.num_cameras
            raw_pose_enc = (
                rearrange(raw_pose_enc, "b (c t) ... -> b c t ...", c=scene_input.num_cameras)
                .mean(dim=2)
                .repeat_interleave(views_per_cam, dim=1)
            )

        c2w_3x4, intrinsic_px = pose_encoding_to_extri_intri(raw_pose_enc, image_size)
        c2w = F.pad(c2w_3x4, (0, 0, 0, 1))
        c2w[:, :, -1, -1] = 1.0
        w2c = affine_inverse(c2w)
        pose_enc = extri_intri_to_pose_encoding(w2c, intrinsic_px, image_size)

        return {
            "extrinsic_c2w": c2w,
            "extrinsic_w2c": w2c,
            "intrinsic":     intrinsic_px,
            "encodings":     [pose_enc],
        }

    def _predict_depth_maps(
        self,
        scene_input: SceneInput,
        layers_tokens: list[LayerTokens],
    ) -> DepthDict:
        """Predict depth maps from backbone tokens."""
        _, _, _, h, w = scene_input.image.shape
        depth_output = self.backbone.depth_head(feats=layers_tokens, image_size=(h, w))

        depth_map = depth_output["depth"]
        depth_conf = depth_output.get("depth_conf")

        return {
            "depth": depth_map,
            "depth_conf": depth_conf
        }

    def _run_heads(
        self,
        scene_input: SceneInput,
        layers_tokens: list[LayerTokens],
        run_gaussian_head: bool,
        average_poses: bool = True,
    ) -> EncoderOutput:
        """Run camera, depth, motion and Gaussian heads."""

        # Predict camera poses and depth maps
        camera_pose = self._predict_camera_poses(scene_input, layers_tokens, average_poses)
        depth = self._predict_depth_maps(scene_input, layers_tokens)

        if not run_gaussian_head:
            return EncoderOutput(camera_pose=camera_pose, depth=depth)

        # Velocity prediction
        optical_flow = velocity = None
        if self.cfg.gaussian_adapter.is_4d:
            velocity, optical_flow = self.motion_encoder(scene_input, layers_tokens, camera_pose, depth)

        # Gaussian prediction
        gaussians = self.gaussian_head(scene_input, layers_tokens, camera_pose, depth, velocity=velocity)

        return EncoderOutput(
            gaussians=gaussians,
            camera_pose=camera_pose,
            depth=depth,
            optical_flow=optical_flow,
        )

    def forward(
        self,
        images: Float[Tensor, "batch view 3 height width"],
        timestamps: Float[Tensor, "batch view"] | None = None,
        run_gaussian_head: bool = True,
        num_cameras: int = 1,
        input_extrinsics: Float[Tensor, "batch view 4 4"] | None = None,
        input_intrinsics: Float[Tensor, "batch view 3 3"] | None = None,
        average_poses: bool = True,
    ) -> EncoderOutput:
        """Encode multi-view images into 3D/4D Gaussians with predicted camera poses.

        Runs the ViT backbone, predicts camera poses, lifts depth to 3D points,
        and assembles Gaussians with optional 4D temporal dynamics.

        Args:
            images: Input images of shape (batch, views, 3, H, W).
            timestamps: Per-view timestamps for temporal encoding, or None for static scenes.
            run_gaussian_head: If False, skips Gaussian prediction (pose-only forward pass).
            num_cameras: Number of distinct cameras (1 for egocentric, >1 for exocentric).
            input_extrinsics: Optional known w2c extrinsics to condition the backbone camera encoder.
            input_intrinsics: Optional known intrinsics to condition the backbone camera encoder.
            average_poses: If True, averages predicted pose encodings across time per camera to enforce static cameras.

        Returns:
            EncoderOutput with Gaussians, camera poses, depth maps, optical flow, and metadata.
        """
        scene_input = SceneInput(
            image=self.backbone.normalize(images),
            num_cameras=int(num_cameras),
            timestamps=timestamps,
            input_extrinsics=input_extrinsics,
            input_intrinsics=input_intrinsics,
        )

        # Run the backbone with its precision context
        device_type = scene_input.image.device.type
        with torch.autocast(device_type, enabled=True, dtype=self._dtype):
            layers_tokens = self._run_backbone(scene_input)

        # Predict heads output in full precision
        with torch.autocast(device_type, enabled=False):
            encoder_output = self._run_heads(scene_input, layers_tokens, run_gaussian_head, average_poses)
        
        return encoder_output
