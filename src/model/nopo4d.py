"""NoPo4D: pose-free feed-forward 4D Gaussian Splatting from multi-view video."""

import huggingface_hub
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from typing import Any

from src.model.decoder.config import Decoder4DGSCfg
from src.model.decoder.decoder_4dgs import Decoder4DGS
from src.model.decoder.types import DecoderOutput
from src.model.encoder.backbone import get_backbone
from src.model.encoder.encoder import EncoderNoPo4D, EncoderNoPo4DCfg
from src.model.encoder.types import EncoderOutput
from src.model.types import Gaussians


class NoPo4D(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """No Pose, No Problem in 4D: Feed-Forward Dynamic Gaussians
    from Unposed Multi-View Videos.

    Encodes multi-view videos into 4D Gaussians via EncoderNoPo4D,
    then renders target views with the CUDA splatting decoder.
    """

    def __init__(self, encoder_cfg: EncoderNoPo4DCfg | dict[str, Any], decoder_cfg: Decoder4DGSCfg | dict[str, Any]) -> None:
        super().__init__()
        if isinstance(encoder_cfg, dict):
            encoder_cfg = EncoderNoPo4DCfg.from_dict(encoder_cfg)
        if isinstance(decoder_cfg, dict):
            decoder_cfg = Decoder4DGSCfg(**decoder_cfg)
        backbone = get_backbone(encoder_cfg.backbone)
        self.encoder = EncoderNoPo4D(encoder_cfg, backbone)
        self.decoder = Decoder4DGS(decoder_cfg)

    def forward(
        self,
        images: Float[Tensor, "batch cam_time 3 height width"],
        timestamps: Float[Tensor, "batch cam_time"] | None = None,
        num_cameras: int = 1,
        input_extrinsics: Float[Tensor, "batch cam_time 4 4"] | None = None,
        input_intrinsics: Float[Tensor, "batch cam_time 3 3"] | None = None,
        **kwargs: Any,
    ) -> EncoderOutput:
        """Encode multi-view video into Gaussians and predicted camera poses.

        Args:
            images:             Input video tensors, shape (B, V, C, H, W), where V = num_cameras
                                * num_frames. Frames are laid out in camera-major order: all
                                time steps of camera 0 come first, followed by all time steps of
                                camera 1, and so on. For example, with 2 cameras and 3 frames each,
                                the view dimension is ordered as [cam0_t0, cam0_t1, cam0_t2,
                                cam1_t0, cam1_t1, cam1_t2]. For a single camera set num_cameras=1
                                and V equals the number of frames.
            timestamps:         Normalised timestamp for each view, shape (B, V), values in [0, 1].
                                Must follow the same camera-major layout as images. Pass
                                None for static scenes.
            num_cameras:        Number of distinct cameras. Used to group views per camera when
                                averaging predicted poses across time.
            input_extrinsics:   Optional known w2c extrinsics to condition the backbone camera encoder.
            input_intrinsics:   Optional known intrinsics to condition the backbone camera encoder.

        Returns:
            EncoderOutput
        """
        return self.encoder(
            images,
            timestamps=timestamps,
            num_cameras=num_cameras,
            input_extrinsics=input_extrinsics,
            input_intrinsics=input_intrinsics,
            **kwargs,
        )

    def render(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch num_views 4 4"],
        intrinsics: Float[Tensor, "batch num_views 3 3"],
        image_shape: tuple[int, int],
        timestamps: Float[Tensor, "batch num_views"] | None = None,
    ) -> DecoderOutput:
        """Render target views from Gaussians and camera parameters.

        Args:
            gaussians:    Gaussian primitives produced by the encoder.
            extrinsics:   w2c matrices, shape (B, V, 4, 4).
            intrinsics:   Normalised camera intrinsics, shape (B, V, 3, 3).
            image_shape:  Output resolution as (H, W).
            timestamps:   Per-view timestamps for 4D temporal rendering, shape (B, V), or None.

        Returns:
            DecoderOutput
        """
        return self.decoder(
            gaussians=gaussians,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_shape=image_shape,
            ts=timestamps,
        )
