import gc

import torch
import wandb
from einops import pack
from lightning.pytorch.utilities import rank_zero_only

from src.utils.image import vis_depth_map
from .camera_trajectory.interpolation import interpolate_extrinsics, interpolate_intrinsics
from .layout import vcat


@rank_zero_only
def render_video_interpolation(wrapper, batch, num_frames: int = 30, stage=""):
    """Render an interpolation trajectory video and delegate to wrapper."""
    _, v, _, _ = batch["context"]["extrinsics"].shape
    num_cameras = int(batch["num_cameras"][0]) if "num_cameras" in batch else 1

    def trajectory_fn(
        t: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        timestamps_batch: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, v, _, _ = extrinsics.shape

        if num_cameras == 1:
            extrinsics = interpolate_extrinsics(
                extrinsics[0, 0],
                extrinsics[0, -1],
                t,
            )
            intrinsics = interpolate_intrinsics(
                intrinsics[0, 0],
                intrinsics[0, -1],
                t,
            )
            return extrinsics[None], intrinsics[None]
        
        views_per_camera = v // num_cameras
        first_frame_indices = torch.arange(0, v, views_per_camera, device=extrinsics.device)

        camera_extrinsics = extrinsics[0, first_frame_indices]
        camera_intrinsics = intrinsics[0, first_frame_indices]
        
        num_segments = num_cameras
        t_scaled = t * num_segments
        segment_idx = torch.floor(t_scaled).long().clamp(0, num_segments - 1)
        t_local = (t_scaled - segment_idx).clamp(0, 1)
        
        result_extrinsics = []
        result_intrinsics = []
        
        for i in range(len(t)):
            start_cam = segment_idx[i]
            end_cam = (start_cam + 1) % num_cameras
            
            ext_interp = interpolate_extrinsics(
                camera_extrinsics[start_cam],
                camera_extrinsics[end_cam],
                t_local[i].unsqueeze(0)
            )[0]
            
            int_interp = interpolate_intrinsics(
                camera_intrinsics[start_cam],
                camera_intrinsics[end_cam],
                t_local[i].unsqueeze(0)
            )[0]
            
            result_extrinsics.append(ext_interp)
            result_intrinsics.append(int_interp)
        
        return torch.stack(result_extrinsics)[None], torch.stack(result_intrinsics)[None]

    return render_video_generic(
        wrapper, batch, trajectory_fn, stage+"rgb", num_frames=num_frames
    )


@rank_zero_only
def render_video_generic(
    wrapper,
    batch,
    trajectory_fn,
    name: str,
    num_frames: int = 30,
    smooth: bool = True,
    loop_reverse: bool = True,
) -> None:
    """Generic renderer used by the specific trajectory helpers."""
    gc.collect()
    torch.cuda.empty_cache()

    num_cameras = int(batch["num_cameras"][0])
    timestamps = None
    if "timestamp" in batch["context"]:
        timestamps = batch["context"]["timestamp"]
    with torch.no_grad():
        encoder_output = wrapper.model.encoder(
            batch["context"]["image"],
            timestamps=timestamps,
            num_cameras=num_cameras,
        )
    gaussians = encoder_output.gaussians
    pred_extrinsics = encoder_output.camera_pose["extrinsic_c2w"]
    pred_intrinsics = encoder_output.camera_pose["intrinsic"]

    t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=wrapper.device)
    if smooth:
        t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

    extrinsics, intrinsics = trajectory_fn(t, pred_extrinsics, pred_intrinsics, timestamps_batch=timestamps)

    _, _, _, h, w = batch["context"]["image"].shape

    with torch.no_grad():
        output = wrapper.model.decoder.forward(
            gaussians, extrinsics, intrinsics, (h, w), ts=t.unsqueeze(0) if timestamps is not None else None
        )
    images = [
        vcat(rgb, depth)
        for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
    ]

    video = torch.stack(images)
    video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
    if loop_reverse:
        video = pack([video, video[::-1][1:-1]], "* c h w")[0]
    visualizations = {f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")}

    try:
        wandb.log(visualizations)
    except Exception:
        pass


