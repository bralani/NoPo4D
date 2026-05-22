import argparse, sys, os
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.utils import flow_to_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.nopo4d import NoPo4D

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_images(image_dir: Path, device: torch.device) -> torch.Tensor:
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"No images found in {image_dir}")
    imgs = [transforms.ToTensor()(Image.open(p).convert("RGB")) for p in paths]
    print(f"Loaded {len(imgs)} image(s) from {image_dir}")
    return torch.stack(imgs).unsqueeze(0).to(device)


def save_frames(frames: torch.Tensor, out_dir: Path, prefix: str, to_pil) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        to_pil(frame.float().cpu()).save(out_dir / f"{prefix}_{i:04d}.png")
    print(f"Saved {len(frames)} frame(s) to {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--num_cameras", type=int, required=True)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--render_timestamps", type=int, default=None, help="Number of timestamps to render per camera (defaults to input frames)")
    args = parser.parse_args()

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoPo4D.from_pretrained("bralani01/nopo4d").to(device).eval()

    # load input images
    images = load_images(Path(args.image_dir), device)
    _, V, _, H, W = images.shape

    # build synchronized timestamps in camera-major order repeated per camera
    num_frames = V // args.num_cameras
    timestamps = torch.linspace(0, 1, num_frames, device=device).repeat(args.num_cameras).unsqueeze(0)

    # run encoder and render images
    with torch.no_grad():
        encoder_output = model.forward(images, timestamps=timestamps, num_cameras=args.num_cameras)
        assert encoder_output.gaussians is not None, "Encoder did not return Gaussians"

        # build render timestamps
        render_frames = args.render_timestamps or num_frames
        render_timestamps = torch.linspace(0, 1, render_frames, device=device).repeat(args.num_cameras).unsqueeze(0)

        # assuming static cameras: one extrinsic/intrinsic per camera, repeated across render_frames
        extrinsics = encoder_output.camera_pose["extrinsic_c2w"][:, ::num_frames]   # [B, num_cameras, 4, 4]
        intrinsics = encoder_output.camera_pose["intrinsic"][:, ::num_frames]        # [B, num_cameras, 3, 3]
        extrinsics = extrinsics.repeat_interleave(render_frames, dim=1)              # [B, num_cameras * render_frames, 4, 4]
        intrinsics = intrinsics.repeat_interleave(render_frames, dim=1)              # [B, num_cameras * render_frames, 3, 3]

        render_output = model.render(
            gaussians=encoder_output.gaussians,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_shape=(H, W),
            timestamps=render_timestamps,
        )

    # save rendered views and optical flow images
    output_dir = Path(args.output_dir)
    to_pil = transforms.ToPILImage()

    save_frames(render_output.color[0].clamp(0, 1), output_dir / "images", "view", to_pil)

    flow = encoder_output.optical_flow
    if flow is not None:
        for key, folder in (("motion_flow_fwd", "flow_fwd"), ("motion_flow_bwd", "flow_bwd")):
            val = flow.get(key)
            if val is not None:
                save_frames(val[0], output_dir / "optical_flow" / folder, "frame", lambda f: to_pil(flow_to_image(f)))


if __name__ == "__main__":
    main()
