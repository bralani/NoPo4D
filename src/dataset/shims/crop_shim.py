import random
import torch
from jaxtyping import Float
from torch import Tensor
import torchvision.transforms.functional as F

from ..types import AnyViews, UnbatchedExample


def rescale_depth(
    depth: Float[Tensor, "1 h w"],
    shape: tuple[int, int],
) -> Float[Tensor, "1 h_out w_out"]:
    h, w = shape
    return F.resize(depth.unsqueeze(0), (h, w), interpolation=F.InterpolationMode.NEAREST).squeeze(0)
    
def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    depths: Float[Tensor, "*#batch 1 h w"] | None = None,
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
    Float[Tensor, "*#batch 1 h_out w_out"] | None,  # updated depths
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    if depths is not None:
        depths = depths[..., row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    
    return images, intrinsics, depths


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    intr_aug: bool = False,
    scale_range: tuple[float, float] = (0.77, 1.0),
    depths: Float[Tensor, "*#batch 1 h w"] | None = None,
    scale: float | None = None,
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
    Float[Tensor, "*#batch 1 h_out w_out"] | None,  # updated depths
]:
    if type(images) == list:
        images_new = []
        intrinsics_new = []
        for i in range(len(images)):
            image = images[i]
            intrinsic = intrinsics[i]
            
            *_, h_in, w_in = image.shape
            h_out, w_out = shape

            scale_factor = max(h_out / h_in, w_out / w_in)
            h_scaled = round(h_in * scale_factor)
            w_scaled = round(w_in * scale_factor)
            image = F.resize(image, (h_scaled, w_scaled))
            image = F.center_crop(image, (h_out, w_out))
            images_new.append(image)
            
            intrinsic_new = intrinsic.clone()
            intrinsic_new[..., 0, 0] *= w_scaled / w_in  # fx
            intrinsic_new[..., 1, 1] *= h_scaled / h_in  # fy
            intrinsics_new.append(intrinsic_new)
        
        if depths is not None:
            depths_new = []
            for i in range(len(depths)):
                depth = depths[i]
                depth = rescale_depth(depth, (h_out, w_out))
                depth = F.center_crop(depth, (h_out, w_out))
                depths_new.append(depth)
            return torch.stack(images_new), torch.stack(intrinsics_new), torch.stack(depths_new)
        else:
            return torch.stack(images_new), torch.stack(intrinsics_new), None

    else:
        # we only support intr_aug for clean datasets
        *_, h_in, w_in = images.shape
        h_out, w_out = shape
        # assert h_out <= h_in and w_out <= w_in # to avoid the case that the image is too small, like co3d
        
        if intr_aug:
            if scale is None:
                scale = random.uniform(*scale_range)
            h_scale = round(h_out * scale)
            w_scale = round(w_out * scale)
        else:
            h_scale = h_out
            w_scale = w_out

        scale_factor = max(h_scale / h_in, w_scale / w_in)
        h_scaled = round(h_in * scale_factor)
        w_scaled = round(w_in * scale_factor)
        assert h_scaled == h_scale or w_scaled == w_scale

        # Reshape the images to the correct size. Assume we don't have to worry about
        # changing the intrinsics based on how the images are rounded.
        *batch, c, h, w = images.shape
        images = images.reshape(-1, c, h, w)
        images = F.resize(images, (h_scaled, w_scaled), interpolation=F.InterpolationMode.BILINEAR, antialias=False)
        images = images.reshape(*batch, c, h_scaled, w_scaled)

        if depths is not None:
            if type(depths) == list:
                depths_new = []
                for i in range(len(depths)):
                    depth = depths[i]
                    depth = rescale_depth(depth, (h_scaled, w_scaled)) 
                    depths_new.append(depth)
                depths = torch.stack(depths_new)
            else:
                depths = depths.reshape(-1, 1, h, w)
                depths = F.resize(depths, (h_scaled, w_scaled), interpolation=F.InterpolationMode.NEAREST)
                depths = depths.reshape(*batch, h_scaled, w_scaled)
            
            images, intrinsics, depths = center_crop(images, intrinsics, (h_scale, w_scale), depths)

            if intr_aug:
                images = F.resize(images, size=(h_out, w_out), interpolation=F.InterpolationMode.BILINEAR)
                depths = F.resize(depths, size=(h_out, w_out), interpolation=F.InterpolationMode.NEAREST)
                
            return images, intrinsics, depths
        else:
            images, intrinsics, _ = center_crop(images, intrinsics, (h_scale, w_scale))

            if intr_aug:
                images = F.resize(images, size=(h_out, w_out))

            return images, intrinsics, None


def apply_crop_shim_to_views(
    views: AnyViews,
    shape: tuple[int, int],
    intr_aug: bool = False,
    scale: float | None = None,
) -> AnyViews:
    h_in, w_in = views["image"].shape[-2], views["image"].shape[-1]

    if "depth" in views.keys():
        images, intrinsics, depths = rescale_and_crop(
            views["image"], views["intrinsics"], shape, depths=views["depth"], intr_aug=intr_aug, scale=scale
        )
        result = {
            **views,
            "image": images,
            "intrinsics": intrinsics,
            "depth": depths,
        }
    else:
        images, intrinsics, _ = rescale_and_crop(
            views["image"], views["intrinsics"], shape, intr_aug=intr_aug, scale=scale
        )
        result = {
            **views,
            "image": images,
            "intrinsics": intrinsics,
        }

    if "track" in views:
        h_out, w_out = images.shape[-2], images.shape[-1]
        result["track"] = views["track"].clone()
        result["track"][..., 0] *= w_out / w_in
        result["track"][..., 1] *= h_out / h_in

    return result
        

def apply_crop_shim(
    example: UnbatchedExample,
    shape: tuple[int, int],
    intr_aug: bool = False,
    scale_range: tuple[float, float] = (0.77, 1.0),
) -> UnbatchedExample:
    """Crop images in the example."""
    scale = random.uniform(*scale_range) if intr_aug else None
    context_cropped = apply_crop_shim_to_views(example["context"], shape, intr_aug, scale)
    target_cropped = apply_crop_shim_to_views(example["target"], shape, intr_aug, scale)
    
    return UnbatchedExample(
        context=context_cropped,
        target=target_cropped,
        scene=example["scene"],
        num_cameras=example["num_cameras"]
    )
