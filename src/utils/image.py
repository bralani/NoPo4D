from typing import Union

import numpy as np
import torch
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from torch import Tensor

from src.training.logger.visualization.color_map import apply_color_map_to_image

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]


def inverse_normalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return tensor.mul(std).add(mean)


def vis_depth_map(result, near=None, far=None):
    if near is None and far is None:
        far = result.view(-1)[:16_000_000].quantile(0.99).log()
        try:
            near = result[result > 0][:16_000_000].quantile(0.01).log()
        except Exception:
            print("No valid depth values found.")
            near = torch.zeros_like(far)
    else:
        near = near.log()
        far = far.log()

    result = result.log()
    result = 1 - (result - near) / (far - near)
    return apply_color_map_to_image(result, "turbo")


def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)
    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()
