# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
# Modified by Botao Ye from https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py.
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from src.dataset.types import BatchedExample
from src.model.decoder.types import DecoderOutput
from src.model.encoder.types import EncoderOutput
from src.model.types import Gaussians
from .loss import Loss


def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input: Tensor, win: Tensor) -> Tensor:
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float,
    win: Tensor,
    size_average: bool = True,
    K: tuple[float, float] | list[float] = (0.01, 0.03),
    retrun_seprate: bool = False,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        data_range (float or int): value range of input images. (usually 1.0 or 255)
        win (torch.Tensor): 1-D gauss kernel
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        retrun_seprate (bool, optional): if True, return brightness, contrast, and structure similarity maps as well

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)

    brightness = contrast = structure = torch.zeros_like(ssim_per_channel)
    if retrun_seprate:
        epsilon = torch.finfo(torch.float32).eps**2
        sigma1_sq = sigma1_sq.clamp(min=epsilon)
        sigma2_sq = sigma2_sq.clamp(min=epsilon)
        sigma12 = torch.sign(sigma12) * torch.minimum(
            torch.sqrt(sigma1_sq * sigma2_sq), torch.abs(sigma12))

        C3 = C2 / 2
        sigma1_sigma2 = torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq)
        brightness_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        contrast_map = (2 * sigma1_sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
        structure_map = (sigma12 + C3) / (sigma1_sigma2 + C3)

        contrast_map = contrast_map.clamp(max=0.98)
        structure_map = structure_map.clamp(max=0.98)

        brightness = brightness_map.flatten(2).mean(-1)
        contrast = contrast_map.flatten(2).mean(-1)
        structure = structure_map.flatten(2).mean(-1)

    return ssim_per_channel, cs, brightness, contrast, structure


def ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Tensor | None = None,
    K: tuple[float, float] | list[float] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
    retrun_seprate: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
        retrun_seprate (bool, optional): if True, return brightness, contrast, and structure similarity maps as well

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs, brightness, contrast, structure \
        = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K, retrun_seprate=retrun_seprate)

    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean(), brightness.mean(), contrast.mean(), structure.mean()
    else:
        return ssim_per_channel.mean(1), brightness.mean(1), contrast.mean(1), structure.mean(1)


@dataclass
class LossSsimCfg:
    weight: float


@dataclass
class LossSsimCfgWrapper:
    ssim: LossSsimCfg


class LossSsim(Loss[LossSsimCfg, LossSsimCfgWrapper]):
    def forward(
        self,
        prediction_context: DecoderOutput,
        prediction_target: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int,
        warmup_steps: int = 0,
        encoder_output_context: EncoderOutput | None = None,
        distill_infos=None,
        visualization_cache: dict | None = None,
    ) -> Float[Tensor, ""]:
        pred_img_flat = rearrange(prediction_target.color, "b v c h w -> (b v) c h w")
        gt_img_flat   = rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w")

        ssim_value = ssim(
            pred_img_flat,
            gt_img_flat,
            data_range=1.0,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            K=(0.01, 0.03),
            nonnegative_ssim=False,
            retrun_seprate=False,
        )[0]

        return self.cfg.weight * (1.0 - ssim_value)
