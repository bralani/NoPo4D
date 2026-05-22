from typing import Sequence

import torch
from torch import Tensor


def scaled_sigmoid(x: Tensor, bias: float, lo: float, hi: float) -> Tensor:
    """Sigmoid-map x into [lo, hi] with a learnable bias shift."""
    return (x + bias).sigmoid() * (hi - lo) + lo


def pad_tensor_list(
    tensors: list[Tensor],
    target_shape: tuple[int, ...],
    value: float = 0.0,
) -> Tensor:
    """Pad a list of tensors along dim 0 to target_shape[0], then stack into a batch."""
    if not tensors:
        raise ValueError("The input list of tensors cannot be empty.")

    batch_size = len(tensors)
    target_n = target_shape[0]
    trailing_dims = tensors[0].shape[1:]

    out = tensors[0].new_full((batch_size, target_n, *trailing_dims), value)
    for i, t in enumerate(tensors):
        n = t.shape[0]
        out[i, :n] = t

    return out
