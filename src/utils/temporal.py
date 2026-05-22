import torch
from jaxtyping import Float
from torch import Tensor

_SINUSOIDAL_BASE = 10000.0


def sinusoidal_time_encoding(
    timestamps: Float[Tensor, "batch seq"],
    embed_dim: int,
) -> Float[Tensor, "batch_seq 1 embed_dim"]:
    """Encode continuous timestamps into sinusoidal positional embeddings.

    Args:
        timestamps: Per-frame timestamps of shape [B, S], values in [0, 1].
        embed_dim: Dimensionality of the output embedding. Must be even.

    Returns:
        Encoded time tokens of shape [B*S, 1, embed_dim], ready to be
        inserted into a ViT token sequence.
    """
    B, S = timestamps.shape
    assert embed_dim % 2 == 0, f"embed_dim must be even, got {embed_dim}"

    # Frequency bands: shape [1, 1, embed_dim // 2]
    num_bands = embed_dim // 2
    freq_bands = 1.0 / (
        _SINUSOIDAL_BASE ** (torch.arange(num_bands, dtype=torch.float32, device=timestamps.device) / num_bands)
    )
    freq_bands = freq_bands.to(dtype=timestamps.dtype).unsqueeze(0).unsqueeze(0)

    # Outer product: [B, S, num_bands]
    angles = timestamps.unsqueeze(-1) * freq_bands

    # Concatenate sin and cos halves: [B, S, embed_dim]
    encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    # Flatten batch and sequence for token insertion: [B*S, 1, embed_dim]
    return encoding.view(B * S, 1, embed_dim)


def dt_to_cov_t(dt: Tensor, marginal_thresh: float = 0.05) -> Tensor:
    """Convert a temporal extent (dt) into a temporal covariance value."""
    dt_t = torch.as_tensor(dt)
    thresh_t = torch.as_tensor(marginal_thresh, device=dt_t.device, dtype=dt_t.dtype)
    return (dt_t ** 2) / (torch.log(thresh_t) / -0.5)



def compute_marginal_t(t: Tensor, mu_t: Tensor, cov_t: Tensor) -> Tensor:
    """Compute the elementwise Gaussian marginal for the temporal variable."""
    return torch.exp(-0.5 * (t - mu_t) ** 2 / cov_t)
