import torch


def create_uv_grid(
    width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None
) -> torch.Tensor:
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    diag_factor = (aspect_ratio ** 2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    x_coords = torch.linspace(-span_x * (width - 1) / width,  span_x * (width - 1) / width,  steps=width,  dtype=dtype, device=device)
    y_coords = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, steps=height, dtype=dtype, device=device)

    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    return torch.stack((uu, vv), dim=-1)


def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)

    emb_x = _make_sincos_embed(embed_dim // 2, pos_flat[:, 0], omega_0)
    emb_y = _make_sincos_embed(embed_dim // 2, pos_flat[:, 1], omega_0)

    return torch.cat([emb_x, emb_y], dim=-1).view(H, W, embed_dim)


def _make_sincos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double, device=pos.device)
    omega = 1.0 / omega_0 ** (omega / (embed_dim / 2.0))
    out = torch.einsum("m,d->md", pos.reshape(-1).double(), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()
