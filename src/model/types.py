"""Gaussians dataclass for 3D/4D Gaussian primitives."""

from dataclasses import dataclass
from typing import TypedDict

import torch
from jaxtyping import Float
from torch import Tensor

from src.utils.geometry import angle_axis_to_quaternion, qmul
from src.utils.temporal import compute_marginal_t


class TransformedGaussians(TypedDict):
    means:     Float[Tensor, "ts gaussian 3"]
    rotations: Float[Tensor, "ts gaussian 4"]
    opacities: Float[Tensor, "ts gaussian"]


@dataclass
class Gaussians:
    # required params for both 3D and 4D Gaussians
    means: Float[Tensor, "batch gaussian 3"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    scales: Float[Tensor, "batch gaussian 3"]
    rotations: Float[Tensor, "batch gaussian 4"]

    # params for 4D Gaussians
    t: Float[Tensor, "batch gaussian"] | None = None  # temporal mean/center of 4D Gaussian
    cov_t: Float[Tensor, "batch gaussian"] | None = None  # temporal covariance
    ms3_fwd: Float[Tensor, "batch gaussian 3"] | None = None  # forward 3D motion/velocity vector
    ms3_bwd: Float[Tensor, "batch gaussian 3"] | None = None  # backward 3D motion/velocity vector
    omega_fwd: Float[Tensor, "batch gaussian 3"] | None = None  # forward angular velocity vector
    omega_bwd: Float[Tensor, "batch gaussian 3"] | None = None  # backward angular velocity vector
    opacity_sh: Float[Tensor, "batch gaussian d_sh"] | None = None  # SH coefficients for opacity

    def is_4d(self) -> bool:
        """Return True if Gaussians has 4D (temporal) attributes."""
        return all(x is not None for x in (
            self.t, self.cov_t, self.ms3_fwd, self.ms3_bwd, self.omega_fwd, self.omega_bwd,
        ))

    def transform_to_target_time(
        self,
        ts: Float[Tensor, "... ts"],
        batch_idx: int,
    ) -> TransformedGaussians:
        """Transform Gaussians to target time(s) using 4D motion model.

        Args:
            ts: Target timestamp(s). Shape [T] or [1, T].
            batch_idx: Batch index to select before transformation.

        Returns:
            TransformedGaussians at the target time(s).
        """
        if not self.is_4d():
            raise ValueError("Cannot transform non-4D Gaussians to target time")

        # extract the requested batch element
        s         = slice(batch_idx, batch_idx + 1)
        means     = self.means[s].unsqueeze(1)
        rotations = self.rotations[s].unsqueeze(1)
        opacities = self.opacities[s].unsqueeze(1)
        ms3_fwd,   ms3_bwd   = self.ms3_fwd[s].unsqueeze(1),   self.ms3_bwd[s].unsqueeze(1)
        omega_fwd, omega_bwd = self.omega_fwd[s].unsqueeze(1), self.omega_bwd[s].unsqueeze(1)
        t,         cov_t     = self.t[s].unsqueeze(1),         self.cov_t[s].unsqueeze(1)

        # compute signed time delta
        ts      = ts.to(means.device).reshape(1, -1, 1)
        delta_t = (ts - t).unsqueeze(-1)

        # determine forward vs backward motion based on sign of time delta
        fwd     = delta_t >= 0
        ms3     = torch.where(fwd, ms3_fwd, ms3_bwd)
        omega   = torch.where(fwd, omega_fwd, omega_bwd)

        # apply motion model to position, rotation, and opacity
        new_means     = means + ms3 * torch.abs(delta_t)
        new_rotations = qmul(rotations, angle_axis_to_quaternion(omega * torch.abs(delta_t)))
        new_opacities = opacities * compute_marginal_t(ts, t, cov_t)

        return TransformedGaussians(
            means=new_means.squeeze(0),
            rotations=new_rotations.squeeze(0),
            opacities=new_opacities.squeeze(0),
        )
