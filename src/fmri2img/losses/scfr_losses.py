"""SCFR-specific losses."""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossCovarianceOrthogonalityLoss(nn.Module):
    """Penalize cross-covariance between visual and subject latents.

    The penalty is computed on batchwise centered representations and can
    optionally standardize each feature dimension before forming the
    cross-covariance matrix.
    """

    def __init__(
        self,
        *,
        standardize: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.standardize = bool(standardize)
        self.eps = float(eps)

    def forward(self, z_vis: torch.Tensor, z_subj: torch.Tensor) -> torch.Tensor:
        if z_vis.ndim != 2 or z_subj.ndim != 2:
            raise ValueError(
                "CrossCovarianceOrthogonalityLoss expects 2D tensors, got "
                f"{tuple(z_vis.shape)} and {tuple(z_subj.shape)}"
            )
        if z_vis.shape[0] != z_subj.shape[0]:
            raise ValueError(
                "CrossCovarianceOrthogonalityLoss batch mismatch: "
                f"{z_vis.shape[0]} vs {z_subj.shape[0]}"
            )
        if z_vis.shape[0] < 2:
            return z_vis.new_tensor(0.0)

        z_vis_centered = z_vis - z_vis.mean(dim=0, keepdim=True)
        z_subj_centered = z_subj - z_subj.mean(dim=0, keepdim=True)

        if self.standardize:
            z_vis_centered = z_vis_centered / (
                z_vis_centered.std(dim=0, keepdim=True, unbiased=False) + self.eps
            )
            z_subj_centered = z_subj_centered / (
                z_subj_centered.std(dim=0, keepdim=True, unbiased=False) + self.eps
            )

        cov = (z_vis_centered.transpose(0, 1) @ z_subj_centered) / max(z_vis.shape[0] - 1, 1)
        return cov.pow(2).mean()
