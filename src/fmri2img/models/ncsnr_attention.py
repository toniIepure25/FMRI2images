"""
NCSNR Voxel Attention Layer (V11).

Uses neuroscience-derived per-voxel noise-ceiling signal-to-noise ratio
(NCSNR) as a learnable architectural inductive bias.  High-NCSNR voxels
start with higher weight, but the parameters remain trainable so the
model can refine the weighting during training.

Prior work either hard-masks voxels by reliability thresholds or
ignores voxel reliability entirely.  This module provides a smooth,
differentiable middle ground.

Usage:
    ncsnr = load_ncsnr("subj01")          # (n_voxels,)
    attn = NCSnrAttention(n_voxels, ncsnr)
    fmri_weighted = attn(fmri)            # (B, n_voxels)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class NCSnrAttention(nn.Module):
    """Learnable per-voxel scaling initialised from NCSNR.

    Each voxel gets a scalar weight ``w_v`` that is applied element-wise
    to the input fMRI vector.  Weights are parameterised through softplus
    to keep them non-negative:

        output_v = input_v * softplus(raw_weight_v)

    Initialisation: ``raw_weight_v = inverse_softplus(sqrt(ncsnr_v / max))``
    so that high-NCSNR voxels start with values close to 1 and low-NCSNR
    voxels start near 0.

    Args:
        n_voxels: Number of input voxels.
        ncsnr: Per-voxel NCSNR array (n_voxels,).  If ``None``, falls back
               to uniform initialisation (all weights = 1).
    """

    def __init__(self, n_voxels: int, ncsnr: np.ndarray | None = None):
        super().__init__()
        if ncsnr is not None:
            ncsnr_t = torch.from_numpy(np.asarray(ncsnr, dtype=np.float32))
            target = torch.sqrt(ncsnr_t / (ncsnr_t.max() + 1e-8))
            target = target.clamp(min=0.01)
            raw = self._inverse_softplus(target)
        else:
            raw = self._inverse_softplus(torch.ones(n_voxels))

        self.raw_weights = nn.Parameter(raw)
        logger.info(
            "NCSnrAttention: %d voxels, init_range=[%.3f, %.3f]",
            n_voxels,
            F.softplus(raw).min().item(),
            F.softplus(raw).max().item(),
        )

    @staticmethod
    def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
        """Numerically stable inverse of softplus: log(exp(x) - 1)."""
        return torch.where(
            x > 20.0, x, torch.log(torch.expm1(x.clamp(min=1e-6)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_voxels) raw fMRI input.

        Returns:
            (B, n_voxels) reliability-weighted fMRI.
        """
        return x * F.softplus(self.raw_weights)
