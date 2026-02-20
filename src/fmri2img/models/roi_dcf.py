"""
Per-ROI Directional Experts with Spherical Consensus Fusion (ROI-DCF)
=====================================================================

Instead of collapsing all ROI information into a single [CLS] token and
producing one (mu, kappa) pair, this module attaches per-ROI vMF prediction
heads to each post-Transformer ROI token.  The per-ROI predictions are
fused via the **spherical weighted mean** (Mardia & Jupp, 2000), yielding:

    mu_fused         — consensus direction on S^{d-1}
    kappa_consensus  — resultant length (naturally lower when ROIs disagree)
    delta            — pairwise directional disagreement score in [0, 1]

The attention weights alpha_r from the final Transformer layer serve as
mixing coefficients, so no extra gating network is needed.

Key theoretical properties:
    * When all ROIs agree (mu_r ≈ mu_s), kappa_consensus ≈ sum(alpha_r * kappa_r)
    * When ROIs disagree, the resultant vector is shorter → lower kappa_consensus
    * delta = 0 when all mu_r are aligned, delta → 1 when they oppose

References:
    - Mardia, K.V. & Jupp, P.E. (2000) Directional Statistics
    - Banerjee et al. (2005)  Clustering on the Unit Hypersphere
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class PerROIVMFHeads(nn.Module):
    """
    Shared or independent vMF prediction heads applied to each ROI token.

    Each head maps a d_model token to (mu_r, kappa_r) on S^{output_dim - 1}.

    Args:
        d_model:     Input token dimension from the Transformer encoder.
        output_dim:  CLIP embedding dimension (e.g. 768).
        n_rois:      Number of ROI tokens (excluding [CLS]).
        shared:      If True, all ROIs share the same projection weights.
        hidden_dim:  Optional hidden layer between token and prediction heads.
        kappa_min:   Lower bound for bounded-sigmoid kappa.
        kappa_max:   Upper bound for bounded-sigmoid kappa.
        dropout:     Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        output_dim: int,
        n_rois: int,
        shared: bool = True,
        hidden_dim: Optional[int] = None,
        kappa_min: float = 1e-3,
        kappa_max: float = 500.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.n_rois = n_rois
        self.shared = shared
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max

        def _make_head() -> nn.Module:
            if hidden_dim is not None:
                backbone = nn.Sequential(
                    nn.Linear(d_model, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                in_dim = hidden_dim
            else:
                backbone = nn.Identity()
                in_dim = d_model
            return nn.ModuleDict({
                "backbone": backbone,
                "mu": nn.Linear(in_dim, output_dim),
                "kappa": nn.Linear(in_dim, 1),
            })

        if shared:
            self.head = _make_head()
        else:
            self.heads = nn.ModuleList([_make_head() for _ in range(n_rois)])

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "PerROIVMFHeads: d_model=%d -> output_dim=%d, n_rois=%d, "
            "shared=%s, hidden=%s, kappa=[%s, %s], params=%s",
            d_model, output_dim, n_rois, shared, hidden_dim,
            kappa_min, kappa_max, f"{n_params:,}",
        )

    def _forward_single(
        self, head: nn.ModuleDict, token: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply one head to a (B, d_model) token."""
        features = head["backbone"](token)
        mu = F.normalize(head["mu"](features), p=2, dim=-1)
        raw_kappa = head["kappa"](features)
        kappa = self.kappa_min + (self.kappa_max - self.kappa_min) * torch.sigmoid(raw_kappa)
        return mu, kappa

    def forward(
        self, roi_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            roi_tokens: (B, n_rois, d_model) post-Transformer ROI token embeddings.

        Returns:
            mus:    (B, n_rois, output_dim) per-ROI mean directions on S^{d-1}.
            kappas: (B, n_rois, 1) per-ROI concentrations.
        """
        B, R, _ = roi_tokens.shape
        mus = []
        kappas = []
        for r in range(R):
            head = self.head if self.shared else self.heads[r]
            mu_r, kappa_r = self._forward_single(head, roi_tokens[:, r])
            mus.append(mu_r)
            kappas.append(kappa_r)
        mus = torch.stack(mus, dim=1)       # (B, R, D)
        kappas = torch.stack(kappas, dim=1)  # (B, R, 1)
        return mus, kappas


class SphericalConsensusFusion(nn.Module):
    """
    Fuse per-ROI vMF predictions via the spherical weighted mean.

    Given per-ROI directions mu_r, concentrations kappa_r, and attention
    weights alpha_r, the fused direction and consensus concentration are:

        R_vec           = sum_r  alpha_r * kappa_r * mu_r
        mu_fused        = normalize(R_vec)
        kappa_consensus = ||R_vec||

    The disagreement score measures pairwise directional consistency:

        delta = 1 - sum_{r,s} alpha_r * alpha_s * cos(mu_r, mu_s)
    """

    def forward(
        self,
        mus: torch.Tensor,
        kappas: torch.Tensor,
        alphas: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mus:    (B, R, D) per-ROI unit-norm mean directions.
            kappas: (B, R, 1) per-ROI concentrations.
            alphas: (B, R) attention-derived mixing weights (should sum to 1).

        Returns:
            mu_fused:        (B, D)  consensus direction.
            kappa_consensus: (B, 1)  resultant length (consensus concentration).
            delta:           (B, 1)  disagreement score in [0, 1].
        """
        # Weighted resultant vector: (B, R, 1) * (B, R, D) -> (B, R, D) -> sum -> (B, D)
        weights = alphas.unsqueeze(-1) * kappas  # (B, R, 1)
        R_vec = (weights * mus).sum(dim=1)       # (B, D)

        kappa_consensus = R_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, 1)
        mu_fused = F.normalize(R_vec, p=2, dim=-1)                          # (B, D)

        # Disagreement: delta = 1 - sum_{r,s} alpha_r * alpha_s * cos(mu_r, mu_s)
        # cos(mu_r, mu_s) = mus @ mus^T  -> (B, R, R)
        cos_matrix = torch.bmm(mus, mus.transpose(1, 2))            # (B, R, R)
        # Outer product of alphas: (B, R, 1) @ (B, 1, R) -> (B, R, R)
        alpha_outer = alphas.unsqueeze(-1) * alphas.unsqueeze(-2)    # (B, R, R)
        weighted_agreement = (alpha_outer * cos_matrix).sum(dim=(1, 2))  # (B,)
        delta = (1.0 - weighted_agreement).clamp(min=0.0, max=1.0).unsqueeze(-1)  # (B, 1)

        return mu_fused, kappa_consensus, delta


class ROIDCFDecoder(nn.Module):
    """
    Complete ROI-DCF decoder: per-ROI vMF heads + spherical consensus fusion.

    Wraps PerROIVMFHeads and SphericalConsensusFusion into a single module
    suitable for integration with UnifiedModel.

    Args:
        d_model:     Transformer token dimension.
        output_dim:  CLIP embedding dimension.
        n_rois:      Number of ROI tokens.
        shared:      Whether ROI heads share weights.
        hidden_dim:  Optional hidden dim for per-ROI heads.
        kappa_min:   Lower kappa bound.
        kappa_max:   Upper kappa bound.
        dropout:     Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        output_dim: int,
        n_rois: int,
        shared: bool = True,
        hidden_dim: Optional[int] = None,
        kappa_min: float = 1e-3,
        kappa_max: float = 500.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = d_model
        self.output_dim = output_dim
        self.n_rois = n_rois

        self.roi_heads = PerROIVMFHeads(
            d_model=d_model,
            output_dim=output_dim,
            n_rois=n_rois,
            shared=shared,
            hidden_dim=hidden_dim,
            kappa_min=kappa_min,
            kappa_max=kappa_max,
            dropout=dropout,
        )
        self.fusion = SphericalConsensusFusion()

        logger.info(
            "ROIDCFDecoder: %d ROIs, d_model=%d, output_dim=%d, shared=%s",
            n_rois, d_model, output_dim, shared,
        )

    def forward(
        self,
        roi_tokens: torch.Tensor,
        alphas: torch.Tensor,
        return_per_roi: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Args:
            roi_tokens: (B, n_rois, d_model) post-Transformer ROI tokens.
            alphas:     (B, n_rois) attention weights from [CLS] -> ROI.
            return_per_roi: If True, also return per-ROI predictions.

        Returns:
            mu_fused:        (B, output_dim) fused direction.
            kappa_consensus: (B, 1) consensus concentration.
            If return_per_roi:
                per_roi_mus:    (B, n_rois, output_dim)
                per_roi_kappas: (B, n_rois, 1)
                delta:          (B, 1) disagreement score.
        """
        per_roi_mus, per_roi_kappas = self.roi_heads(roi_tokens)  # (B,R,D), (B,R,1)
        mu_fused, kappa_consensus, delta = self.fusion(
            per_roi_mus, per_roi_kappas, alphas
        )

        if return_per_roi:
            return mu_fused, kappa_consensus, per_roi_mus, per_roi_kappas, delta
        return mu_fused, kappa_consensus
