"""
Hierarchical CLIP Alignment Loss
=================================

Aligns per-ROI vMF predictions to the corresponding CLIP layer based on
the visual hierarchy of the brain:

    Early visual cortex (V1, V2, V3) → CLIP intermediate layer 12
    Mid-level areas (V3A, V3B, V4)   → CLIP intermediate layer 18
    High-level areas (FFA, PPA, EBA)  → CLIP final layer

This creates a neuroscience-grounded auxiliary supervision signal that
forces each ROI expert to predict features at the appropriate level of
abstraction — edges/textures for early visual cortex, semantic categories
for high-level areas.

The loss is computed as the weighted cosine similarity between the
attention-weighted average of per-ROI mu predictions within each tier
and the corresponding CLIP layer target.

References:
    - Liu et al. (2023) BrainMCLIP: Multi-layer CLIP alignment
    - Luo et al. (2024) MindHier: Hierarchical brain decoding
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Default ROI name vocabulary (must match the order used by the encoder)
DEFAULT_ROI_NAMES: List[str] = [
    "V1v", "V1d", "V2v", "V2d", "V3v", "V3d",   # early
    "V3A", "V3B", "V4",                            # mid
    "FFA1", "FFA2", "PPA", "EBA", "OFA", "OPA",    # high
    "nsdgeneral_other",                             # residual (excluded)
]

# Default tier assignments: tier_name -> list of ROI indices
DEFAULT_TIER_INDICES: Dict[str, List[int]] = {
    "early": [0, 1, 2, 3, 4, 5],      # V1v, V1d, V2v, V2d, V3v, V3d
    "mid":   [6, 7, 8],               # V3A, V3B, V4
    "high":  [9, 10, 11, 12, 13, 14],  # FFA1, FFA2, PPA, EBA, OFA, OPA
}

# Default tier -> CLIP column mapping
DEFAULT_TIER_CLIP_COLUMNS: Dict[str, str] = {
    "early": "layer_12_proj",
    "mid":   "layer_18_proj",
    "high":  "final",
}


class HierarchicalCLIPLoss(nn.Module):
    """
    Auxiliary loss aligning per-ROI predictions to CLIP layer hierarchy.

    For each visual hierarchy tier, computes the attention-weighted average
    of per-ROI mu predictions and measures cosine distance to the
    corresponding CLIP layer target.

    Args:
        tier_indices:     Dict mapping tier names to lists of ROI indices.
        tier_clip_columns: Dict mapping tier names to CLIP column names.
                           Used for logging/documentation (actual targets
                           are passed at forward time).
    """

    def __init__(
        self,
        tier_indices: Optional[Dict[str, List[int]]] = None,
        tier_clip_columns: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.tier_indices = tier_indices or DEFAULT_TIER_INDICES
        self.tier_clip_columns = tier_clip_columns or DEFAULT_TIER_CLIP_COLUMNS

        # Pre-register tier index tensors as buffers
        for tier_name, indices in self.tier_indices.items():
            self.register_buffer(
                f"_idx_{tier_name}",
                torch.tensor(indices, dtype=torch.long),
            )

        logger.info(
            "HierarchicalCLIPLoss: %d tiers (%s)",
            len(self.tier_indices),
            ", ".join(
                f"{name}={len(idx)} ROIs→{self.tier_clip_columns.get(name, '?')}"
                for name, idx in self.tier_indices.items()
            ),
        )

    def forward(
        self,
        per_roi_mus: torch.Tensor,
        alphas: torch.Tensor,
        tier_targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hierarchical alignment loss.

        Args:
            per_roi_mus:  (B, R, D) per-ROI unit-norm mean directions.
            alphas:       (B, R) attention weights from [CLS] -> ROI.
            tier_targets: Dict mapping tier names to (B, D) CLIP layer
                          embeddings (unit-norm).  Must contain all keys
                          in ``self.tier_indices``.

        Returns:
            loss:    Scalar mean cosine distance across all tiers.
            details: Dict of per-tier losses for logging.
        """
        tier_losses = []
        details: Dict[str, float] = {}

        for tier_name in self.tier_indices:
            if tier_name not in tier_targets:
                continue

            target = tier_targets[tier_name]      # (B, D)
            idx = getattr(self, f"_idx_{tier_name}")  # (T,)

            # Select per-ROI mus and alphas for this tier
            tier_mus = per_roi_mus[:, idx]         # (B, T, D)
            tier_alphas = alphas[:, idx]            # (B, T)

            # Renormalise alphas within this tier
            tier_alphas = tier_alphas / tier_alphas.sum(
                dim=-1, keepdim=True
            ).clamp(min=1e-8)                      # (B, T)

            # Weighted average of per-ROI predictions
            weighted_mu = (tier_alphas.unsqueeze(-1) * tier_mus).sum(dim=1)  # (B, D)
            weighted_mu = F.normalize(weighted_mu, p=2, dim=-1)

            # Cosine distance: 1 - cos_sim
            cos_sim = (weighted_mu * target).sum(dim=-1)  # (B,)
            tier_loss = (1.0 - cos_sim).mean()

            tier_losses.append(tier_loss)
            details[f"hier_{tier_name}"] = tier_loss.item()

        if not tier_losses:
            zero = torch.tensor(0.0, device=per_roi_mus.device)
            return zero, details

        loss = torch.stack(tier_losses).mean()
        return loss, details
