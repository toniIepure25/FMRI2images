"""
Contrastive Projection Head
============================

Separate MLP projection head for contrastive losses (SimCLR-style).

The key insight (Chen et al., 2020; Scotti et al., 2024) is that
contrastive losses distort the representation in ways that hurt
downstream tasks.  By training contrastive losses through a *separate*
projection, the backbone representation stays clean for retrieval.

In the vMF context, kappa remains attached to the *pre-projection*
representation, coupling uncertainty naturally with the retrieval
embedding rather than the contrastive-optimized one.

References:
    - Chen et al. (2020) SimCLR — projection head dramatically improves
      contrastive representation quality
    - Scotti et al. (2024) MindEye2 — separate retrieval submodule
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class ContrastiveProjectionHead(nn.Module):
    """MLP projection head for contrastive losses.

    Architecture: Linear → LayerNorm → GELU → Dropout → Linear → L2-norm

    Args:
        d_model:    Input dimension (backbone output / mu dimension).
        hidden_dim: Hidden layer width.
        out_dim:    Output dimension (should match CLIP embedding dim).
        dropout:    Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: int = 2048,
        out_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        logger.info(
            "ContrastiveProjectionHead: %d -> %d -> %d (dropout=%.2f)",
            d_model, hidden_dim, out_dim, dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalise.

        Args:
            x: (B, d_model) input embedding (detached from kappa path).

        Returns:
            (B, out_dim) L2-normalised projected embedding.
        """
        return F.normalize(self.mlp(x), p=2, dim=-1)
