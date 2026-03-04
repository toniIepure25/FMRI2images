"""
Von Mises-Fisher decoder with configurable kappa parameterisation.

The vMF distribution is the natural probabilistic model for L2-normalised
CLIP embeddings on the unit hypersphere S^{d-1}:

    p(z | mu, kappa) = C_d(kappa) * exp(kappa * mu^T z)

This decoder outputs:
    mu    — (B, D) unit-norm mean direction
    kappa — (B, 1) concentration parameter (> 0)

Kappa modes:
    bounded_sigmoid: kappa = kappa_min + (kappa_max - kappa_min) * sigmoid(raw)
                     Smooth but gradient-dead near bounds.
    softplus:        kappa = softplus(raw) + 1.0
                     Clamped at min(kappa_max, 5000) for stability.
                     Set kappa_max in config to control effective cap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

KAPPA_AMP_CEIL = 5000.0


def kappa_activation(
    raw: torch.Tensor,
    mode: str = "bounded_sigmoid",
    kappa_min: float = 1e-3,
    kappa_max: float = 500.0,
) -> torch.Tensor:
    """Convert raw logits to positive kappa values.

    Args:
        raw:       (*, 1) unbounded logits from a linear head.
        mode:      ``"bounded_sigmoid"`` or ``"softplus"``.
        kappa_min: Lower bound (bounded_sigmoid) or floor (softplus).
        kappa_max: Upper bound (bounded_sigmoid) or AMP clamp (softplus).

    Returns:
        kappa with the same shape, guaranteed > 0.
    """
    if mode == "softplus":
        return (F.softplus(raw) + 1.0).clamp(max=min(kappa_max, KAPPA_AMP_CEIL))
    # Default: bounded_sigmoid (backward-compatible)
    return kappa_min + (kappa_max - kappa_min) * torch.sigmoid(raw)


class VonMisesFisherDecoder(nn.Module):
    """
    Geometry-correct vMF decoder with configurable concentration activation.

    Args:
        input_dim:   Latent dimension from encoder.
        output_dim:  Output dimension (CLIP embedding size, e.g. 768).
        hidden_dims: Hidden layer dimensions for the shared backbone.
        activation:  Activation function ("gelu" or "relu").
        dropout:     Dropout probability.
        kappa_min:   Lower bound for concentration (bounded_sigmoid mode).
        kappa_max:   Upper bound for concentration (bounded_sigmoid mode).
        kappa_mode:  ``"bounded_sigmoid"`` (default) or ``"softplus"``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "gelu",
        dropout: float = 0.1,
        kappa_min: float = 1e-3,
        kappa_max: float = 500.0,
        kappa_mode: str = "bounded_sigmoid",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.kappa_mode = kappa_mode

        if hidden_dims is None or len(hidden_dims) == 0:
            self.shared_backbone = nn.Identity()
            backbone_out_dim = input_dim
        else:
            layers: list[nn.Module] = []
            in_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                in_dim = hidden_dim
            self.shared_backbone = nn.Sequential(*layers)
            backbone_out_dim = hidden_dims[-1]

        self.mu_head = nn.Linear(backbone_out_dim, output_dim)
        self.kappa_head = nn.Linear(backbone_out_dim, 1)

        logger.info(
            "VonMisesFisherDecoder: %d -> (mu, kappa) %d "
            "(hidden=%s, kappa_mode=%s, kappa=[%s, %s])",
            input_dim, output_dim, hidden_dims, kappa_mode,
            kappa_min, kappa_max,
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, input_dim) latent features

        Returns:
            mu:    (B, output_dim) L2-normalised mean direction
            kappa: (B, 1) positive concentration
        """
        features = self.shared_backbone(h)

        mu = F.normalize(self.mu_head(features), p=2, dim=-1)

        raw_kappa = self.kappa_head(features)
        kappa = kappa_activation(
            raw_kappa,
            mode=self.kappa_mode,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
        )

        return mu, kappa
