"""
Von Mises-Fisher decoder with bounded-sigmoid kappa parameterisation.

The vMF distribution is the natural probabilistic model for L2-normalised
CLIP embeddings on the unit hypersphere S^{d-1}:

    p(z | mu, kappa) = C_d(kappa) * exp(kappa * mu^T z)

This decoder outputs:
    mu    — (B, D) unit-norm mean direction
    kappa — (B, 1) concentration in [kappa_min, kappa_max]

Kappa is parameterised as a bounded sigmoid for smooth, gradient-friendly
behaviour without hard clamps:

    kappa = kappa_min + (kappa_max - kappa_min) * sigmoid(raw)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VonMisesFisherDecoder(nn.Module):
    """
    Geometry-correct vMF decoder with bounded-sigmoid concentration.

    Args:
        input_dim:   Latent dimension from encoder
        output_dim:  Output dimension (CLIP embedding size, e.g. 768)
        hidden_dims: Hidden layer dimensions for the shared backbone
        activation:  Activation function ("gelu" or "relu")
        dropout:     Dropout probability
        kappa_min:   Lower bound for concentration (prevents collapse)
        kappa_max:   Upper bound for concentration (prevents overflow)
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
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max

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
            f"VonMisesFisherDecoder: {input_dim} -> (mu, kappa) {output_dim} "
            f"(hidden={hidden_dims}, kappa=[{kappa_min}, {kappa_max}])"
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, input_dim) latent features

        Returns:
            mu:    (B, output_dim) L2-normalised mean direction
            kappa: (B, 1) concentration in [kappa_min, kappa_max]
        """
        features = self.shared_backbone(h)

        mu = nn.functional.normalize(self.mu_head(features), p=2, dim=-1)

        raw_kappa = self.kappa_head(features)
        kappa = self.kappa_min + (self.kappa_max - self.kappa_min) * torch.sigmoid(raw_kappa)

        return mu, kappa
