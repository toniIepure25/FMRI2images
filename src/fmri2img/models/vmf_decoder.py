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
from typing import Literal, Optional, Tuple, Union
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

    Supports an optional **regression head** (MindEye-style dual-head) that
    produces un-normalised output in R^d alongside the L2-normalised mu on
    S^{d-1}.  MSE regression on un-normalised predictions avoids the mean-
    collapse failure observed in V13/V22/V23c (where MSE on L2-normalised
    vectors pulled predictions toward the hypersphere mean).

    Args:
        input_dim:   Latent dimension from encoder.
        output_dim:  Output dimension (CLIP embedding size, e.g. 768 or 197376).
        hidden_dims: Hidden layer dimensions for the shared backbone.
        activation:  Activation function ("gelu" or "relu").
        dropout:     Dropout probability.
        kappa_min:   Lower bound for concentration (bounded_sigmoid mode).
        kappa_max:   Upper bound for concentration (bounded_sigmoid mode).
        kappa_mode:  ``"bounded_sigmoid"`` (default) or ``"softplus"``.
        num_tokens:  If > 0, enable token-level output mode.  ``output_dim``
                     should equal ``num_tokens * token_dim``.  The mu head
                     output is reshaped to ``(B, num_tokens, token_dim)``
                     and L2-normalised **per-token**, then re-flattened and
                     globally L2-normalised (MindEye-style).
        token_dim:   Per-token dimension (e.g. 768 for ViT-L/14 projected).
        regression_head: If ``True``, adds a second linear head that outputs
                     un-normalised predictions for MSE regression.  The
                     forward method returns ``(mu, kappa, reg_pred)`` when
                     enabled instead of ``(mu, kappa)``.
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
        num_tokens: int = 0,
        token_dim: int = 768,
        regression_head: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.kappa_mode = kappa_mode
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.has_regression_head = regression_head

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

        # Optional regression head: un-normalised output for MSE (dual-head)
        self.regression_head: Optional[nn.Linear] = None
        if regression_head:
            self.regression_head = nn.Linear(backbone_out_dim, output_dim)
            logger.info(
                "VonMisesFisherDecoder DUAL-HEAD: regression_head enabled "
                "(%d -> %d, un-normalised output for MSE)",
                backbone_out_dim, output_dim,
            )

        if num_tokens > 0:
            assert output_dim == num_tokens * token_dim, (
                f"output_dim ({output_dim}) must equal "
                f"num_tokens * token_dim ({num_tokens} * {token_dim} = {num_tokens * token_dim})"
            )
            logger.info(
                "VonMisesFisherDecoder TOKEN MODE: %d -> %d tokens × %d dim "
                "(hidden=%s, kappa_mode=%s, kappa=[%s, %s])",
                input_dim, num_tokens, token_dim, hidden_dims, kappa_mode,
                kappa_min, kappa_max,
            )
        else:
            logger.info(
                "VonMisesFisherDecoder: %d -> (mu, kappa) %d "
                "(hidden=%s, kappa_mode=%s, kappa=[%s, %s])",
                input_dim, output_dim, hidden_dims, kappa_mode,
                kappa_min, kappa_max,
            )

    def forward(self, h: torch.Tensor) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Args:
            h: (B, input_dim) latent features

        Returns:
            Without regression head:
                mu:    (B, output_dim) L2-normalised mean direction.
                kappa: (B, 1) positive concentration
            With regression head (dual-head mode):
                mu:       (B, output_dim) L2-normalised mean direction.
                kappa:    (B, 1) positive concentration
                reg_pred: (B, output_dim) un-normalised regression prediction
        """
        features = self.shared_backbone(h)

        raw_mu = self.mu_head(features)  # (B, output_dim)

        if self.num_tokens > 0:
            # Reshape → per-token L2-norm → flatten → global L2-norm
            B = raw_mu.shape[0]
            tokens = raw_mu.view(B, self.num_tokens, self.token_dim)
            tokens = F.normalize(tokens, p=2, dim=-1)       # per-token
            mu = tokens.reshape(B, -1)                       # (B, T*D)
            mu = F.normalize(mu, p=2, dim=-1)                # global
        else:
            mu = F.normalize(raw_mu, p=2, dim=-1)

        raw_kappa = self.kappa_head(features)
        kappa = kappa_activation(
            raw_kappa,
            mode=self.kappa_mode,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
        )

        if self.regression_head is not None:
            reg_pred = self.regression_head(features)  # (B, output_dim) un-normalised
            return mu, kappa, reg_pred

        return mu, kappa
