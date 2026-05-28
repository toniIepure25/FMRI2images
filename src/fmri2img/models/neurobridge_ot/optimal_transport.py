"""
Optimal Transport ROI Alignment — Module C
============================================

Differentiable Sinkhorn optimal-transport alignment from subject ROI tokens
to a canonical cortical token space.

Core idea: each subject produces ROI tokens; a learned canonical token bank
represents a shared cortical-semantic space; Sinkhorn OT computes soft
alignment between subject ROI tokens and canonical tokens.
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def sinkhorn_transport(
    cost: Tensor,
    epsilon: float = 0.1,
    n_iters: int = 50,
    marginal_a: Optional[Tensor] = None,
    marginal_b: Optional[Tensor] = None,
) -> Tensor:
    """Compute entropy-regularized optimal transport via Sinkhorn iterations.

    Solves the entropic OT problem:
        min_{T >= 0} <T, C> + epsilon * KL(T || a b^T)
        s.t. T @ 1 = a, T^T @ 1 = b

    Args:
        cost: Cost matrix, shape (B, M, N) where M = source tokens, N = target tokens.
        epsilon: Entropy regularization strength (lower = more peaked transport).
        n_iters: Number of Sinkhorn iterations.
        marginal_a: Source marginal, shape (B, M). Defaults to uniform.
        marginal_b: Target marginal, shape (B, N). Defaults to uniform.

    Returns:
        Transport plan T, shape (B, M, N), doubly stochastic (rows/cols sum to marginals).
    """
    B, M, N = cost.shape
    device = cost.device
    dtype = cost.dtype

    if marginal_a is None:
        marginal_a = torch.ones(B, M, device=device, dtype=dtype) / M
    if marginal_b is None:
        marginal_b = torch.ones(B, N, device=device, dtype=dtype) / N

    # Gibbs kernel
    K = torch.exp(-cost / epsilon.clamp(min=1e-4) if isinstance(epsilon, Tensor) else -cost / max(epsilon, 1e-4))

    # Initialize dual variables
    u = torch.ones(B, M, device=device, dtype=dtype)
    v = torch.ones(B, N, device=device, dtype=dtype)

    for _ in range(n_iters):
        u = marginal_a / (K @ v.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
        v = marginal_b / (K.transpose(-2, -1) @ u.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)

    # Transport plan
    T = u.unsqueeze(-1) * K * v.unsqueeze(-2)  # (B, M, N)
    return T


class CostFunction(nn.Module):
    """Configurable cost function for OT alignment."""

    def __init__(self, cost_type: str = "cosine"):
        super().__init__()
        self.cost_type = cost_type

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        """Compute pairwise cost matrix.

        Args:
            source: Shape (B, M, D).
            target: Shape (B, N, D) or (N, D) if shared canonical bank.

        Returns:
            Cost matrix, shape (B, M, N).
        """
        if target.dim() == 2:
            target = target.unsqueeze(0).expand(source.shape[0], -1, -1)

        if self.cost_type == "cosine":
            source_norm = F.normalize(source, dim=-1)
            target_norm = F.normalize(target, dim=-1)
            similarity = torch.bmm(source_norm, target_norm.transpose(-2, -1))
            return 1.0 - similarity  # cosine distance
        elif self.cost_type == "euclidean":
            diff = source.unsqueeze(2) - target.unsqueeze(1)
            return (diff ** 2).sum(dim=-1)
        else:
            raise ValueError(f"Unknown cost type: {self.cost_type}")


class OptimalTransportAlignment(nn.Module):
    """Differentiable OT alignment from subject ROI tokens to canonical space.

    Maintains a learnable canonical token bank and computes soft alignment
    via Sinkhorn iterations during the forward pass.

    Args:
        n_canonical_tokens: Number of canonical tokens in the shared space.
        d_model: Token embedding dimensionality.
        n_source_tokens: Expected number of source ROI tokens.
        epsilon: Sinkhorn entropy regularization.
        n_sinkhorn_iters: Number of Sinkhorn iterations.
        cost_type: Cost function type ('cosine' or 'euclidean').
        use_anatomical_prior: Whether to add anatomical prior regularization.
        alignment_mode: One of 'ot', 'linear', 'identity', 'ot_prior'.
    """

    def __init__(
        self,
        n_canonical_tokens: int = 17,
        d_model: int = 768,
        n_source_tokens: int = 17,
        epsilon: float = 0.1,
        n_sinkhorn_iters: int = 50,
        cost_type: str = "cosine",
        use_anatomical_prior: bool = False,
        alignment_mode: str = "ot",
    ):
        super().__init__()
        self.n_canonical_tokens = n_canonical_tokens
        self.d_model = d_model
        self.n_source_tokens = n_source_tokens
        self.epsilon = epsilon
        self.n_sinkhorn_iters = n_sinkhorn_iters
        self.alignment_mode = alignment_mode
        self.use_anatomical_prior = use_anatomical_prior

        # Learnable canonical token bank
        self.canonical_bank = nn.Parameter(
            torch.empty(n_canonical_tokens, d_model)
        )
        nn.init.kaiming_uniform_(self.canonical_bank, a=math.sqrt(5))

        self.cost_fn = CostFunction(cost_type)

        # Linear alignment fallback
        if alignment_mode in ("linear", "ot"):
            self.linear_proj = nn.Linear(d_model, d_model)

        # Anatomical prior: soft preference for nearby ROI-canonical mappings
        if use_anatomical_prior:
            self.anatomical_prior = nn.Parameter(
                torch.zeros(n_source_tokens, n_canonical_tokens)
            )

        # Output projection after alignment
        self.output_norm = nn.LayerNorm(d_model)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "OptimalTransportAlignment: K=%d, d=%d, eps=%.3f, iters=%d, mode=%s, params=%d",
            n_canonical_tokens, d_model, epsilon, n_sinkhorn_iters, alignment_mode, n_params,
        )

    def forward(
        self,
        roi_tokens: Tensor,
        roi_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Align subject ROI tokens to canonical space.

        Args:
            roi_tokens: Subject ROI tokens, shape (B, M, d_model).
            roi_mask: Valid token mask, shape (B, M), True = valid.

        Returns:
            Dict with keys:
                'aligned_tokens': (B, K, d_model) aligned canonical tokens.
                'transport_matrix': (B, M, K) soft assignment.
                'ot_cost': Scalar OT transport cost.
                'canonical_bank': (K, d_model) current canonical bank state.
        """
        B, M, D = roi_tokens.shape
        device = roi_tokens.device

        if self.alignment_mode == "identity":
            aligned = roi_tokens
            if M != self.n_canonical_tokens:
                if M > self.n_canonical_tokens:
                    aligned = aligned[:, : self.n_canonical_tokens]
                else:
                    pad = torch.zeros(
                        B, self.n_canonical_tokens - M, D, device=device
                    )
                    aligned = torch.cat([aligned, pad], dim=1)
            T = torch.eye(min(M, self.n_canonical_tokens), device=device)
            T = T.unsqueeze(0).expand(B, -1, -1)
            return {
                "aligned_tokens": self.output_norm(aligned),
                "transport_matrix": T,
                "ot_cost": torch.tensor(0.0, device=device),
                "canonical_bank": self.canonical_bank.detach(),
            }

        if self.alignment_mode == "linear":
            projected = self.linear_proj(roi_tokens)  # (B, M, D)
            canonical = self.canonical_bank.unsqueeze(0).expand(B, -1, -1)
            cost = self.cost_fn(projected, canonical)
            T = sinkhorn_transport(cost, self.epsilon, self.n_sinkhorn_iters)
            aligned = torch.bmm(T.transpose(-2, -1), projected)  # (B, K, D)
            ot_cost = (T * cost).sum(dim=(-2, -1)).mean()
            return {
                "aligned_tokens": self.output_norm(aligned),
                "transport_matrix": T,
                "ot_cost": ot_cost,
                "canonical_bank": self.canonical_bank.detach(),
            }

        # Default OT or OT + prior
        canonical = self.canonical_bank.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)
        cost = self.cost_fn(roi_tokens, canonical)  # (B, M, K)

        # Apply anatomical prior as cost reduction
        if self.use_anatomical_prior and self.alignment_mode == "ot_prior":
            prior = torch.sigmoid(self.anatomical_prior)  # (M, K)
            cost = cost - 0.1 * prior.unsqueeze(0)

        # Handle masked ROIs by setting high cost
        if roi_mask is not None:
            invalid = ~roi_mask  # (B, M)
            cost = cost + invalid.unsqueeze(-1).float() * 1e6

        T = sinkhorn_transport(cost, self.epsilon, self.n_sinkhorn_iters)  # (B, M, K)

        # Transport-weighted combination: aligned = T^T @ roi_tokens
        aligned = torch.bmm(T.transpose(-2, -1), roi_tokens)  # (B, K, D)
        ot_cost = (T * cost).sum(dim=(-2, -1)).mean()

        return {
            "aligned_tokens": self.output_norm(aligned),
            "transport_matrix": T,
            "ot_cost": ot_cost,
            "canonical_bank": self.canonical_bank.detach(),
        }
