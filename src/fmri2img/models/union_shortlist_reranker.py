"""Candidate-level residual reranker for V39 union-shortlist reranking.

This is a small model that takes per-candidate feature vectors within a
union shortlist and predicts which candidate is the correct match. It is
NOT an embedding model — it operates on expert-derived scores, ranks, and
agreement features.

Architecture:
    - Per-candidate MLP encoder: shared across all candidates in a shortlist
    - Output: one scalar logit per candidate
    - Training: shortlist-level cross-entropy (softmax over candidates)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateReranker(nn.Module):
    """Per-candidate MLP reranker.

    Args:
        input_dim: Number of features per candidate.
        hidden_dim: Hidden layer width.
        num_layers: Number of hidden layers (default 2).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score each candidate.

        Args:
            features: (B, max_candidates, F) per-candidate features
            mask: (B, max_candidates) bool, True for valid candidates

        Returns:
            logits: (B, max_candidates) with -inf for invalid positions
        """
        logits = self.net(features).squeeze(-1)  # (B, max_candidates)
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits


def shortlist_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Shortlist-level cross-entropy loss.

    For each query, treats the shortlist as a classification problem:
    which candidate is the correct match?

    Args:
        logits: (B, max_candidates) raw scores, -inf for invalid
        labels: (B, max_candidates) binary, 1 for GT candidate
        mask: (B, max_candidates) bool, True for valid candidates

    Returns:
        Scalar loss, averaged over queries where GT is in the shortlist.
    """
    # Only compute loss for queries where GT is present
    has_gt = labels.any(dim=1)  # (B,)
    if not has_gt.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    logits_gt = logits[has_gt]         # (B', max_candidates)
    labels_gt = labels[has_gt]         # (B', max_candidates)

    # Target: index of the GT candidate
    target = labels_gt.argmax(dim=1)   # (B',)

    loss = F.cross_entropy(logits_gt, target)
    return loss
