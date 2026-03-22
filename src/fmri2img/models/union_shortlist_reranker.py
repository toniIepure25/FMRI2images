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


class VMFEvidenceReranker(nn.Module):
    """Shortlist resolver with candidate context for vMF-aware evidence fusion.

    This model still operates on precomputed candidate features, but it adds
    shortlist-context pooling so the final score can depend on both
    candidate-local evidence and the overall disagreement structure of the
    shortlist. It is intended for OOF union-shortlist reranking where the
    feature set already contains compact / legacy / rerank scores plus
    vMF-derived ambiguity cues.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        cand_layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(num_layers, 1)):
            cand_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        self.candidate_encoder = nn.Sequential(*cand_layers)

        fusion_dim = hidden_dim * 3 + input_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score each candidate using local and shortlist-context evidence."""
        cand_repr = self.candidate_encoder(features)  # (B, K, H)
        mask_f = mask.unsqueeze(-1).float()

        denom = mask_f.sum(dim=1).clamp_min(1.0)
        pooled_mean = (cand_repr * mask_f).sum(dim=1) / denom

        neg_inf = torch.full_like(cand_repr, float("-inf"))
        pooled_max = torch.where(mask.unsqueeze(-1), cand_repr, neg_inf).amax(dim=1)
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))

        pooled_mean = pooled_mean.unsqueeze(1).expand_as(cand_repr)
        pooled_max = pooled_max.unsqueeze(1).expand_as(cand_repr)
        fused = torch.cat([features, cand_repr, pooled_mean, pooled_max], dim=-1)

        logits = self.head(fused).squeeze(-1)
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



def shortlist_pairwise_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 0.2,
    hard_neg_k: int = 5,
) -> torch.Tensor:
    """Pairwise margin loss on shortlist logits.

    Encourages the GT candidate to outrank the hardest negatives by a margin.
    This complements shortlist cross-entropy by explicitly shaping the local
    ordering near the top of the shortlist.
    """
    has_gt = labels.any(dim=1)
    if not has_gt.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    logits_gt = logits[has_gt]
    labels_gt = labels[has_gt].bool()
    mask_gt = mask[has_gt]

    pos_idx = labels_gt.float().argmax(dim=1, keepdim=True)
    pos_logits = logits_gt.gather(1, pos_idx)

    neg_logits = logits_gt.masked_fill(labels_gt | ~mask_gt, float("-inf"))
    max_valid_neg = int(mask_gt.sum(dim=1).min().item()) - 1
    top_k = min(max(hard_neg_k, 1), max(max_valid_neg, 1))
    hard_negs = neg_logits.topk(top_k, dim=1).values
    hard_negs = hard_negs[torch.isfinite(hard_negs)]
    if hard_negs.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    pos_expanded = pos_logits.expand(-1, top_k).reshape(-1)
    margin_loss = F.relu(margin - pos_expanded[: hard_negs.numel()] + hard_negs)
    return margin_loss.mean()
