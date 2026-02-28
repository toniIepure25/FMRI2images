"""
MixCo: Mixup Contrastive Data Augmentation
===========================================

Bidirectional MixCo augmentation as described in Scotti et al. (2024) MindEye.
Interpolates pairs of (fMRI, CLIP) samples with mixing coefficient
lambda ~ Beta(alpha, alpha), producing soft contrastive labels.

Reference:
    Li et al., 2021 — MixCo: Mix-up Contrastive Learning for Visual Representation
    Scotti et al., 2024 — MindEye: adapted MixCo for fMRI-to-CLIP mapping
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def mixco_augment(
    fmri: torch.Tensor,
    clip_emb: torch.Tensor,
    alpha: float = 0.2,
    perm: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply MixCo augmentation to a batch of (fMRI, CLIP) pairs.

    Args:
        fmri: (B, V) fMRI features
        clip_emb: (B, D) CLIP embeddings
        alpha: Beta distribution parameter (smaller = more extreme mixing)
        perm: Optional pre-computed permutation for reproducibility

    Returns:
        fmri_mixed: (B, V) mixed fMRI features
        clip_mixed: (B, D) mixed CLIP embeddings
        soft_labels: (B, B) soft target matrix for cross-entropy
    """
    B = fmri.shape[0]
    lam = torch.from_numpy(
        np.random.beta(alpha, alpha, size=B).astype(np.float32)
    ).to(fmri.device)

    if perm is None:
        perm = torch.randperm(B, device=fmri.device)

    lam_v = lam.unsqueeze(1)
    fmri_mixed = lam_v * fmri + (1 - lam_v) * fmri[perm]
    clip_mixed = lam_v * clip_emb + (1 - lam_v) * clip_emb[perm]

    soft_labels = torch.zeros(B, B, device=fmri.device, dtype=torch.float32)
    arange = torch.arange(B, device=fmri.device)
    soft_labels[arange, arange] = lam
    soft_labels[arange, perm] += (1 - lam)

    return fmri_mixed, clip_mixed, soft_labels


def mixco_nce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    soft_labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute soft contrastive loss with MixCo labels.

    Args:
        pred: (B, D) predicted CLIP embeddings from model
        target: (B, D) mixed CLIP target embeddings
        soft_labels: (B, B) soft target matrix
        temperature: Contrastive temperature (aligned with InfoNCE default)

    Returns:
        Scalar loss value
    """
    pred_norm = F.normalize(pred, dim=-1)
    target_norm = F.normalize(target, dim=-1)

    logits = pred_norm @ target_norm.T / temperature
    log_probs = F.log_softmax(logits, dim=-1)

    loss_fwd = -(soft_labels * log_probs).sum(dim=-1).mean()

    logits_rev = target_norm @ pred_norm.T / temperature
    log_probs_rev = F.log_softmax(logits_rev, dim=-1)
    loss_rev = -(soft_labels.T * log_probs_rev).sum(dim=-1).mean()

    return (loss_fwd + loss_rev) / 2
