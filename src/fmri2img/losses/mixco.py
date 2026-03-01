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


def slerp(
    v0: torch.Tensor,
    v1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Spherical linear interpolation on the unit hypersphere.

    Interpolates along the great-circle arc between v0 and v1,
    preserving angular relationships required for vMF optimization.

    slerp(v0, v1, t) = sin((1-t)*theta) / sin(theta) * v0
                      + sin(t*theta) / sin(theta) * v1

    where theta = arccos(v0 . v1).

    Args:
        v0: (B, D) unit-norm embeddings (start).
        v1: (B, D) unit-norm embeddings (end).
        t: (B,) interpolation parameter in [0, 1].

    Returns:
        (B, D) interpolated unit-norm embeddings on the sphere.
    """
    dot = (v0 * v1).sum(dim=-1, keepdim=True).clamp(-0.9999, 0.9999)
    theta = torch.acos(dot)  # (B, 1)
    sin_theta = torch.sin(theta).clamp(min=1e-6)

    t_2d = t.unsqueeze(-1)  # (B, 1)
    s0 = torch.sin((1.0 - t_2d) * theta) / sin_theta
    s1 = torch.sin(t_2d * theta) / sin_theta

    return s0 * v0 + s1 * v1


def mixco_augment(
    fmri: torch.Tensor,
    clip_emb: torch.Tensor,
    alpha: float = 0.2,
    perm: torch.Tensor | None = None,
    use_slerp: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply MixCo augmentation to a batch of (fMRI, CLIP) pairs.

    Args:
        fmri: (B, V) fMRI features
        clip_emb: (B, D) CLIP embeddings (L2-normalized)
        alpha: Beta distribution parameter (smaller = more extreme mixing)
        perm: Optional pre-computed permutation for reproducibility
        use_slerp: If True, mix CLIP embeddings via spherical linear
            interpolation (great-circle arc) instead of linear interpolation.

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

    if use_slerp:
        clip_mixed = slerp(clip_emb, clip_emb[perm], 1.0 - lam)
    else:
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
