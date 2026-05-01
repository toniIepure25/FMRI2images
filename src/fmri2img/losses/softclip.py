"""
SoftCLIP Knowledge Distillation Loss

Uses CLIP-CLIP similarity as soft targets instead of one-hot labels,
preserving semantic structure in the contrastive learning signal.

Standard InfoNCE treats all negatives as equally wrong, which is
semantically destructive — a dog image is more similar to a cat image
than to a building. SoftCLIP encodes this graded similarity via KL
divergence against the teacher (CLIP-CLIP) distribution.

Reference: inspired by knowledge distillation principles applied to
CLIP contrastive training (Hinton et al., 2015; Scotti et al., 2024).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SoftCLIPLoss(nn.Module):
    """KL-divergence loss using CLIP-CLIP similarity as the teacher distribution.

    Given a batch of predicted embeddings and ground-truth CLIP embeddings:
      teacher = softmax(clip @ clip^T / tau)
      student = log_softmax(pred @ clip^T / tau)
      loss    = KL(teacher || student) = -(teacher * student).sum(-1).mean()

    Optionally includes memory queue negatives for a larger contrastive pool.

    Args:
        tau: Temperature for both teacher and student distributions.
        use_queue: Whether to incorporate memory queue negatives.
        symmetric: If True, compute loss in both directions and average.
    """

    def __init__(
        self,
        tau: float = 0.07,
        use_queue: bool = True,
        symmetric: bool = True,
    ):
        super().__init__()
        self.tau = tau
        self.use_queue = use_queue
        self.symmetric = symmetric

    def forward(
        self,
        pred_embeddings: torch.Tensor,
        gt_embeddings: torch.Tensor,
        queue: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Compute SoftCLIP loss.

        Args:
            pred_embeddings: (B, D) predicted fMRI-decoded embeddings.
            gt_embeddings: (B, D) ground-truth CLIP embeddings.
            queue: Optional MemoryQueue with additional negatives.

        Returns:
            Scalar loss.
        """
        pred_norm = F.normalize(pred_embeddings.float(), dim=1, p=2)
        gt_norm = F.normalize(gt_embeddings.float(), dim=1, p=2)

        all_keys = gt_norm
        if self.use_queue and queue is not None and queue.is_ready():
            queue_embs = F.normalize(queue.get_queue().float(), dim=1, p=2)
            all_keys = torch.cat([gt_norm, queue_embs], dim=0)

        # Teacher distribution: CLIP-to-all-keys similarity
        teacher_logits = torch.matmul(gt_norm, all_keys.T) / self.tau
        teacher_dist = F.softmax(teacher_logits, dim=-1)

        # Student distribution: predicted-to-all-keys similarity
        student_logits = torch.matmul(pred_norm, all_keys.T) / self.tau
        student_log_dist = F.log_softmax(student_logits, dim=-1)

        # KL(teacher || student) per row, averaged over batch
        loss_fwd = -(teacher_dist * student_log_dist).sum(dim=-1).mean()

        if self.symmetric:
            # Reverse direction: keys-to-batch
            teacher_logits_rev = torch.matmul(all_keys, gt_norm.T) / self.tau
            teacher_dist_rev = F.softmax(teacher_logits_rev, dim=-1)

            student_logits_rev = torch.matmul(all_keys, pred_norm.T) / self.tau
            student_log_dist_rev = F.log_softmax(student_logits_rev, dim=-1)

            loss_rev = -(teacher_dist_rev * student_log_dist_rev).sum(dim=-1).mean()
            return (loss_fwd + loss_rev) / 2.0

        return loss_fwd


class VMFSoftCLIPLoss(nn.Module):
    """Probabilistic SoftCLIP for von Mises-Fisher embeddings.

    Uses the model's predicted kappa as the student's per-sample inverse
    temperature, replacing the fixed tau used in standard SoftCLIP.  The
    CLIP teacher distribution remains fixed-temperature.

    When the brain signal is clear (high kappa), the student makes a sharp
    prediction.  When the signal is noisy (low kappa), the student's
    distribution is naturally flatter, preventing the network from being
    penalized for uncertainty on ambiguous trials.

    Teacher:  P_t = softmax(gt @ keys^T / teacher_tau)
    Student:  P_s = log_softmax(kappa_i * mu_i @ keys^T)

    Args:
        teacher_tau: Fixed temperature for the CLIP teacher distribution.
        use_queue: Whether to incorporate memory queue negatives.
        symmetric: If True, compute loss in both directions and average.
    """

    def __init__(
        self,
        teacher_tau: float = 0.05,
        use_queue: bool = True,
        symmetric: bool = True,
    ):
        super().__init__()
        self.teacher_tau = teacher_tau
        self.use_queue = use_queue
        self.symmetric = symmetric
        logger.info(
            "VMFSoftCLIPLoss: teacher_tau=%s, use_queue=%s, symmetric=%s",
            teacher_tau, use_queue, symmetric,
        )

    def forward(
        self,
        mu: torch.Tensor,
        kappa: torch.Tensor,
        gt_embeddings: torch.Tensor,
        queue: Optional[nn.Module] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute vMF-SoftCLIP loss.

        Args:
            mu: (B, D) predicted mean directions (unit norm).
            kappa: (B, 1) predicted concentration parameters.
            gt_embeddings: (B, D) ground-truth CLIP embeddings.
            queue: Optional MemoryQueue with additional negatives.
            sample_weights: (B,) per-sample importance weights for
                            kappa-gated alignment (V60c). If None,
                            uniform weighting.

        Returns:
            Scalar loss.
        """
        mu_norm = F.normalize(mu.float(), dim=1, p=2)
        gt_norm = F.normalize(gt_embeddings.float(), dim=1, p=2)
        kappa_flat = kappa.float().squeeze(-1)  # (B,)

        all_keys = gt_norm
        if self.use_queue and queue is not None and queue.is_ready():
            queue_embs = F.normalize(queue.get_queue().float(), dim=1, p=2)
            all_keys = torch.cat([gt_norm, queue_embs], dim=0)

        teacher_logits = torch.matmul(gt_norm, all_keys.T) / self.teacher_tau
        teacher_dist = F.softmax(teacher_logits, dim=-1)

        cos_sim = torch.matmul(mu_norm, all_keys.T)  # (B, M)
        student_logits = kappa_flat.unsqueeze(1) * cos_sim  # (B, M)
        student_log_dist = F.log_softmax(student_logits, dim=-1)

        per_sample_fwd = -(teacher_dist * student_log_dist).sum(dim=-1)  # (B,)
        if sample_weights is not None:
            loss_fwd = (per_sample_fwd * sample_weights).mean()
        else:
            loss_fwd = per_sample_fwd.mean()

        if self.symmetric:
            mean_kappa = kappa_flat.mean()
            teacher_logits_rev = torch.matmul(all_keys, gt_norm.T) / self.teacher_tau
            teacher_dist_rev = F.softmax(teacher_logits_rev, dim=-1)

            student_logits_rev = mean_kappa * torch.matmul(all_keys, mu_norm.T)
            student_log_dist_rev = F.log_softmax(student_logits_rev, dim=-1)

            loss_rev = -(teacher_dist_rev * student_log_dist_rev).sum(dim=-1).mean()
            return (loss_fwd + loss_rev) / 2.0

        return loss_fwd
