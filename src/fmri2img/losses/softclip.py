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
