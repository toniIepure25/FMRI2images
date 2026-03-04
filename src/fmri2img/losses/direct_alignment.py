"""
Direct Cosine Alignment Loss (V11).

Provides a batch-independent gradient signal by directly maximising the
cosine similarity between predicted mu and the ground-truth CLIP embedding.

Unlike contrastive losses (vMF-NCE, SoftCLIP) which optimise *relative*
ranking within a batch, this loss gives an *absolute* alignment target
that does not depend on the batch composition.  This is especially
helpful early in training when contrastive gradients are noisy.

On the unit sphere, 1 - cos(mu, gt) = ||mu - gt||^2 / 2, so this is
equivalent to half the squared Euclidean distance between normalised
embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class DirectAlignmentLoss(nn.Module):
    """1 - cosine_similarity(pred, target), averaged over the batch.

    Args:
        reduction: ``"mean"`` or ``"none"``.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        logger.info("DirectAlignmentLoss: reduction=%s", reduction)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, D) predicted embeddings (unit-norm recommended).
            targets:     (B, D) ground-truth embeddings (unit-norm).

        Returns:
            Scalar loss (or (B,) if reduction='none').
        """
        cos = F.cosine_similarity(predictions, targets, dim=-1)  # (B,)
        loss = 1.0 - cos
        if self.reduction == "mean":
            return loss.mean()
        return loss
