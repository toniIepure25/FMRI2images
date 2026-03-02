"""
Centered Kernel Alignment (CKA) Loss
=====================================

Measures global representational similarity between predicted fMRI
embeddings and ground-truth CLIP embeddings at the batch level.

While vMF-NCE enforces instance-wise alignment (each sample matches its
own target), CKA measures whether the overall *structure* of the
predicted embedding space matches the CLIP embedding space.  This
creates a "Multi-Granularity" training signal:

    - vMF-NCE:  local, per-sample alignment
    - CKA:      global, manifold-level alignment

CKA is based on the Hilbert-Schmidt Independence Criterion (HSIC):

    CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))

where HSIC uses linear kernels (K = X @ X^T) with centering:

    K_c = H @ K @ H,  H = I - (1/n) * 11^T

The loss is  1 - CKA  (minimise to maximise alignment).

References:
    - Kornblith et al. (2019) Similarity of Neural Network
      Representations Revisited
    - Nguyen et al. (2020) Do Wide and Deep Networks Learn the
      Same Things? (linear CKA)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CKALoss(nn.Module):
    """
    Centered Kernel Alignment loss using linear kernels.

    Returns ``1 - CKA(pred, target)`` so that minimising the loss
    maximises representational alignment.

    Args:
        eps: Small constant for numerical stability in the denominator.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        logger.info("CKALoss: linear kernel, eps=%s", eps)

    @staticmethod
    def _center_kernel(K: torch.Tensor) -> torch.Tensor:
        """Center a Gram matrix: K_c = H @ K @ H, H = I - 1/n."""
        n = K.size(0)
        row_mean = K.mean(dim=1, keepdim=True)
        col_mean = K.mean(dim=0, keepdim=True)
        total_mean = K.mean()
        return K - row_mean - col_mean + total_mean

    @staticmethod
    def _hsic(K_x: torch.Tensor, K_y: torch.Tensor) -> torch.Tensor:
        """Unbiased HSIC estimator with centered kernels.

        HSIC(X, Y) = tr(K_x_c @ K_y_c) / (n-1)^2
        """
        n = K_x.size(0)
        K_xc = CKALoss._center_kernel(K_x)
        K_yc = CKALoss._center_kernel(K_y)
        return (K_xc * K_yc).sum() / max((n - 1) ** 2, 1)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CKA loss.

        Args:
            pred:   (B, D) predicted embeddings.
            target: (B, D) ground-truth embeddings.

        Returns:
            Scalar loss: 1 - CKA(pred, target).
        """
        # Linear kernels
        K_x = pred @ pred.T        # (B, B)
        K_y = target @ target.T    # (B, B)

        hsic_xy = self._hsic(K_x, K_y)
        hsic_xx = self._hsic(K_x, K_x)
        hsic_yy = self._hsic(K_y, K_y)

        denom = torch.sqrt(hsic_xx * hsic_yy).clamp(min=self.eps)
        cka = hsic_xy / denom

        # Clamp CKA to [0, 1] for numerical safety
        cka = cka.clamp(0.0, 1.0)

        return 1.0 - cka
