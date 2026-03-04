"""
Spherical Uniformity Regularisation Loss (V11).

Encourages predicted embeddings to be uniformly distributed on the unit
hypersphere, directly preventing hub formation and cluster collapse.

Reference:
    Wang & Isola (2020), "Understanding Contrastive Representation Learning
    through Alignment and Uniformity on the Hypersphere", ICML.

The uniformity metric is defined as:

    L_uniform = log E_{(x,y) ~ p_data} [ exp(-t * ||f(x) - f(y)||^2) ]

Lower (more negative) = more uniform.  We minimise this as a regulariser.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class UniformityLoss(nn.Module):
    """Log-average-exp of negative squared pairwise distances.

    Args:
        t: Temperature controlling sensitivity to close pairs.
           Higher t penalises close pairs more strongly.
    """

    def __init__(self, t: float = 2.0):
        super().__init__()
        self.t = t
        logger.info("UniformityLoss: t=%.1f", t)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) L2-normalised embeddings.

        Returns:
            Scalar uniformity loss.
        """
        sq_pdist = torch.pdist(embeddings, p=2).pow(2)
        return sq_pdist.mul(-self.t).exp().mean().log()
