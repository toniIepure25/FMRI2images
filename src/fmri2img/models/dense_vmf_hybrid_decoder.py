"""Dense-first retrieval decoder with an auxiliary vMF uncertainty head.

This decoder keeps retrieval primarily in a standard normalized embedding
space, while retaining a lightweight vMF head for confidence/ambiguity
signals. It is designed to avoid the collapse issues seen when the compact
vMF branch is asked to be the main retrieval expert.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fmri2img.models.vmf_decoder import kappa_activation

logger = logging.getLogger(__name__)


class DenseVMFHybridOutput(NamedTuple):
    dense_pred: torch.Tensor
    vmf_mu: torch.Tensor
    vmf_kappa: torch.Tensor
    reg_pred: Optional[torch.Tensor]
    rerank_pred: Optional[torch.Tensor]


class DenseVMFHybridDecoder(nn.Module):
    """Primary dense retrieval head + auxiliary vMF confidence head."""

    def __init__(
        self,
        input_dim: int,
        retrieval_dim: int = 768,
        token_dim: int = 197376,
        rerank_dim: int = 2048,
        rerank_enabled: bool = False,
        regression_enabled: bool = False,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "gelu",
        dropout: float = 0.1,
        kappa_min: float = 1e-3,
        kappa_max: float = 500.0,
        kappa_mode: str = "softplus",
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.retrieval_dim = retrieval_dim
        self.token_dim = token_dim
        self.rerank_dim = rerank_dim
        self.has_rerank = bool(rerank_enabled)
        self.has_regression = bool(regression_enabled)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.kappa_mode = str(kappa_mode)

        if hidden_dims is None or len(hidden_dims) == 0:
            self.shared_backbone = nn.Identity()
            backbone_out = input_dim
        else:
            layers: list[nn.Module] = []
            in_d = input_dim
            for hd in hidden_dims:
                layers.extend([
                    nn.Linear(in_d, hd),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                in_d = hd
            self.shared_backbone = nn.Sequential(*layers)
            backbone_out = hidden_dims[-1]

        self.dense_head = nn.Linear(backbone_out, retrieval_dim)
        self.vmf_mu_head = nn.Linear(backbone_out, retrieval_dim)
        self.vmf_kappa_head = nn.Linear(backbone_out, 1)

        self.regression_head: Optional[nn.Linear] = None
        if self.has_regression:
            self.regression_head = nn.Linear(backbone_out, token_dim)

        self.rerank_head: Optional[nn.Linear] = None
        if self.has_rerank:
            self.rerank_head = nn.Linear(backbone_out, rerank_dim)

        self.output_dim = retrieval_dim

        n_total = sum(p.numel() for p in self.parameters())
        logger.info(
            "DenseVMFHybridDecoder: %d -> backbone(%s, %d) -> dense(%d) + vmf(%d,+kappa) + rerank(%s) + regression(%s) = %.2fM total",
            self.input_dim,
            hidden_dims,
            backbone_out,
            self.retrieval_dim,
            self.retrieval_dim,
            self.rerank_dim if self.has_rerank else "off",
            self.token_dim if self.has_regression else "off",
            n_total / 1e6,
        )

    def forward(self, h: torch.Tensor) -> DenseVMFHybridOutput:
        features = self.shared_backbone(h)

        dense_pred = F.normalize(self.dense_head(features), p=2, dim=-1)
        vmf_mu = F.normalize(self.vmf_mu_head(features), p=2, dim=-1)
        vmf_kappa = kappa_activation(
            self.vmf_kappa_head(features),
            mode=self.kappa_mode,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
        )

        reg_pred = self.regression_head(features) if self.regression_head is not None else None
        rerank_pred = None
        if self.rerank_head is not None:
            rerank_pred = F.normalize(self.rerank_head(features), p=2, dim=-1)

        return DenseVMFHybridOutput(
            dense_pred=dense_pred,
            vmf_mu=vmf_mu,
            vmf_kappa=vmf_kappa,
            reg_pred=reg_pred,
            rerank_pred=rerank_pred,
        )
