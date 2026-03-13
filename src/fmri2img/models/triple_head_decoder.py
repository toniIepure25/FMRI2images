"""
Triple-head vMF decoder for V30 architecture.

Separates retrieval, regression, and (optional) perceptual objectives into
independent heads so that contrastive losses on the compact hypersphere
do not interfere with Euclidean regression of rich token targets.

Architecture
------------
shared_backbone(input_dim -> hidden_dims)
    |
    +-- retrieval_mu_head  -> (B, retrieval_dim) L2-normalised
    +-- retrieval_kappa_head -> (B, 1) positive concentration
    +-- regression_head    -> (B, token_dim) un-normalised
    +-- [perceptual_head]  -> (B, perceptual_dim) un-normalised (optional)

``retrieval_dim`` is configurable (768, 1024, 2048) and need not equal the
backbone's hidden dimension nor the regression output dimension.
"""

import logging
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fmri2img.models.vmf_decoder import kappa_activation

logger = logging.getLogger(__name__)


class TripleHeadOutput(NamedTuple):
    mu: torch.Tensor                        # (B, retrieval_dim) L2-normalised
    kappa: torch.Tensor                     # (B, 1) positive
    reg_pred: torch.Tensor                  # (B, token_dim) un-normalised
    perc_pred: Optional[torch.Tensor]       # (B, perceptual_dim) or None


class TripleHeadVMFDecoder(nn.Module):
    """Decoder with compact vMF retrieval head, rich regression head, and
    optional perceptual head.

    Parameters
    ----------
    input_dim : int
        Dimension of the encoder output (latent vector).
    retrieval_dim : int
        Compact retrieval space dimension (e.g. 768, 1024, 2048).
    token_dim : int
        Rich regression target dimension (e.g. 197376 = 257 * 768).
    perceptual_dim : int
        Perceptual head output dimension (e.g. 768).
    perceptual_enabled : bool
        Whether to instantiate the perceptual head.
    hidden_dims : list[int] or None
        Hidden layer sizes for the shared backbone.
    activation : str
        ``"gelu"`` or ``"relu"``.
    dropout : float
        Dropout probability in the shared backbone.
    kappa_min, kappa_max : float
        Bounds for kappa activation.
    kappa_mode : str
        ``"softplus"`` or ``"bounded_sigmoid"``.
    """

    def __init__(
        self,
        input_dim: int,
        retrieval_dim: int = 768,
        token_dim: int = 197376,
        perceptual_dim: int = 768,
        perceptual_enabled: bool = False,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "gelu",
        dropout: float = 0.1,
        kappa_min: float = 1e-3,
        kappa_max: float = 500.0,
        kappa_mode: str = "softplus",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.retrieval_dim = retrieval_dim
        self.token_dim = token_dim
        self.perceptual_dim = perceptual_dim
        self.has_perceptual = perceptual_enabled
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.kappa_mode = kappa_mode

        # --- Shared backbone ---
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

        # --- Head A: compact vMF retrieval ---
        self.retrieval_mu_head = nn.Linear(backbone_out, retrieval_dim)
        self.retrieval_kappa_head = nn.Linear(backbone_out, 1)

        # --- Head B: rich regression (un-normalised, Euclidean) ---
        self.regression_head = nn.Linear(backbone_out, token_dim)

        # --- Head C: perceptual (optional, disabled by default) ---
        self.perceptual_head: Optional[nn.Linear] = None
        if perceptual_enabled:
            self.perceptual_head = nn.Linear(backbone_out, perceptual_dim)

        # Expose output_dim for compatibility with UnifiedModel.get_config()
        self.output_dim = retrieval_dim

        self._log_architecture(backbone_out, hidden_dims)

    def _log_architecture(self, backbone_out: int,
                          hidden_dims: Optional[list[int]]) -> None:
        n_backbone = sum(p.numel() for p in self.shared_backbone.parameters())
        n_retrieval = (self.retrieval_mu_head.weight.numel()
                       + self.retrieval_mu_head.bias.numel()
                       + self.retrieval_kappa_head.weight.numel()
                       + self.retrieval_kappa_head.bias.numel())
        n_regression = (self.regression_head.weight.numel()
                        + self.regression_head.bias.numel())
        n_perceptual = 0
        if self.perceptual_head is not None:
            n_perceptual = (self.perceptual_head.weight.numel()
                            + self.perceptual_head.bias.numel())

        total = n_backbone + n_retrieval + n_regression + n_perceptual
        logger.info(
            "TripleHeadVMFDecoder: %d -> backbone(%s, %d) -> "
            "retrieval(%d, %d params) + regression(%d, %d params) "
            "+ perceptual(%s, %d params) = %.2fM total",
            self.input_dim,
            hidden_dims, backbone_out,
            self.retrieval_dim, n_retrieval,
            self.token_dim, n_regression,
            self.perceptual_dim if self.has_perceptual else "off",
            n_perceptual,
            total / 1e6,
        )

    def forward(self, h: torch.Tensor) -> TripleHeadOutput:
        """
        Args:
            h: (B, input_dim) latent features from encoder.

        Returns:
            TripleHeadOutput namedtuple with mu, kappa, reg_pred, perc_pred.
        """
        features = self.shared_backbone(h)

        # Head A: compact vMF retrieval
        mu = F.normalize(self.retrieval_mu_head(features), p=2, dim=-1)
        kappa = kappa_activation(
            self.retrieval_kappa_head(features),
            mode=self.kappa_mode,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
        )

        # Head B: rich regression (un-normalised)
        reg_pred = self.regression_head(features)

        # Head C: perceptual (optional)
        perc_pred = None
        if self.perceptual_head is not None:
            perc_pred = self.perceptual_head(features)

        return TripleHeadOutput(mu=mu, kappa=kappa,
                                reg_pred=reg_pred, perc_pred=perc_pred)
