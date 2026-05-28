"""
Multi-Target Decoding Heads — Module E
========================================

Multiple decoding heads operating on the shared semantic embedding:
1. CLIP global head (768-D normalized embedding).
2. Rich/token target head (257x768 CLIP token space).
3. vMF uncertainty head (mu + kappa).
4. Calibration/confidence head.
5. Target registry for optional DINO/SigLIP/OpenCLIP heads.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class CLIPGlobalHead(nn.Module):
    """Predicts a 768-D L2-normalized CLIP image embedding.

    Args:
        d_model: Input feature dimensionality.
        output_dim: CLIP embedding dimensionality (768 for ViT-L/14).
        hidden_dim: Optional intermediate hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 768,
        output_dim: int = 768,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model * 2

        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Predict normalized CLIP embedding.

        Args:
            features: Input features, shape (B, d_model).

        Returns:
            L2-normalized CLIP embedding, shape (B, output_dim).
        """
        out = self.head(features)
        return F.normalize(out, dim=-1)


class RichTokenHead(nn.Module):
    """Predicts rich CLIP token-space targets (e.g., 257x768).

    Falls back gracefully if token targets are unavailable.

    Args:
        d_model: Input feature dimensionality.
        n_tokens: Number of CLIP tokens to predict (257 = 1 CLS + 256 patch).
        token_dim: Per-token dimensionality.
        n_roi_tokens: Number of input ROI tokens for cross-attention.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_tokens: int = 257,
        token_dim: int = 768,
        n_roi_tokens: int = 17,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = token_dim

        # Learnable query tokens for cross-attention to ROI tokens
        self.token_queries = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, token_dim)

    def forward(
        self,
        cls_features: Tensor,
        roi_tokens: Tensor,
    ) -> Tensor:
        """Predict rich token targets via cross-attention.

        Args:
            cls_features: Global CLS embedding, shape (B, d_model).
            roi_tokens: Contextualized ROI tokens, shape (B, K, d_model).

        Returns:
            Token predictions, shape (B, n_tokens, token_dim).
        """
        B = cls_features.shape[0]
        queries = self.token_queries.expand(B, -1, -1)

        # Cross-attend from learned queries to ROI tokens
        kv = torch.cat([cls_features.unsqueeze(1), roi_tokens], dim=1)
        out, _ = self.cross_attn(queries, kv, kv)
        out = self.norm(out)
        return self.proj(out)


class VMFUncertaintyHead(nn.Module):
    """von Mises-Fisher uncertainty head: predicts mean direction mu and concentration kappa.

    Args:
        d_model: Input feature dimensionality.
        output_dim: Embedding dimensionality for mu.
        hidden_dim: Hidden layer dimensionality.
        kappa_min: Minimum kappa value.
        kappa_max: Maximum kappa value (for softplus clamping).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 768,
        output_dim: int = 768,
        hidden_dim: Optional[int] = None,
        kappa_min: float = 1e-3,
        kappa_max: float = 500.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model * 2
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max

        self.mu_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.kappa_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict vMF parameters.

        Args:
            features: Input features, shape (B, d_model).

        Returns:
            mu: Unit-normalized mean direction, shape (B, output_dim).
            kappa: Positive concentration parameter, shape (B,).
        """
        mu = self.mu_head(features)
        mu = F.normalize(mu, dim=-1)

        kappa_raw = self.kappa_head(features).squeeze(-1)  # (B,)
        kappa = F.softplus(kappa_raw) + self.kappa_min
        kappa = kappa.clamp(max=self.kappa_max)

        return mu, kappa


class CalibrationHead(nn.Module):
    """Scalar confidence/logit scaling head for selective retrieval.

    Args:
        d_model: Input feature dimensionality.
        hidden_dim: Hidden layer size.
    """

    def __init__(self, d_model: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Predict confidence score.

        Args:
            features: Input features, shape (B, d_model).

        Returns:
            Confidence logit, shape (B,).
        """
        return self.head(features).squeeze(-1)


class DecodingHeads(nn.Module):
    """Container for all decoding heads with feature-flag control.

    Args:
        config: Dict with head-specific configuration and enabled flags.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        d_model = config.get("d_model", 768)
        output_dim = config.get("output_dim", 768)
        dropout = config.get("dropout", 0.1)

        # Always-on CLIP global head
        self.clip_head = CLIPGlobalHead(
            d_model=d_model,
            output_dim=output_dim,
            hidden_dim=config.get("clip_hidden_dim"),
            dropout=dropout,
        )

        # Optional rich/token target head
        self.use_token_head = config.get("use_token_head", False)
        if self.use_token_head:
            self.token_head = RichTokenHead(
                d_model=d_model,
                n_tokens=config.get("n_clip_tokens", 257),
                token_dim=config.get("token_dim", 768),
                n_roi_tokens=config.get("n_canonical_tokens", 17),
                dropout=dropout,
            )

        # Optional vMF uncertainty head
        self.use_vmf_head = config.get("use_vmf_head", True)
        if self.use_vmf_head:
            self.vmf_head = VMFUncertaintyHead(
                d_model=d_model,
                output_dim=output_dim,
                hidden_dim=config.get("vmf_hidden_dim"),
                kappa_min=config.get("kappa_min", 1e-3),
                kappa_max=config.get("kappa_max", 500.0),
                dropout=dropout,
            )

        # Optional calibration head
        self.use_calibration_head = config.get("use_calibration_head", False)
        if self.use_calibration_head:
            self.calibration_head = CalibrationHead(d_model=d_model)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info("DecodingHeads: total params=%d", n_params)

    def forward(
        self,
        cls_features: Tensor,
        roi_tokens: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Run all enabled heads.

        Args:
            cls_features: Global [BRAIN_CLS] embedding, shape (B, d_model).
            roi_tokens: Contextualized ROI tokens, shape (B, K, d_model).

        Returns:
            Dict with predictions from all enabled heads.
        """
        outputs: Dict[str, Tensor] = {}

        # CLIP global embedding (always computed)
        outputs["clip_embedding"] = self.clip_head(cls_features)

        # Rich token targets
        if self.use_token_head and roi_tokens is not None:
            outputs["token_predictions"] = self.token_head(cls_features, roi_tokens)

        # vMF uncertainty
        if self.use_vmf_head:
            mu, kappa = self.vmf_head(cls_features)
            outputs["vmf_mu"] = mu
            outputs["vmf_kappa"] = kappa

        # Calibration confidence
        if self.use_calibration_head:
            outputs["confidence"] = self.calibration_head(cls_features)

        return outputs
