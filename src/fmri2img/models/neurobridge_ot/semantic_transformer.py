"""
Shared Semantic Transformer — Module D
========================================

Operates on aligned ROI tokens in the canonical cortical space.
Produces a global brain semantic embedding (from [BRAIN_CLS]) and
per-ROI contextualized tokens for downstream heads.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class DropPath(nn.Module):
    """Stochastic depth (drop path) for residual blocks."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob


class PreNormTransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block with stochastic depth.

    Architecture:
        x -> LN -> MHSA -> DropPath -> + x
        x -> LN -> FFN  -> DropPath -> + x
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(d_model)
        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input tokens, shape (B, S, d_model).
            key_padding_mask: Mask for padded positions (B, S), True = ignore.

        Returns:
            Output tokens, shape (B, S, d_model).
        """
        # Self-attention with pre-norm
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = residual + self.drop_path1(attn_out)

        # FFN with pre-norm
        residual = x
        x = residual + self.drop_path2(self.ffn(self.norm2(x)))
        return x


class SemanticTransformer(nn.Module):
    """Shared semantic transformer backbone for NeuroBridge-OT.

    Processes aligned canonical ROI tokens with a prepended [BRAIN_CLS]
    token. Supports optional subject embedding, stochastic depth,
    gradient checkpointing, and mixed precision.

    Args:
        d_model: Model dimensionality.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        dim_feedforward: FFN hidden dimensionality (default 4*d_model).
        dropout: Attention and FFN dropout rate.
        drop_path_rate: Maximum stochastic depth rate (linearly increases).
        n_canonical_tokens: Number of input canonical tokens (for positional embedding).
        n_subjects: Number of subjects for optional subject embedding.
        use_subject_embedding: Whether to inject subject embedding.
        activation: Activation function name.
        gradient_checkpointing: Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 6,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        drop_path_rate: float = 0.15,
        n_canonical_tokens: int = 17,
        n_subjects: int = 8,
        use_subject_embedding: bool = False,
        activation: str = "gelu",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.gradient_checkpointing = gradient_checkpointing

        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        # [BRAIN_CLS] learnable token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional embeddings for CLS + canonical tokens
        max_seq_len = n_canonical_tokens + 1  # +1 for CLS
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # ROI type embedding (learnable per canonical slot)
        self.roi_type_embed = nn.Parameter(
            torch.randn(1, n_canonical_tokens, d_model) * 0.02
        )

        # Optional subject embedding
        self.use_subject_embedding = use_subject_embedding
        if use_subject_embedding:
            self.subject_embed = nn.Embedding(n_subjects, d_model)
            nn.init.normal_(self.subject_embed.weight, std=0.02)

        # Stochastic depth schedule (linear increase)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            PreNormTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                drop_path=dpr[i],
                activation=activation,
            )
            for i in range(n_layers)
        ])

        # Final LayerNorm
        self.final_norm = nn.LayerNorm(d_model)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "SemanticTransformer: d=%d, heads=%d, layers=%d, ff=%d, params=%d",
            d_model, n_heads, n_layers, dim_feedforward, n_params,
        )

    def forward(
        self,
        aligned_tokens: Tensor,
        subject_id: Optional[Tensor] = None,
        token_mask: Optional[Tensor] = None,
        return_all_layers: bool = False,
    ) -> Dict[str, Tensor]:
        """Process aligned ROI tokens through shared transformer.

        Args:
            aligned_tokens: Canonical-space tokens, shape (B, K, d_model).
            subject_id: Optional subject IDs for seen-subject embedding, shape (B,).
            token_mask: Valid token mask (B, K), True = valid.
            return_all_layers: Whether to return intermediate layer outputs.

        Returns:
            Dict with keys:
                'cls_embedding': Global brain semantic embedding, (B, d_model).
                'roi_tokens': Per-ROI contextualized tokens, (B, K, d_model).
                'all_layers': Optional list of per-layer outputs.
        """
        B, K, D = aligned_tokens.shape
        device = aligned_tokens.device

        # Add ROI type embeddings
        if K <= self.roi_type_embed.shape[1]:
            aligned_tokens = aligned_tokens + self.roi_type_embed[:, :K, :]
        else:
            # If K exceeds expected, pad type embedding
            extra = K - self.roi_type_embed.shape[1]
            pad = torch.zeros(1, extra, D, device=device)
            type_embed = torch.cat([self.roi_type_embed, pad], dim=1)
            aligned_tokens = aligned_tokens + type_embed

        # Optional subject embedding (added to all tokens)
        if self.use_subject_embedding and subject_id is not None:
            sid = subject_id.long().to(device)
            subj_emb = self.subject_embed(sid).unsqueeze(1)  # (B, 1, D)
            aligned_tokens = aligned_tokens + subj_emb

        # Prepend [BRAIN_CLS]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, aligned_tokens], dim=1)  # (B, 1+K, D)

        # Add positional embeddings
        seq_len = x.shape[1]
        if seq_len <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, :seq_len, :]
        else:
            # Interpolate positional embeddings if sequence is longer
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2), size=seq_len, mode="linear"
            ).transpose(1, 2)
            x = x + pos

        # Build padding mask for transformer (True = ignore)
        key_padding_mask = None
        if token_mask is not None:
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
            key_padding_mask = ~torch.cat([cls_mask, token_mask], dim=1)

        # Forward through transformer blocks
        all_layers = []
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, key_padding_mask, use_reentrant=False
                )
            else:
                x = block(x, key_padding_mask=key_padding_mask)
            if return_all_layers:
                all_layers.append(x.clone())

        x = self.final_norm(x)

        cls_out = x[:, 0, :]  # (B, D) — global brain semantic embedding
        roi_out = x[:, 1:, :]  # (B, K, D) — per-ROI contextualized tokens

        result = {
            "cls_embedding": cls_out,
            "roi_tokens": roi_out,
        }
        if return_all_layers:
            result["all_layers"] = all_layers

        return result
