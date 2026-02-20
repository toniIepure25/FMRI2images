"""
ROI-Tokenized Transformer Encoder for Brain-Topology-Aware fMRI Decoding
=========================================================================

Groups voxels by anatomical ROI and processes them as a sequence of tokens
through a Transformer encoder.  A learnable [CLS] token aggregates
information across all ROIs.

Architecture:
    fMRI voxels -> ROI grouping -> per-ROI MLP projection -> tokens
    tokens + [CLS] + positional encoding -> Transformer -> [CLS] output

This provides *interpretability for free*: the attention weights over
ROI tokens reveal which brain regions contribute to each prediction,
connecting ML performance directly to neuroscience.

NSD ROIs used: V1v, V1d, V2v, V2d, V3v, V3d, V3A, V3B, V4, FFA-1,
FFA-2, PPA, EBA, FBA-1, FBA-2, OFA, OPA, RSC, and a catch-all
'nsdgeneral_other' for remaining voxels.

References:
    - Scotti et al. (2024) MindEye2 -- uses a similar tokenization idea
    - Dosovitskiy et al. (2021) ViT [CLS] token design
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ROITransformerOutput:
    """Output container when ``return_roi_tokens=True``.

    Attributes:
        cls_out:          (B, d_model)  [CLS] representation.
        roi_tokens:       (B, n_rois, d_model)  per-ROI token embeddings
                          from the final Transformer layer.
        cls_to_roi_alpha: (B, n_rois)  [CLS]-to-ROI attention weights
                          from the final layer, normalised over ROIs.
    """
    cls_out: torch.Tensor
    roi_tokens: torch.Tensor
    cls_to_roi_alpha: torch.Tensor


class ROIProjection(nn.Module):
    """Project variable-size ROI voxels to a fixed-dim token."""

    def __init__(self, n_voxels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_voxels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ROITransformerEncoder(nn.Module):
    """
    Brain-region-aware Transformer encoder.

    Args:
        roi_dims: Ordered dict  roi_name -> n_voxels  that defines the
                  tokenisation.  Each ROI maps to one token.  Voxels in
                  the input tensor are expected to be concatenated in this
                  same order.
        d_model:  Token / hidden dimension (default 512)
        nhead:    Number of attention heads
        num_layers: Number of Transformer encoder layers
        dropout:  Dropout probability
        activation: Activation for feedforward ('gelu' or 'relu')
    """

    def __init__(
        self,
        roi_dims: Dict[str, int],
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.roi_names: List[str] = list(roi_dims.keys())
        self.roi_sizes: List[int] = list(roi_dims.values())
        self.n_rois = len(self.roi_names)
        self.input_dim = sum(self.roi_sizes)
        self.output_dim = d_model
        self.d_model = d_model

        # Per-ROI projection layers
        self.roi_projections = nn.ModuleList([
            ROIProjection(n_voxels, d_model, dropout)
            for n_voxels in self.roi_sizes
        ])

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable positional embedding (n_rois + 1 for [CLS])
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_rois + 1, d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.final_norm = nn.LayerNorm(d_model)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"ROITransformerEncoder: {self.n_rois} ROIs "
            f"({self.input_dim} voxels) -> d_model={d_model}, "
            f"layers={num_layers}, heads={nhead}, params={n_params:,}"
        )

    def _project_rois(self, x: torch.Tensor) -> torch.Tensor:
        """Split fMRI input by ROI and project each to a d_model token.

        Args:
            x: (B, V) flattened fMRI with voxels ordered by ROI.

        Returns:
            (B, n_rois, d_model) projected ROI tokens.
        """
        tokens = []
        offset = 0
        for i, n_vox in enumerate(self.roi_sizes):
            roi_voxels = x[:, offset:offset + n_vox]
            tokens.append(self.roi_projections[i](roi_voxels))
            offset += n_vox
        return torch.stack(tokens, dim=1)

    def _prepend_cls_and_embed(
        self, tokens: torch.Tensor
    ) -> torch.Tensor:
        """Prepend [CLS] and add positional embedding.

        Args:
            tokens: (B, n_rois, d_model).

        Returns:
            (B, n_rois+1, d_model) with [CLS] at position 0.
        """
        B = tokens.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        return tokens + self.pos_embed

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_roi_tokens: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, V) flattened fMRI with voxels ordered by ROI.
            return_attention: if True, returns (cls_out, attn_weights).
                              Ignored when return_roi_tokens is True.
            return_roi_tokens: if True, returns an ROITransformerOutput
                               containing cls_out, roi_tokens, and the
                               [CLS]-to-ROI attention weights from the
                               final layer.  Designed for ROI-DCF.

        Returns:
            Default:            (B, d_model)  [CLS] token.
            return_roi_tokens:  ROITransformerOutput dataclass.
        """
        tokens = self._project_rois(x)
        tokens = self._prepend_cls_and_embed(tokens)

        if return_roi_tokens:
            return self._forward_with_roi_output(tokens)

        out = self.transformer(tokens)
        cls_out = self.final_norm(out[:, 0])
        return cls_out

    def _forward_with_roi_output(
        self, tokens: torch.Tensor
    ) -> "ROITransformerOutput":
        """Run transformer layer-by-layer to capture final-layer attention.

        Args:
            tokens: (B, n_rois+1, d_model) input including [CLS].

        Returns:
            ROITransformerOutput with cls_out, roi_tokens, and cls_to_roi_alpha.
        """
        for i, layer in enumerate(self.transformer.layers):
            if i == len(self.transformer.layers) - 1:
                # Extract attention from the last layer
                normed = layer.norm1(tokens) if hasattr(layer, "norm1") else tokens
                _, attn_weights = layer.self_attn(
                    normed, normed, normed,
                    need_weights=True,
                    average_attn_weights=True,
                )
                # attn_weights: (B, seq, seq), averaged across heads
                tokens = layer(tokens)
            else:
                tokens = layer(tokens)

        out = tokens
        cls_out = self.final_norm(out[:, 0])             # (B, d_model)
        roi_tokens = self.final_norm(out[:, 1:])          # (B, n_rois, d_model)

        # [CLS]-to-ROI attention: row 0 (CLS query), columns 1: (ROI keys)
        cls_to_roi_alpha = attn_weights[:, 0, 1:]         # (B, n_rois)
        # Re-normalise so weights sum to 1 over ROIs only
        cls_to_roi_alpha = cls_to_roi_alpha / cls_to_roi_alpha.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)

        return ROITransformerOutput(
            cls_out=cls_out,
            roi_tokens=roi_tokens,
            cls_to_roi_alpha=cls_to_roi_alpha,
        )

    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract per-layer attention weights for interpretability.

        Returns list of (B, n_heads, n_rois+1, n_rois+1) tensors.
        """
        tokens = self._project_rois(x)
        tokens = self._prepend_cls_and_embed(tokens)

        attn_weights = []
        for layer in self.transformer.layers:
            q = k = v = layer.norm1(tokens) if hasattr(layer, 'norm1') else tokens
            _, aw = layer.self_attn(q, k, v, need_weights=True, average_attn_weights=False)
            attn_weights.append(aw)  # (B, n_heads, seq, seq)
            tokens = layer(tokens)

        return attn_weights
