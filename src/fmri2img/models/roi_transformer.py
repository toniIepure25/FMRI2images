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


class DropPath(nn.Module):
    """Stochastic depth (Huang et al., 2016) — drops entire residual branches.

    During training, each residual block is skipped with probability
    ``drop_prob``, effectively reducing network depth on a per-sample basis.
    At test time the module is an identity.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class TransformerLayerWithDropPath(nn.Module):
    """Pre-norm Transformer encoder layer with stochastic depth on both
    self-attention and feedforward residual branches."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(d_model)
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(
        self,
        src: torch.Tensor,
        src_mask=None,
        src_key_padding_mask=None,
        **kwargs,
    ) -> torch.Tensor:
        normed = self.norm1(src)
        attn_out, _ = self.self_attn(normed, normed, normed)
        src = src + self.drop_path1(attn_out)
        src = src + self.drop_path2(self.ff(self.norm2(src)))
        return src


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
        roi_indices: Optional[Dict[str, "torch.Tensor"]] = None,
        dim_feedforward: Optional[int] = None,
        drop_path_rate: float = 0.0,
        roi_token_dropout: float = 0.0,
    ):
        super().__init__()
        self.roi_token_drop_rate = roi_token_dropout

        self._use_indices = roi_indices is not None
        _ff_dim = dim_feedforward if dim_feedforward is not None else d_model * 4

        if self._use_indices:
            self.roi_names = list(roi_indices.keys())
            self.roi_sizes = [len(idx) for idx in roi_indices.values()]
            for i, (name, idx) in enumerate(roi_indices.items()):
                self.register_buffer(
                    f"_roi_idx_{i}",
                    idx if isinstance(idx, torch.Tensor) else torch.as_tensor(idx, dtype=torch.long),
                )
        else:
            self.roi_names = list(roi_dims.keys())
            self.roi_sizes = list(roi_dims.values())

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

        # Transformer encoder with optional stochastic depth
        if drop_path_rate > 0.0:
            dpr = [drop_path_rate * i / max(num_layers - 1, 1) for i in range(num_layers)]
            layers = nn.ModuleList([
                TransformerLayerWithDropPath(
                    d_model=d_model, nhead=nhead, dim_feedforward=_ff_dim,
                    dropout=dropout, activation=activation, drop_path=dp,
                )
                for dp in dpr
            ])
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=_ff_dim,
                    dropout=dropout, activation=activation,
                    batch_first=True, norm_first=True,
                ),
                num_layers=num_layers,
            )
            self.transformer.layers = layers
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=_ff_dim,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers,
            )

        self.final_norm = nn.LayerNorm(d_model)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"ROITransformerEncoder: {self.n_rois} ROIs "
            f"({self.input_dim} voxels) -> d_model={d_model}, "
            f"ff_dim={_ff_dim}, layers={num_layers}, heads={nhead}, "
            f"drop_path={drop_path_rate:.2f}, "
            f"roi_token_drop={roi_token_dropout:.2f}, params={n_params:,}"
        )

    def _project_rois(self, x: torch.Tensor) -> torch.Tensor:
        """Split fMRI input by ROI and project each to a d_model token.

        When ``roi_indices`` were provided at construction, voxels are
        gathered by index (supporting non-contiguous ROI layouts such as
        those produced by ``build_roi_index``).  Otherwise falls back to
        the legacy contiguous-slice path.

        Args:
            x: (B, V) flattened fMRI vector.

        Returns:
            (B, n_rois, d_model) projected ROI tokens.
        """
        tokens = []
        if self._use_indices:
            for i in range(self.n_rois):
                idx = getattr(self, f"_roi_idx_{i}")
                roi_voxels = x[:, idx]
                tokens.append(self.roi_projections[i](roi_voxels))
        else:
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

        if self.training and self.roi_token_drop_rate > 0:
            mask = (torch.rand(tokens.size(0), tokens.size(1), 1,
                               device=tokens.device) > self.roi_token_drop_rate).float()
            tokens = tokens * mask

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
