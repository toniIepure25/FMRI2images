"""
Frequency-ROI Interaction Module
================================

Learns which spatial frequency bands each brain ROI is most informative
about, providing both improved encoding and neuroscience interpretability.

Inspired by FreqSelect (2025) but applied per-ROI instead of globally:
    - V1 may encode high-frequency edges
    - PPA may encode low-frequency spatial layout
    - FFA may encode mid-frequency face-specific features

Architecture:
    For each ROI r, apply a learnable frequency filter before projection:

    FreqROI_r(f_r) = sum_b  w_{r,b} * BandFilter_b(f_r)

    where BandFilter_b applies a 1D band-pass filter in the DCT domain
    and w_{r,b} are learnable weights that sum to 1 (via softmax).

The frequency band weights are interpretable: they reveal which frequency
content each brain region provides to the decoder.

References:
    - FreqSelect (2025) — Frequency-Aware fMRI-to-Image Reconstruction
    - V1 spatial frequency tuning: De Valois et al. (1982)
"""

import logging
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FrequencyBandFilter(nn.Module):
    """
    Learnable frequency band filter in the DCT domain.

    Applies a smooth band-pass filter to 1D voxel signals using
    the Discrete Cosine Transform (DCT).

    Args:
        n_voxels: Number of voxels in the ROI.
        n_bands: Number of frequency bands to decompose into.
    """

    def __init__(self, n_voxels: int, n_bands: int = 4):
        super().__init__()
        self.n_voxels = n_voxels
        self.n_bands = n_bands

        # Precompute DCT-II basis (not trainable)
        basis = self._dct_basis(n_voxels)
        self.register_buffer("dct_basis", basis)           # (V, V)
        self.register_buffer("idct_basis", basis.T.clone()) # (V, V)

        # Precompute band masks: soft gaussian windows over frequency axis
        band_centers = torch.linspace(0, n_voxels - 1, n_bands + 2)[1:-1]
        band_width = n_voxels / n_bands
        freq_idx = torch.arange(n_voxels, dtype=torch.float32)

        masks = []
        for center in band_centers:
            mask = torch.exp(-0.5 * ((freq_idx - center) / (band_width * 0.5)) ** 2)
            masks.append(mask)
        # Stack and normalize so bands sum approximately to 1
        band_masks = torch.stack(masks, dim=0)  # (n_bands, V)
        band_masks = band_masks / (band_masks.sum(dim=0, keepdim=True) + 1e-8)
        self.register_buffer("band_masks", band_masks)

    @staticmethod
    def _dct_basis(n: int) -> torch.Tensor:
        """Compute DCT-II basis matrix."""
        basis = torch.zeros(n, n)
        for k in range(n):
            for i in range(n):
                basis[k, i] = math.cos(math.pi * k * (2 * i + 1) / (2 * n))
        basis[0] *= 1.0 / math.sqrt(n)
        basis[1:] *= math.sqrt(2.0 / n)
        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decompose input into frequency bands.

        Args:
            x: (B, V) voxel activations for one ROI.

        Returns:
            bands: (B, n_bands, V) frequency-band-filtered signals.
        """
        # DCT: project to frequency domain
        X_freq = x @ self.dct_basis.T  # (B, V)

        # Apply each band mask
        bands = []
        for b in range(self.n_bands):
            X_filtered = X_freq * self.band_masks[b]  # (B, V)
            x_filtered = X_filtered @ self.idct_basis.T  # (B, V)
            bands.append(x_filtered)

        return torch.stack(bands, dim=1)  # (B, n_bands, V)


class FreqROIProjection(nn.Module):
    """
    Per-ROI frequency-aware projection.

    Decomposes voxel signals into frequency bands, applies learnable
    band weights, and projects to token space.

    Args:
        n_voxels: Number of voxels in the ROI.
        d_model: Output token dimension.
        n_bands: Number of frequency bands.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_voxels: int,
        d_model: int,
        n_bands: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_voxels = n_voxels
        self.d_model = d_model
        self.n_bands = n_bands

        self.freq_filter = FrequencyBandFilter(n_voxels, n_bands)

        # Learnable band importance weights (before softmax)
        self.band_logits = nn.Parameter(torch.zeros(n_bands))

        # Project combined signal to d_model
        self.proj = nn.Sequential(
            nn.Linear(n_voxels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @property
    def band_weights(self) -> torch.Tensor:
        """Softmax-normalized band weights (interpretable)."""
        return F.softmax(self.band_logits, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, V) voxel activations for this ROI.

        Returns:
            token: (B, d_model) frequency-filtered ROI token.
        """
        bands = self.freq_filter(x)  # (B, n_bands, V)
        weights = self.band_weights   # (n_bands,)

        # Weighted combination of frequency bands
        weighted = (bands * weights.view(1, -1, 1)).sum(dim=1)  # (B, V)

        return self.proj(weighted)


class FreqROITransformerEncoder(nn.Module):
    """
    ROI Transformer with per-ROI frequency-aware preprocessing.

    Replaces the standard ROIProjection with FreqROIProjection for
    each brain region, adding frequency band decomposition and
    learnable band importance.

    The band weights are interpretable: after training, they reveal
    which spatial frequencies each ROI contributes most to decoding.

    Args:
        roi_dims: Dict mapping ROI name -> number of voxels.
        d_model: Token / hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer layers.
        n_bands: Number of frequency bands per ROI.
        dropout: Dropout probability.
        activation: Feedforward activation.
    """

    def __init__(
        self,
        roi_dims: Dict[str, int],
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        n_bands: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.roi_names = list(roi_dims.keys())
        self.roi_sizes = list(roi_dims.values())
        self.n_rois = len(self.roi_names)
        self.input_dim = sum(self.roi_sizes)
        self.output_dim = d_model
        self.d_model = d_model
        self.n_bands = n_bands

        # Per-ROI frequency-aware projections
        self.roi_projections = nn.ModuleList([
            FreqROIProjection(n_voxels, d_model, n_bands, dropout)
            for n_voxels in self.roi_sizes
        ])

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_rois + 1, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.final_norm = nn.LayerNorm(d_model)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "FreqROITransformerEncoder: %d ROIs (%d voxels), "
            "d_model=%d, n_bands=%d, layers=%d, params=%s",
            self.n_rois, self.input_dim, d_model, n_bands,
            num_layers, f"{n_params:,}",
        )

    def _project_rois(self, x: torch.Tensor) -> torch.Tensor:
        tokens = []
        offset = 0
        for i, n_vox in enumerate(self.roi_sizes):
            roi_voxels = x[:, offset:offset + n_vox]
            tokens.append(self.roi_projections[i](roi_voxels))
            offset += n_vox
        return torch.stack(tokens, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        return_roi_tokens: bool = False,
    ) -> torch.Tensor:
        from fmri2img.models.roi_transformer import ROITransformerOutput

        tokens = self._project_rois(x)
        B = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed

        if not return_roi_tokens:
            out = self.transformer(tokens)
            return self.final_norm(out[:, 0])

        # Layer-by-layer for attention extraction
        for i, layer in enumerate(self.transformer.layers):
            if i == len(self.transformer.layers) - 1:
                normed = layer.norm1(tokens) if hasattr(layer, "norm1") else tokens
                _, attn_weights = layer.self_attn(
                    normed, normed, normed,
                    need_weights=True, average_attn_weights=True,
                )
                tokens = layer(tokens)
            else:
                tokens = layer(tokens)

        cls_out = self.final_norm(tokens[:, 0])
        roi_tokens = self.final_norm(tokens[:, 1:])
        cls_to_roi = attn_weights[:, 0, 1:]
        cls_to_roi = cls_to_roi / cls_to_roi.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return ROITransformerOutput(
            cls_out=cls_out,
            roi_tokens=roi_tokens,
            cls_to_roi_alpha=cls_to_roi,
        )

    def get_frequency_importance(self) -> Dict[str, List[float]]:
        """
        Extract per-ROI frequency band importance weights.

        Returns:
            Dict mapping ROI name -> list of band weights (sum to 1).
            Interpretable: higher weight = that frequency band is more
            important for that brain region's contribution to decoding.
        """
        importance = {}
        for name, proj in zip(self.roi_names, self.roi_projections):
            weights = proj.band_weights.detach().cpu().tolist()
            importance[name] = weights
        return importance
