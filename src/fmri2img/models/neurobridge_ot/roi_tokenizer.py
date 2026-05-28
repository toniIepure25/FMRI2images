"""
ROI Tokenizer — Module A
=========================

Converts variable-length subject-specific voxel vectors into a fixed set
of ROI tokens (B, n_rois, d_model). Supports four tokenizer variants:

1. FixedROISummaryTokenizer: mean/std/max statistical pooling baseline.
2. LearnedROITokenizer: learned scalar voxel embedding + attention pooling.
3. SubjectAdaptiveROITokenizer: learned tokenizer with subject-conditioned adapters.
4. HyperAdapterROITokenizer: tokenizer modulated by hyper-adapter weights.
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class BaseROITokenizer(nn.Module, ABC):
    """Abstract base class for ROI tokenizers.

    All tokenizers produce (B, n_rois, d_model) token tensors from
    subject-specific fMRI voxel vectors and ROI index mappings.
    """

    def __init__(
        self,
        n_rois: int,
        d_model: int,
        roi_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_rois = n_rois
        self.d_model = d_model
        self.roi_dropout = roi_dropout

    @abstractmethod
    def forward(
        self,
        fmri: Tensor,
        roi_indices: Dict[str, Tensor],
        subject_id: Optional[Tensor] = None,
        adapters: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Tokenize fMRI voxels into ROI tokens.

        Args:
            fmri: Raw fMRI voxel activations, shape (B, V_subj).
            roi_indices: Dict mapping ROI names to index tensors into fmri dim.
            subject_id: Optional subject identifiers, shape (B,).
            adapters: Optional adapter weights from hyper-network.

        Returns:
            tokens: ROI token tensor, shape (B, n_rois, d_model).
            mask: Boolean mask, shape (B, n_rois), True = valid token.
        """
        ...

    def _gather_roi_voxels(
        self, fmri: Tensor, roi_indices: Dict[str, Tensor]
    ) -> Tuple[List[Tensor], Tensor]:
        """Gather voxels per ROI and produce a validity mask.

        Returns:
            roi_voxels: List of tensors, each (B, n_voxels_roi_i).
            mask: Boolean mask (B, n_rois), True where ROI has >0 voxels.
        """
        B = fmri.shape[0]
        device = fmri.device
        roi_voxels = []
        mask = torch.ones(B, self.n_rois, dtype=torch.bool, device=device)

        for i, (roi_name, idx) in enumerate(roi_indices.items()):
            if i >= self.n_rois:
                break
            if idx is None or len(idx) == 0:
                roi_voxels.append(torch.zeros(B, 1, device=device))
                mask[:, i] = False
            else:
                idx_dev = idx.to(device) if idx.device != device else idx
                roi_voxels.append(fmri[:, idx_dev])

        while len(roi_voxels) < self.n_rois:
            roi_voxels.append(torch.zeros(B, 1, device=device))
            mask[:, len(roi_voxels) - 1] = False

        return roi_voxels, mask

    def _apply_roi_dropout(self, tokens: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Randomly zero out ROI tokens during training for robustness."""
        if self.training and self.roi_dropout > 0.0:
            drop_mask = torch.bernoulli(
                torch.full_like(mask.float(), 1.0 - self.roi_dropout)
            ).bool()
            drop_mask = drop_mask & mask
            tokens = tokens * drop_mask.unsqueeze(-1).float()
            mask = drop_mask
        return tokens, mask


class FixedROISummaryTokenizer(BaseROITokenizer):
    """Statistical pooling baseline: mean/std/max per ROI -> linear projection.

    Concatenates [mean, std, max] of voxels per ROI (3*1 = 3 scalars per
    ROI or 3*n_voxels if keeping full stats) and projects to d_model.
    """

    def __init__(
        self,
        n_rois: int,
        d_model: int,
        pool_mode: str = "mean_std_max",
        roi_dropout: float = 0.0,
    ):
        super().__init__(n_rois=n_rois, d_model=d_model, roi_dropout=roi_dropout)
        self.pool_mode = pool_mode
        pool_dim = {"mean": 1, "mean_std": 2, "mean_std_max": 3}[pool_mode]
        self.proj = nn.Linear(pool_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        logger.info(
            "FixedROISummaryTokenizer: n_rois=%d, d_model=%d, pool=%s",
            n_rois, d_model, pool_mode,
        )

    def forward(
        self,
        fmri: Tensor,
        roi_indices: Dict[str, Tensor],
        subject_id: Optional[Tensor] = None,
        adapters: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        roi_voxels, mask = self._gather_roi_voxels(fmri, roi_indices)
        B = fmri.shape[0]
        device = fmri.device
        tokens = torch.zeros(B, self.n_rois, self.d_model, device=device)

        for i, vox in enumerate(roi_voxels):
            if not mask[:, i].any():
                continue
            stats = [vox.mean(dim=-1, keepdim=True)]
            if self.pool_mode in ("mean_std", "mean_std_max"):
                stats.append(vox.std(dim=-1, keepdim=True).clamp(min=1e-6))
            if self.pool_mode == "mean_std_max":
                stats.append(vox.max(dim=-1, keepdim=True).values)
            pooled = torch.cat(stats, dim=-1)  # (B, pool_dim)
            tokens[:, i] = self.norm(self.proj(pooled))

        tokens, mask = self._apply_roi_dropout(tokens, mask)
        return tokens, mask


class VoxelEmbedding(nn.Module):
    """Embed scalar voxel values into d_model dimensional space."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, voxels: Tensor) -> Tensor:
        """Embed voxel scalars.

        Args:
            voxels: (B, n_voxels) scalar voxel values.

        Returns:
            (B, n_voxels, d_model) embedded voxel representations.
        """
        x = voxels.unsqueeze(-1)  # (B, n_voxels, 1)
        x = self.linear(x)
        x = self.norm(x)
        x = self.dropout(F.gelu(x))
        return x


class AttentionPooling(nn.Module):
    """Multi-head attention pooling to produce a single ROI token from voxel embeddings."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, voxel_embeddings: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Pool voxel embeddings into a single token.

        Args:
            voxel_embeddings: (B, n_voxels, d_model).
            mask: Optional padding mask (B, n_voxels), True = ignore.

        Returns:
            (B, d_model) pooled token.
        """
        B = voxel_embeddings.shape[0]
        q = self.query.expand(B, -1, -1)  # (B, 1, d_model)
        out, _ = self.attn(q, voxel_embeddings, voxel_embeddings, key_padding_mask=mask)
        return self.norm(out.squeeze(1))


class LearnedROITokenizer(BaseROITokenizer):
    """Learned scalar voxel embedding + attention pooling per ROI.

    Each voxel scalar is embedded into d_model, then per-ROI attention
    pooling produces a single ROI token.
    """

    def __init__(
        self,
        n_rois: int,
        d_model: int,
        n_attn_heads: int = 4,
        dropout: float = 0.1,
        roi_dropout: float = 0.0,
        max_voxels_per_roi: int = 2000,
    ):
        super().__init__(n_rois=n_rois, d_model=d_model, roi_dropout=roi_dropout)
        self.voxel_embed = VoxelEmbedding(d_model, dropout=dropout)
        self.pool = AttentionPooling(d_model, n_heads=n_attn_heads, dropout=dropout)
        self.max_voxels_per_roi = max_voxels_per_roi
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "LearnedROITokenizer: n_rois=%d, d_model=%d, params=%d",
            n_rois, d_model, n_params,
        )

    def forward(
        self,
        fmri: Tensor,
        roi_indices: Dict[str, Tensor],
        subject_id: Optional[Tensor] = None,
        adapters: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        roi_voxels, mask = self._gather_roi_voxels(fmri, roi_indices)
        B = fmri.shape[0]
        device = fmri.device
        tokens = torch.zeros(B, self.n_rois, self.d_model, device=device)

        for i, vox in enumerate(roi_voxels):
            if not mask[:, i].any():
                continue
            if vox.shape[-1] > self.max_voxels_per_roi:
                vox = vox[:, : self.max_voxels_per_roi]
            embedded = self.voxel_embed(vox)  # (B, n_vox, d_model)
            tokens[:, i] = self.pool(embedded)

        tokens, mask = self._apply_roi_dropout(tokens, mask)
        return tokens, mask


class SubjectAdaptiveROITokenizer(BaseROITokenizer):
    """Learned tokenizer with per-subject low-rank adapters.

    Adds a small subject-specific affine modulation after voxel embedding,
    enabling the model to learn subject-specific activation patterns.
    """

    def __init__(
        self,
        n_rois: int,
        d_model: int,
        n_subjects: int,
        adapter_rank: int = 16,
        n_attn_heads: int = 4,
        dropout: float = 0.1,
        roi_dropout: float = 0.0,
        max_voxels_per_roi: int = 2000,
    ):
        super().__init__(n_rois=n_rois, d_model=d_model, roi_dropout=roi_dropout)
        self.voxel_embed = VoxelEmbedding(d_model, dropout=dropout)
        self.pool = AttentionPooling(d_model, n_heads=n_attn_heads, dropout=dropout)
        self.max_voxels_per_roi = max_voxels_per_roi

        self.subject_scale = nn.Embedding(n_subjects, d_model)
        self.subject_bias = nn.Embedding(n_subjects, d_model)
        nn.init.ones_(self.subject_scale.weight)
        nn.init.zeros_(self.subject_bias.weight)

        self.adapter_down = nn.Embedding(n_subjects, d_model * adapter_rank)
        self.adapter_up = nn.Embedding(n_subjects, adapter_rank * d_model)
        self.adapter_rank = adapter_rank

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "SubjectAdaptiveROITokenizer: n_rois=%d, d_model=%d, n_subjects=%d, params=%d",
            n_rois, d_model, n_subjects, n_params,
        )

    def forward(
        self,
        fmri: Tensor,
        roi_indices: Dict[str, Tensor],
        subject_id: Optional[Tensor] = None,
        adapters: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        roi_voxels, mask = self._gather_roi_voxels(fmri, roi_indices)
        B = fmri.shape[0]
        device = fmri.device
        tokens = torch.zeros(B, self.n_rois, self.d_model, device=device)

        scale = None
        bias = None
        adapter_matrix = None
        if subject_id is not None:
            sid = subject_id.long().to(device)
            scale = self.subject_scale(sid)  # (B, d_model)
            bias = self.subject_bias(sid)
            down = self.adapter_down(sid).view(B, self.d_model, self.adapter_rank)
            up = self.adapter_up(sid).view(B, self.adapter_rank, self.d_model)
            adapter_matrix = torch.bmm(down, up)  # (B, d_model, d_model)

        for i, vox in enumerate(roi_voxels):
            if not mask[:, i].any():
                continue
            if vox.shape[-1] > self.max_voxels_per_roi:
                vox = vox[:, : self.max_voxels_per_roi]
            embedded = self.voxel_embed(vox)  # (B, n_vox, d_model)

            if scale is not None:
                embedded = embedded * scale.unsqueeze(1) + bias.unsqueeze(1)

            if adapter_matrix is not None:
                embedded = embedded + torch.bmm(
                    embedded, adapter_matrix
                ) * 0.1  # scaled residual

            tokens[:, i] = self.pool(embedded)

        tokens, mask = self._apply_roi_dropout(tokens, mask)
        return tokens, mask


class HyperAdapterROITokenizer(BaseROITokenizer):
    """Tokenizer modulated by hyper-adapter weights from subject fingerprint.

    The hyper-network generates lightweight FiLM parameters (scale, bias)
    for each transformer layer within the tokenizer based on a subject
    fingerprint vector.
    """

    def __init__(
        self,
        n_rois: int,
        d_model: int,
        fingerprint_dim: int = 128,
        n_attn_heads: int = 4,
        dropout: float = 0.1,
        roi_dropout: float = 0.0,
        max_voxels_per_roi: int = 2000,
    ):
        super().__init__(n_rois=n_rois, d_model=d_model, roi_dropout=roi_dropout)
        self.voxel_embed = VoxelEmbedding(d_model, dropout=dropout)
        self.pool = AttentionPooling(d_model, n_heads=n_attn_heads, dropout=dropout)
        self.max_voxels_per_roi = max_voxels_per_roi
        self.fingerprint_dim = fingerprint_dim

        self.film_generator = nn.Sequential(
            nn.Linear(fingerprint_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "HyperAdapterROITokenizer: n_rois=%d, d_model=%d, fp_dim=%d, params=%d",
            n_rois, d_model, fingerprint_dim, n_params,
        )

    def forward(
        self,
        fmri: Tensor,
        roi_indices: Dict[str, Tensor],
        subject_id: Optional[Tensor] = None,
        adapters: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        roi_voxels, mask = self._gather_roi_voxels(fmri, roi_indices)
        B = fmri.shape[0]
        device = fmri.device
        tokens = torch.zeros(B, self.n_rois, self.d_model, device=device)

        film_params = None
        if adapters is not None and "fingerprint" in adapters:
            fp = adapters["fingerprint"]  # (B, fingerprint_dim)
            film_raw = self.film_generator(fp)  # (B, 2*d_model)
            film_scale = film_raw[:, : self.d_model].sigmoid() * 2.0  # (B, d_model)
            film_bias = film_raw[:, self.d_model :]
            film_params = (film_scale, film_bias)

        for i, vox in enumerate(roi_voxels):
            if not mask[:, i].any():
                continue
            if vox.shape[-1] > self.max_voxels_per_roi:
                vox = vox[:, : self.max_voxels_per_roi]
            embedded = self.voxel_embed(vox)

            if film_params is not None:
                scale, bias = film_params
                embedded = embedded * scale.unsqueeze(1) + bias.unsqueeze(1)

            tokens[:, i] = self.pool(embedded)

        tokens, mask = self._apply_roi_dropout(tokens, mask)
        return tokens, mask


def create_roi_tokenizer(config: Dict[str, Any]) -> BaseROITokenizer:
    """Factory function to create the appropriate ROI tokenizer.

    Args:
        config: Tokenizer configuration dict with key 'type'.

    Returns:
        Instantiated tokenizer module.
    """
    tok_type = config.get("type", "learned")
    n_rois = config.get("n_rois", 17)
    d_model = config.get("d_model", 768)
    roi_dropout = config.get("roi_dropout", 0.0)
    dropout = config.get("dropout", 0.1)

    if tok_type == "fixed_summary":
        return FixedROISummaryTokenizer(
            n_rois=n_rois,
            d_model=d_model,
            pool_mode=config.get("pool_mode", "mean_std_max"),
            roi_dropout=roi_dropout,
        )
    elif tok_type == "learned":
        return LearnedROITokenizer(
            n_rois=n_rois,
            d_model=d_model,
            n_attn_heads=config.get("n_attn_heads", 4),
            dropout=dropout,
            roi_dropout=roi_dropout,
            max_voxels_per_roi=config.get("max_voxels_per_roi", 2000),
        )
    elif tok_type == "subject_adaptive":
        return SubjectAdaptiveROITokenizer(
            n_rois=n_rois,
            d_model=d_model,
            n_subjects=config.get("n_subjects", 8),
            adapter_rank=config.get("adapter_rank", 16),
            n_attn_heads=config.get("n_attn_heads", 4),
            dropout=dropout,
            roi_dropout=roi_dropout,
            max_voxels_per_roi=config.get("max_voxels_per_roi", 2000),
        )
    elif tok_type == "hyper_adapter":
        return HyperAdapterROITokenizer(
            n_rois=n_rois,
            d_model=d_model,
            fingerprint_dim=config.get("fingerprint_dim", 128),
            n_attn_heads=config.get("n_attn_heads", 4),
            dropout=dropout,
            roi_dropout=roi_dropout,
            max_voxels_per_roi=config.get("max_voxels_per_roi", 2000),
        )
    else:
        raise ValueError(f"Unknown tokenizer type: {tok_type}")
