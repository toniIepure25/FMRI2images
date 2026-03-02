"""
Multi-Subject ROI Transformer Encoder
======================================

Wraps per-subject ROI projection layers around a **shared** Transformer
backbone.  All subjects produce the same (B, n_rois, d_model) token tensor
after their subject-specific projections, which is then processed by a
single shared Transformer encoder and decoder.

This architecture quadruples the effective training data for ROI
Transformer models (4 NSD subjects x ~24K trials each) while keeping the
shared backbone identical across subjects.

Architecture::

    Per-Subject ROI Projections (nn.ModuleDict per subject)
                ↓
    Subject Embedding (nn.Embedding, added to all tokens)
                ↓
    Shared [CLS] Token + Positional Encoding
                ↓
    Shared Transformer Encoder Layers
                ↓
    Shared Final LayerNorm
                ↓
    [CLS] output  (or ROITransformerOutput with per-ROI tokens)

References:
    - Scotti et al., 2024 (MindEye2) — multi-subject pretraining
    - Dosovitskiy et al., 2021 (ViT) — [CLS] token design
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from fmri2img.models.roi_transformer import (
    ROIProjection,
    ROITransformerOutput,
)

logger = logging.getLogger(__name__)


class MultiSubjectROITransformer(nn.Module):
    """
    ROI Transformer with per-subject input projections and a shared backbone.

    Args:
        subject_roi_dims: ``{subject_id: {roi_name: n_voxels, ...}, ...}``
            Defines per-subject ROI sizes.  ROI **names** must be identical
            across subjects (same ROI vocabulary); only voxel counts differ.
        subject_roi_indices: Optional ``{subject_id: {roi_name: Tensor, ...}}``
            for index-based voxel gathering (from ``build_roi_index``).
        d_model:          Token / hidden dimension.
        nhead:            Number of attention heads.
        num_layers:       Number of Transformer encoder layers.
        dim_feedforward:  Feedforward dimension (default 4 * d_model).
        dropout:          Dropout probability.
        activation:       Activation for feedforward sublayers.
    """

    def __init__(
        self,
        subject_roi_dims: Dict[str, Dict[str, int]],
        subject_roi_indices: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 6,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.d_model = d_model
        self.output_dim = d_model

        _ff_dim = dim_feedforward or d_model * 4

        self.subjects = sorted(subject_roi_dims.keys())
        self.n_subjects = len(self.subjects)
        self._subj_to_int = {s: i for i, s in enumerate(self.subjects)}

        ref_rois = list(subject_roi_dims[self.subjects[0]].keys())
        self.roi_names = ref_rois
        self.n_rois = len(ref_rois)

        for subj in self.subjects:
            if list(subject_roi_dims[subj].keys()) != ref_rois:
                raise ValueError(
                    f"ROI name mismatch: {subj} has "
                    f"{list(subject_roi_dims[subj].keys())} "
                    f"vs reference {ref_rois}"
                )

        # --- Per-subject ROI projections ---
        self._use_indices = subject_roi_indices is not None
        self.subject_projections = nn.ModuleDict()
        self._subject_roi_sizes: Dict[str, List[int]] = {}

        for subj in self.subjects:
            roi_dims = subject_roi_dims[subj]
            projs = nn.ModuleList()

            if self._use_indices and subj in subject_roi_indices:
                idx_dict = subject_roi_indices[subj]
                sizes = []
                for roi_name in self.roi_names:
                    idx = idx_dict[roi_name]
                    n_vox = len(idx)
                    sizes.append(n_vox)
                    projs.append(ROIProjection(n_vox, d_model, dropout))
                self._subject_roi_sizes[subj] = sizes

                for i, roi_name in enumerate(self.roi_names):
                    idx = idx_dict[roi_name]
                    self.register_buffer(
                        f"_idx_{subj}_{i}",
                        idx if isinstance(idx, torch.Tensor)
                        else torch.as_tensor(idx, dtype=torch.long),
                    )
            else:
                sizes = []
                for roi_name in self.roi_names:
                    n_vox = roi_dims[roi_name]
                    sizes.append(n_vox)
                    projs.append(ROIProjection(n_vox, d_model, dropout))
                self._subject_roi_sizes[subj] = sizes

            self.subject_projections[subj] = projs

        # --- Subject embedding ---
        self.subject_embedding = nn.Embedding(self.n_subjects, d_model)

        # --- Shared Transformer backbone ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_rois + 1, d_model) * 0.02
        )

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
            "MultiSubjectROITransformer: %d subjects, %d ROIs, "
            "d_model=%d, ff=%d, layers=%d, heads=%d, params=%s",
            self.n_subjects, self.n_rois,
            d_model, _ff_dim, num_layers, nhead,
            f"{n_params:,}",
        )

    def _project_rois_for_subject(
        self,
        x: torch.Tensor,
        subject_id: str,
    ) -> torch.Tensor:
        """Project a single subject's fMRI vector into ROI tokens.

        Args:
            x:          (B, V_subj) raw fMRI vector for *one* subject.
            subject_id: Which subject this belongs to.

        Returns:
            (B, n_rois, d_model) token tensor.
        """
        projs = self.subject_projections[subject_id]
        tokens = []

        if self._use_indices:
            for i in range(self.n_rois):
                idx = getattr(self, f"_idx_{subject_id}_{i}")
                roi_voxels = x[:, idx]
                tokens.append(projs[i](roi_voxels))
        else:
            offset = 0
            for i, n_vox in enumerate(self._subject_roi_sizes[subject_id]):
                roi_voxels = x[:, offset:offset + n_vox]
                tokens.append(projs[i](roi_voxels))
                offset += n_vox

        return torch.stack(tokens, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: Union[torch.Tensor, List[int], int],
        return_roi_tokens: bool = False,
    ) -> Union[torch.Tensor, "ROITransformerOutput"]:
        """
        Forward pass for a (possibly mixed-subject) batch.

        For training efficiency, batches should be **single-subject** (all
        samples from one subject).  Mixed-subject batches are supported but
        slower because samples are grouped and projected separately.

        Args:
            x:            (B, V_subj) fMRI features.  When the batch is
                          single-subject, V_subj is the voxel count of that
                          subject.  For mixed batches, V must be padded to
                          the maximum across subjects.
            subject_ids:  Integer subject IDs (one per sample, or a scalar
                          when the whole batch is from one subject).
            return_roi_tokens: If True, return ``ROITransformerOutput``.

        Returns:
            Default:            (B, d_model) — the [CLS] token.
            return_roi_tokens:  ``ROITransformerOutput`` (cls_out + roi_tokens
                                + cls_to_roi_alpha).
        """
        if isinstance(subject_ids, int):
            return self._forward_single_subject(
                x, self.subjects[subject_ids], return_roi_tokens,
            )

        if isinstance(subject_ids, torch.Tensor):
            subject_ids_list = subject_ids.tolist()
        else:
            subject_ids_list = list(subject_ids)

        unique_ids = set(subject_ids_list)
        if len(unique_ids) == 1:
            return self._forward_single_subject(
                x, self.subjects[subject_ids_list[0]], return_roi_tokens,
            )

        return self._forward_mixed_batch(
            x, subject_ids_list, return_roi_tokens,
        )

    def _forward_single_subject(
        self,
        x: torch.Tensor,
        subject_id: str,
        return_roi_tokens: bool,
    ) -> Union[torch.Tensor, "ROITransformerOutput"]:
        B = x.size(0)
        tokens = self._project_rois_for_subject(x, subject_id)

        subj_int = self._subj_to_int[subject_id]
        subj_emb = self.subject_embedding(
            torch.tensor(subj_int, device=x.device)
        )
        tokens = tokens + subj_emb.unsqueeze(0).unsqueeze(0)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.pos_embed

        if return_roi_tokens:
            return self._forward_with_roi_output(tokens)

        out = self.transformer(tokens)
        cls_out = self.final_norm(out[:, 0])
        return cls_out

    def _forward_mixed_batch(
        self,
        x: torch.Tensor,
        subject_ids: List[int],
        return_roi_tokens: bool,
    ) -> Union[torch.Tensor, "ROITransformerOutput"]:
        """Handle a batch with samples from different subjects."""
        B = x.size(0)

        all_tokens = torch.zeros(
            B, self.n_rois, self.d_model,
            device=x.device, dtype=x.dtype,
        )

        for subj_int in set(subject_ids):
            mask = [i for i, s in enumerate(subject_ids) if s == subj_int]
            mask_t = torch.tensor(mask, device=x.device, dtype=torch.long)
            subj_x = x[mask_t]
            subj_id = self.subjects[subj_int]
            subj_tokens = self._project_rois_for_subject(subj_x, subj_id)

            subj_emb = self.subject_embedding(
                torch.tensor(subj_int, device=x.device)
            )
            subj_tokens = subj_tokens + subj_emb.unsqueeze(0).unsqueeze(0)

            all_tokens[mask_t] = subj_tokens

        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, all_tokens], dim=1)
        tokens = tokens + self.pos_embed

        if return_roi_tokens:
            return self._forward_with_roi_output(tokens)

        out = self.transformer(tokens)
        cls_out = self.final_norm(out[:, 0])
        return cls_out

    def _forward_with_roi_output(
        self,
        tokens: torch.Tensor,
    ) -> "ROITransformerOutput":
        """Run transformer with attention capture for ROI-DCF."""
        for i, layer in enumerate(self.transformer.layers):
            if i == len(self.transformer.layers) - 1:
                normed = layer.norm1(tokens) if hasattr(layer, "norm1") else tokens
                _, attn_weights = layer.self_attn(
                    normed, normed, normed,
                    need_weights=True,
                    average_attn_weights=True,
                )
                tokens = layer(tokens)
            else:
                tokens = layer(tokens)

        cls_out = self.final_norm(tokens[:, 0])
        roi_tokens = self.final_norm(tokens[:, 1:])

        cls_to_roi_alpha = attn_weights[:, 0, 1:]
        cls_to_roi_alpha = cls_to_roi_alpha / cls_to_roi_alpha.sum(
            dim=-1, keepdim=True,
        ).clamp(min=1e-8)

        return ROITransformerOutput(
            cls_out=cls_out,
            roi_tokens=roi_tokens,
            cls_to_roi_alpha=cls_to_roi_alpha,
        )
