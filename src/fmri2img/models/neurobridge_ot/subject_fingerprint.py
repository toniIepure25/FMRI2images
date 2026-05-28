"""
Subject Fingerprint Module — Module B
======================================

Computes a compact subject fingerprint from calibration data and ROI
metadata. Used by the hyper-adapter network to generate subject-specific
modulation weights without requiring labeled target images.

For true zero-shot experiments, the fingerprint uses only unsupervised
statistics (voxel counts, ROI signal statistics, SNR). No CLIP targets
or image labels are included.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class SubjectFingerprint(nn.Module):
    """Compute a compact subject fingerprint vector.

    The fingerprint encodes subject-specific anatomical and functional
    characteristics that the hyper-adapter uses to generate modulation
    weights for unseen subjects.

    Features encoded:
        - Voxel count per ROI (normalized).
        - ROI mean/std activation over calibration trials.
        - ROI signal-to-noise ratio (tSNR).
        - Missing ROI indicators.
        - Optional: subject ID embedding for seen subjects.

    Args:
        n_rois: Number of ROI regions.
        fingerprint_dim: Output fingerprint dimensionality.
        n_subjects: Number of known subjects (for optional learned embedding).
        use_subject_embedding: Whether to include learnable subject embedding.
        roi_stat_dim: Dimensionality of per-ROI statistics input.
    """

    def __init__(
        self,
        n_rois: int = 17,
        fingerprint_dim: int = 128,
        n_subjects: int = 8,
        use_subject_embedding: bool = True,
        roi_stat_dim: int = 5,
    ):
        super().__init__()
        self.n_rois = n_rois
        self.fingerprint_dim = fingerprint_dim
        self.use_subject_embedding = use_subject_embedding

        # Per-ROI statistics: [voxel_count_norm, mean, std, snr, missing_flag]
        self.roi_stat_dim = roi_stat_dim
        input_dim = n_rois * roi_stat_dim

        if use_subject_embedding:
            self.subject_embed = nn.Embedding(n_subjects, fingerprint_dim // 2)
            mlp_out = fingerprint_dim // 2
        else:
            mlp_out = fingerprint_dim

        self.stat_encoder = nn.Sequential(
            nn.Linear(input_dim, fingerprint_dim * 2),
            nn.GELU(),
            nn.LayerNorm(fingerprint_dim * 2),
            nn.Linear(fingerprint_dim * 2, mlp_out),
            nn.GELU(),
            nn.LayerNorm(mlp_out),
        )

        if use_subject_embedding:
            self.combiner = nn.Sequential(
                nn.Linear(fingerprint_dim, fingerprint_dim),
                nn.GELU(),
                nn.LayerNorm(fingerprint_dim),
            )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "SubjectFingerprint: n_rois=%d, fp_dim=%d, use_subj_embed=%s, params=%d",
            n_rois, fingerprint_dim, use_subject_embedding, n_params,
        )

    def forward(
        self,
        roi_stats: Tensor,
        subject_id: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute subject fingerprint.

        Args:
            roi_stats: Per-ROI statistics tensor, shape (B, n_rois, roi_stat_dim).
                       Contains [voxel_count_norm, mean_activation, std_activation,
                       tSNR, missing_flag] per ROI.
            subject_id: Optional integer subject IDs, shape (B,).
                        Used only when use_subject_embedding=True and subject is known.

        Returns:
            Fingerprint vector, shape (B, fingerprint_dim).
        """
        B = roi_stats.shape[0]
        device = roi_stats.device

        flat_stats = roi_stats.reshape(B, -1)  # (B, n_rois * roi_stat_dim)
        stat_features = self.stat_encoder(flat_stats)  # (B, mlp_out)

        if self.use_subject_embedding and subject_id is not None:
            sid = subject_id.long().to(device)
            subj_emb = self.subject_embed(sid)  # (B, fingerprint_dim // 2)
            combined = torch.cat([stat_features, subj_emb], dim=-1)
            fingerprint = self.combiner(combined)
        else:
            fingerprint = stat_features

        return fingerprint

    @staticmethod
    def compute_roi_stats(
        fmri_data: Tensor,
        roi_indices: Dict[str, Tensor],
        n_rois: int = 17,
        max_calibration_trials: int = 100,
    ) -> Tensor:
        """Compute per-ROI statistics from calibration fMRI data.

        This is typically called once per subject during data loading to
        precompute the fingerprint inputs.

        Args:
            fmri_data: Calibration fMRI trials, shape (N_cal, V).
            roi_indices: Dict mapping ROI names to voxel index tensors.
            n_rois: Number of ROIs expected.
            max_calibration_trials: Max trials to use for statistics.

        Returns:
            roi_stats: Shape (1, n_rois, 5) containing
                       [voxel_count_norm, mean, std, tSNR, missing_flag].
        """
        device = fmri_data.device
        if fmri_data.shape[0] > max_calibration_trials:
            fmri_data = fmri_data[:max_calibration_trials]

        stats = torch.zeros(1, n_rois, 5, device=device)
        max_voxels = max(len(idx) for idx in roi_indices.values() if idx is not None and len(idx) > 0)
        max_voxels = max(max_voxels, 1)

        for i, (roi_name, idx) in enumerate(roi_indices.items()):
            if i >= n_rois:
                break
            if idx is None or len(idx) == 0:
                stats[0, i, 4] = 1.0  # missing flag
                continue
            idx_dev = idx.to(device)
            roi_vox = fmri_data[:, idx_dev]  # (N_cal, n_vox)
            stats[0, i, 0] = len(idx) / max_voxels  # normalized voxel count
            stats[0, i, 1] = roi_vox.mean()
            stats[0, i, 2] = roi_vox.std().clamp(min=1e-6)
            temporal_mean = roi_vox.mean(dim=0)
            temporal_std = roi_vox.std(dim=0).clamp(min=1e-6)
            tsnr = (temporal_mean / temporal_std).mean()
            stats[0, i, 3] = tsnr.clamp(min=0, max=100) / 100.0  # normalized tSNR

        return stats
