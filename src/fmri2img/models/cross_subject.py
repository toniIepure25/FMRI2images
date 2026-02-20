"""
Cross-Subject Functional Alignment for Brain Decoding
=====================================================

Implements functional alignment strategies to enable cross-subject
generalization in fMRI-to-embedding decoding.

Approach 1 — Ridge Alignment (MindEye2-style):
    Per-subject ridge regression maps brain activations to a shared
    latent space.  Train on 7 subjects, fine-tune on the target subject
    with minimal data.

Approach 2 — Shared Backbone with Subject Adapters:
    A shared ROI-DCF backbone processes all subjects.  Lightweight
    per-subject adapter layers handle anatomical variability.

Approach 3 — Procrustes Alignment:
    Learn orthogonal transformations that align each subject's
    embedding predictions to a shared space.

References:
    - Scotti et al. (2024) MindEye2: Shared-Subject Models
    - Haxby et al. (2020) Hyperalignment
    - Brain-IT (2025) Cross-subject brain tokenization
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RidgeAligner(nn.Module):
    """
    Ridge regression alignment to a shared latent space.

    Maps subject-specific fMRI activations through a regularized
    linear projection before feeding into the shared decoder.

    Args:
        input_dim: Number of voxels for this subject.
        output_dim: Shared latent dimension.
        alpha: Ridge regularization strength.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.projection = nn.Linear(input_dim, output_dim, bias=True)
        self._fitted = False

        logger.info(
            "RidgeAligner: %d -> %d (alpha=%.4f)",
            input_dim, output_dim, alpha,
        )

    def fit(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> None:
        """
        Fit ridge regression weights analytically.

        W = (X^T X + alpha I)^{-1} X^T Y

        Args:
            X: (N, input_dim) fMRI activations.
            Y: (N, output_dim) target shared representations.
        """
        device = X.device
        I = torch.eye(self.input_dim, device=device) * self.alpha
        XtX = X.T @ X + I
        XtY = X.T @ Y
        W = torch.linalg.solve(XtX, XtY)

        bias = Y.mean(dim=0) - X.mean(dim=0) @ W

        with torch.no_grad():
            self.projection.weight.copy_(W.T)
            self.projection.bias.copy_(bias)

        self._fitted = True
        logger.info("RidgeAligner fitted on %d samples", X.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class SubjectAdapter(nn.Module):
    """
    Lightweight per-subject adapter layer.

    Adds a small residual transformation after the shared encoder
    to handle subject-specific anatomical variations.

    Architecture: x -> LayerNorm -> Linear -> GELU -> Linear -> x + residual

    Args:
        d_model: Feature dimension (same as shared encoder output).
        bottleneck_dim: Internal bottleneck dimension (typically d_model // 4).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = max(d_model // 4, 32)

        self.adapter = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, d_model),
            nn.Dropout(dropout),
        )

        # Initialize near-identity
        nn.init.zeros_(self.adapter[-2].weight)
        nn.init.zeros_(self.adapter[-2].bias)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "SubjectAdapter: d_model=%d, bottleneck=%d, params=%s",
            d_model, bottleneck_dim, f"{n_params:,}",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapter(x)


class CrossSubjectModel(nn.Module):
    """
    Shared backbone with per-subject alignment for cross-subject decoding.

    Architecture:
        fMRI_subj -> RidgeAligner_subj -> SharedEncoder -> SubjectAdapter_subj -> Decoder

    The shared encoder and decoder are frozen during subject-specific
    fine-tuning; only the RidgeAligner and SubjectAdapter are trained.

    Args:
        shared_encoder: Pretrained shared encoder (e.g., ROITransformerEncoder).
        shared_decoder: Pretrained shared decoder (e.g., ROIDCFDecoder).
        subject_configs: Dict mapping subject_id -> {"input_dim": int, ...}.
        shared_dim: Dimension of the shared latent space.
        ridge_alpha: Ridge regularization for aligners.
        adapter_bottleneck: Bottleneck dim for adapters.
    """

    def __init__(
        self,
        shared_encoder: nn.Module,
        shared_decoder: nn.Module,
        subject_configs: Dict[str, Dict],
        shared_dim: int,
        ridge_alpha: float = 1.0,
        adapter_bottleneck: Optional[int] = None,
    ):
        super().__init__()
        self.shared_encoder = shared_encoder
        self.shared_decoder = shared_decoder
        self.shared_dim = shared_dim

        # Per-subject alignment layers
        self.aligners = nn.ModuleDict()
        self.adapters = nn.ModuleDict()

        for subj_id, cfg in subject_configs.items():
            self.aligners[subj_id] = RidgeAligner(
                input_dim=cfg["input_dim"],
                output_dim=shared_dim,
                alpha=ridge_alpha,
            )
            self.adapters[subj_id] = SubjectAdapter(
                d_model=shared_dim,
                bottleneck_dim=adapter_bottleneck,
            )

        logger.info(
            "CrossSubjectModel: %d subjects, shared_dim=%d",
            len(subject_configs), shared_dim,
        )

    def freeze_shared(self) -> None:
        """Freeze shared encoder and decoder for subject-specific fine-tuning."""
        for param in self.shared_encoder.parameters():
            param.requires_grad = False
        for param in self.shared_decoder.parameters():
            param.requires_grad = False
        logger.info("Shared encoder and decoder frozen")

    def unfreeze_shared(self) -> None:
        """Unfreeze shared components for joint training."""
        for param in self.shared_encoder.parameters():
            param.requires_grad = True
        for param in self.shared_decoder.parameters():
            param.requires_grad = True

    def trainable_params(self, subject_id: str) -> List[nn.Parameter]:
        """Get trainable parameters for a specific subject."""
        params = []
        if subject_id in self.aligners:
            params.extend(self.aligners[subject_id].parameters())
        if subject_id in self.adapters:
            params.extend(self.adapters[subject_id].parameters())
        return params

    def forward(
        self,
        x: torch.Tensor,
        subject_id: str,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for a specific subject.

        Args:
            x: (B, V_subj) fMRI activations for subject.
            subject_id: Subject identifier (e.g., "subj01").

        Returns:
            Same as UnifiedModel.forward(): (prediction, uncertainty)
        """
        aligned = self.aligners[subject_id](x)
        h = self.shared_encoder(aligned)

        if isinstance(h, torch.Tensor):
            h = self.adapters[subject_id](h)
        else:
            h.cls_out = self.adapters[subject_id](h.cls_out)

        return self.shared_decoder(h)


def procrustes_align(
    source: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orthogonal Procrustes alignment between two embedding spaces.

    Find R = argmin_R ||source @ R - target||_F  s.t.  R^T R = I

    This aligns the source subject's predictions to the target space
    using an orthogonal transformation that preserves geometry.

    Args:
        source: (N, D) embeddings from source subject.
        target: (N, D) embeddings from target subject (same stimuli).

    Returns:
        R: (D, D) orthogonal rotation matrix.
        aligned: (N, D) aligned source embeddings.
    """
    # Center
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)

    # Use scipy's orthogonal_procrustes for numerical reliability
    from scipy.linalg import orthogonal_procrustes as _ortho_proc
    R, _ = _ortho_proc(source_centered, target_centered)

    aligned = source_centered @ R + target.mean(axis=0)
    return R, aligned
