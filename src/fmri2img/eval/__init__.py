"""fmri2img.eval

Evaluation helper modules.

Some metrics (e.g. LPIPS/SSIM image metrics) depend on optional heavy
dependencies such as `torchvision`. To keep lightweight imports working in
minimal environments (and to allow unit tests that don't need image metrics),
we only re-export the dependency-light retrieval utilities here.

Import image metrics directly from `fmri2img.eval.image_metrics` when needed.
"""

from __future__ import annotations

from .retrieval import cosine_sim, retrieval_at_k, compute_ranking_metrics
from .retrieval import clip_score as clip_score_embeddings
from .roi_kappa_extraction import (
    extract_per_roi_kappas,
    compute_roi_kappa_topography,
    compute_category_conditional_kappa,
    compute_per_roi_calibration,
    compute_kappa_ncsnr_dissociation,
    PerROIKappaResult,
    save_roi_kappa_results,
    load_roi_kappa_results,
    DEFAULT_ROI_NAMES,
    ROI_TIERS,
)

__all__ = [
    "cosine_sim",
    "retrieval_at_k",
    "compute_ranking_metrics",
    "clip_score_embeddings",
    # Per-ROI kappa extraction (uncertainty-decomposed decoding)
    "extract_per_roi_kappas",
    "compute_roi_kappa_topography",
    "compute_category_conditional_kappa",
    "compute_per_roi_calibration",
    "compute_kappa_ncsnr_dissociation",
    "PerROIKappaResult",
    "save_roi_kappa_results",
    "load_roi_kappa_results",
    "DEFAULT_ROI_NAMES",
    "ROI_TIERS",
]
