"""
Per-ROI Kappa Extraction and Analysis for ROI-DCF Models
=========================================================

Extends evaluation to extract, store, and analyze the full per-ROI kappa
tensor (B, n_rois) for every trial — enabling topographic analysis of
regional encoding fidelity.

Core outputs:
    - per_roi_kappas:  (N_trials, n_rois) array of per-ROI concentrations
    - per_roi_mus:     (N_trials, n_rois, D) per-ROI direction vectors
    - consensus_kappa: (N_trials,) fused concentration
    - delta:           (N_trials,) inter-ROI disagreement
    - alphas:          (N_trials, n_rois) attention mixing weights

Analysis functions:
    - ROI kappa topography: mean kappa per ROI across trials
    - Category-conditional kappa: kappa_r conditioned on stimulus category
    - Kappa-NCSNR dissociation: partial correlation analysis
    - Per-ROI calibration: does higher kappa_r correlate with better
      per-ROI mu_r alignment to ground truth?

References:
    - Plan: "Implement per-ROI kappa extraction during evaluation"
    - Paper 2 contribution: "Per-ROI Encoding Fidelity Map"
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# Canonical ROI ordering (17 regions matching ROITransformerEncoder)
DEFAULT_ROI_NAMES: List[str] = [
    "V1v", "V1d", "V2v", "V2d", "V3v", "V3d",
    "V3A", "V3B", "V4",
    "FFA1", "FFA2", "PPA", "EBA", "OFA", "OPA",
    "RSC", "nsdgeneral_other",
]

# Functional tiers for grouped analysis
ROI_TIERS: Dict[str, List[str]] = {
    "early_visual": ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d"],
    "mid_visual": ["V3A", "V3B", "V4"],
    "face_selective": ["FFA1", "FFA2", "OFA"],
    "scene_selective": ["PPA", "OPA", "RSC"],
    "body_selective": ["EBA"],
    "residual": ["nsdgeneral_other"],
}


@dataclass
class PerROIKappaResult:
    """Container for per-ROI kappa extraction results."""

    per_roi_kappas: np.ndarray       # (N, n_rois)
    consensus_kappa: np.ndarray      # (N,)
    delta: np.ndarray                # (N,)
    alphas: np.ndarray               # (N, n_rois)
    per_roi_mus: Optional[np.ndarray] = None  # (N, n_rois, D) — large, optional
    mu_fused: Optional[np.ndarray] = None     # (N, D)
    roi_names: List[str] = field(default_factory=lambda: list(DEFAULT_ROI_NAMES))
    trial_ids: Optional[np.ndarray] = None    # (N,) nsd_ids or trial indices
    n_trials: int = 0
    n_rois: int = 17

    def __post_init__(self):
        self.n_trials = len(self.per_roi_kappas)
        self.n_rois = self.per_roi_kappas.shape[1] if self.n_trials > 0 else 0


@torch.no_grad()
def extract_per_roi_kappas(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    store_mus: bool = False,
    max_batches: Optional[int] = None,
) -> PerROIKappaResult:
    """
    Run inference on a dataset and extract full per-ROI kappa tensors.

    Requires the model to be a UnifiedModel with model_type='vmf_dcf'
    and _return_per_roi=True (default for vmf_dcf).

    Parameters
    ----------
    model : UnifiedModel with vmf_dcf decoder
    dataloader : yields (fmri_batch, targets, ...) or (fmri_batch, targets)
    device : compute device
    store_mus : if True, also store per-ROI mu vectors (memory-intensive)
    max_batches : limit number of batches (for debugging)

    Returns
    -------
    PerROIKappaResult with all per-trial, per-ROI kappa values
    """
    model.eval()
    model.to(device)

    all_per_roi_kappas = []
    all_consensus_kappa = []
    all_delta = []
    all_alphas = []
    all_per_roi_mus = [] if store_mus else None
    all_mu_fused = []

    roi_names = getattr(model.encoder, "roi_names", list(DEFAULT_ROI_NAMES))

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        if isinstance(batch, (list, tuple)):
            fmri = batch[0]
        else:
            fmri = batch

        fmri = fmri.to(device).float()

        mu_fused, kappa_consensus = model(fmri)

        extras = model._last_dcf_extras
        per_roi_kappas = extras["per_roi_kappas"]  # (B, R, 1)
        delta = extras["delta"]                     # (B, 1)
        alphas = extras["cls_to_roi_alpha"]         # (B, R)

        all_per_roi_kappas.append(per_roi_kappas.squeeze(-1).cpu().numpy())
        all_consensus_kappa.append(kappa_consensus.squeeze(-1).cpu().numpy())
        all_delta.append(delta.squeeze(-1).cpu().numpy())
        all_alphas.append(alphas.cpu().numpy())
        all_mu_fused.append(mu_fused.cpu().numpy())

        if store_mus:
            per_roi_mus = extras["per_roi_mus"]  # (B, R, D)
            all_per_roi_mus.append(per_roi_mus.cpu().numpy())

    result = PerROIKappaResult(
        per_roi_kappas=np.concatenate(all_per_roi_kappas, axis=0),
        consensus_kappa=np.concatenate(all_consensus_kappa, axis=0),
        delta=np.concatenate(all_delta, axis=0),
        alphas=np.concatenate(all_alphas, axis=0),
        per_roi_mus=np.concatenate(all_per_roi_mus, axis=0) if store_mus else None,
        mu_fused=np.concatenate(all_mu_fused, axis=0),
        roi_names=roi_names,
    )

    logger.info(
        "Extracted per-ROI kappas: %d trials, %d ROIs. "
        "Consensus kappa: mean=%.2f, std=%.2f. Delta: mean=%.4f",
        result.n_trials, result.n_rois,
        result.consensus_kappa.mean(), result.consensus_kappa.std(),
        result.delta.mean(),
    )

    return result


def compute_roi_kappa_topography(
    result: PerROIKappaResult,
) -> Dict[str, Any]:
    """
    Compute mean kappa per ROI — the encoding fidelity topography.

    Returns a dict with per-ROI statistics and tier-level aggregates.
    """
    per_roi_mean = result.per_roi_kappas.mean(axis=0)  # (n_rois,)
    per_roi_std = result.per_roi_kappas.std(axis=0)
    per_roi_median = np.median(result.per_roi_kappas, axis=0)

    roi_stats = {}
    for i, name in enumerate(result.roi_names):
        roi_stats[name] = {
            "mean_kappa": float(per_roi_mean[i]),
            "std_kappa": float(per_roi_std[i]),
            "median_kappa": float(per_roi_median[i]),
            "mean_alpha": float(result.alphas[:, i].mean()),
        }

    tier_stats = {}
    for tier_name, tier_rois in ROI_TIERS.items():
        roi_indices = [
            i for i, name in enumerate(result.roi_names) if name in tier_rois
        ]
        if roi_indices:
            tier_kappas = result.per_roi_kappas[:, roi_indices]
            tier_stats[tier_name] = {
                "mean_kappa": float(tier_kappas.mean()),
                "std_kappa": float(tier_kappas.std()),
                "n_rois": len(roi_indices),
            }

    ranked_rois = sorted(roi_stats.items(), key=lambda x: x[1]["mean_kappa"], reverse=True)

    report = {
        "roi_stats": roi_stats,
        "tier_stats": tier_stats,
        "ranked_rois": [(name, stats["mean_kappa"]) for name, stats in ranked_rois],
        "global_stats": {
            "mean_kappa_all_rois": float(per_roi_mean.mean()),
            "max_roi": ranked_rois[0][0],
            "min_roi": ranked_rois[-1][0],
            "kappa_range": float(per_roi_mean.max() - per_roi_mean.min()),
        },
    }

    logger.info(
        "ROI kappa topography: highest=%s (%.2f), lowest=%s (%.2f), range=%.2f",
        report["global_stats"]["max_roi"], per_roi_mean.max(),
        report["global_stats"]["min_roi"], per_roi_mean.min(),
        report["global_stats"]["kappa_range"],
    )

    return report


def compute_category_conditional_kappa(
    result: PerROIKappaResult,
    category_labels: np.ndarray,
    category_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute per-ROI kappa conditioned on stimulus category.

    This reveals the functional specialization of encoding fidelity:
    e.g., FFA has high kappa for faces, PPA for scenes.

    Parameters
    ----------
    result : PerROIKappaResult from extract_per_roi_kappas
    category_labels : (N,) integer category labels per trial
    category_names : optional list mapping label indices to names

    Returns
    -------
    Dict with shape (n_categories, n_rois) kappa matrix and statistics
    """
    unique_cats = np.unique(category_labels)
    n_cats = len(unique_cats)
    n_rois = result.n_rois

    kappa_matrix = np.zeros((n_cats, n_rois))
    kappa_std_matrix = np.zeros((n_cats, n_rois))
    counts = np.zeros(n_cats, dtype=int)

    for ci, cat in enumerate(unique_cats):
        mask = category_labels == cat
        counts[ci] = mask.sum()
        kappa_matrix[ci] = result.per_roi_kappas[mask].mean(axis=0)
        kappa_std_matrix[ci] = result.per_roi_kappas[mask].std(axis=0)

    if category_names is None:
        category_names = [f"cat_{c}" for c in unique_cats]

    # Find per-ROI "preferred category" (highest mean kappa)
    preferred_category = {}
    for roi_idx, roi_name in enumerate(result.roi_names):
        best_cat_idx = int(kappa_matrix[:, roi_idx].argmax())
        preferred_category[roi_name] = {
            "category": category_names[best_cat_idx],
            "kappa": float(kappa_matrix[best_cat_idx, roi_idx]),
            "selectivity": float(
                kappa_matrix[best_cat_idx, roi_idx] - kappa_matrix[:, roi_idx].mean()
            ),
        }

    # F-statistic: does kappa vary significantly across categories for each ROI?
    f_stats = {}
    for roi_idx, roi_name in enumerate(result.roi_names):
        groups = [
            result.per_roi_kappas[category_labels == cat, roi_idx]
            for cat in unique_cats
            if (category_labels == cat).sum() > 1
        ]
        if len(groups) >= 2:
            f_val, p_val = sp_stats.f_oneway(*groups)
            f_stats[roi_name] = {"F": float(f_val), "p": float(p_val)}

    report = {
        "kappa_matrix": kappa_matrix.tolist(),
        "kappa_std_matrix": kappa_std_matrix.tolist(),
        "category_names": category_names,
        "roi_names": result.roi_names,
        "counts_per_category": counts.tolist(),
        "preferred_category": preferred_category,
        "category_selectivity_f_tests": f_stats,
        "n_categories": n_cats,
        "n_rois": n_rois,
    }

    n_significant = sum(1 for v in f_stats.values() if v["p"] < 0.05)
    logger.info(
        "Category-conditional kappa: %d categories x %d ROIs. "
        "%d/%d ROIs show significant category modulation (p<0.05).",
        n_cats, n_rois, n_significant, n_rois,
    )

    return report


def compute_per_roi_calibration(
    result: PerROIKappaResult,
    ground_truth_embeddings: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Per-ROI calibration: does higher kappa_r predict better mu_r-to-GT alignment?

    For each ROI r, compute Spearman(kappa_r, cos(mu_r, GT)) across trials.
    A well-calibrated ROI head should show positive correlation.

    Parameters
    ----------
    result : PerROIKappaResult (must have per_roi_mus stored)
    ground_truth_embeddings : (N, D) CLIP ground truth embeddings
    n_bins : number of bins for reliability diagram

    Returns
    -------
    Dict with per-ROI calibration statistics
    """
    if result.per_roi_mus is None:
        raise ValueError("per_roi_mus not stored. Re-run extraction with store_mus=True.")

    gt_norm = ground_truth_embeddings / np.maximum(
        np.linalg.norm(ground_truth_embeddings, axis=-1, keepdims=True), 1e-8
    )

    calibration = {}
    for roi_idx, roi_name in enumerate(result.roi_names):
        mu_r = result.per_roi_mus[:, roi_idx]  # (N, D)
        kappa_r = result.per_roi_kappas[:, roi_idx]  # (N,)

        cosine_r = np.sum(mu_r * gt_norm, axis=-1)  # (N,)

        sp = sp_stats.spearmanr(kappa_r, cosine_r)

        calibration[roi_name] = {
            "spearman_rho": float(sp.statistic),
            "spearman_p": float(sp.pvalue),
            "mean_cosine": float(cosine_r.mean()),
            "mean_kappa": float(kappa_r.mean()),
            "is_calibrated": sp.statistic > 0 and sp.pvalue < 0.05,
        }

    # Also calibrate the fused output
    if result.mu_fused is not None:
        fused_cosine = np.sum(
            (result.mu_fused / np.maximum(
                np.linalg.norm(result.mu_fused, axis=-1, keepdims=True), 1e-8
            )) * gt_norm,
            axis=-1,
        )
        sp_fused = sp_stats.spearmanr(result.consensus_kappa, fused_cosine)
        calibration["_fused"] = {
            "spearman_rho": float(sp_fused.statistic),
            "spearman_p": float(sp_fused.pvalue),
            "mean_cosine": float(fused_cosine.mean()),
            "mean_kappa": float(result.consensus_kappa.mean()),
            "is_calibrated": sp_fused.statistic > 0 and sp_fused.pvalue < 0.05,
        }

    n_calibrated = sum(
        1 for k, v in calibration.items()
        if k != "_fused" and v["is_calibrated"]
    )
    logger.info(
        "Per-ROI calibration: %d/%d ROIs are well-calibrated "
        "(positive Spearman kappa-cosine, p<0.05). "
        "Fused calibration rho=%.3f.",
        n_calibrated, result.n_rois,
        calibration.get("_fused", {}).get("spearman_rho", 0.0),
    )

    return calibration


def compute_kappa_ncsnr_dissociation(
    result: PerROIKappaResult,
    ncsnr_per_roi: np.ndarray,
) -> Dict[str, Any]:
    """
    Test whether per-ROI kappa captures information beyond noise ceiling (NCSNR).

    Computes:
    1. Pearson(mean_kappa_r, ncsnr_r) across ROIs -- expected positive
    2. Partial correlation controlling for NCSNR
    3. Unique variance explained by kappa beyond NCSNR

    Parameters
    ----------
    result : PerROIKappaResult
    ncsnr_per_roi : (n_rois,) mean NCSNR per ROI (from reliability module)

    Returns
    -------
    Dict with dissociation statistics
    """
    mean_kappa_per_roi = result.per_roi_kappas.mean(axis=0)  # (n_rois,)

    # Correlation between mean kappa and NCSNR across ROIs
    r_kappa_ncsnr, p_kappa_ncsnr = sp_stats.pearsonr(mean_kappa_per_roi, ncsnr_per_roi)

    # Spearman (rank-based, more robust)
    sp_kappa_ncsnr = sp_stats.spearmanr(mean_kappa_per_roi, ncsnr_per_roi)

    # Residual after regressing out NCSNR
    from numpy.polynomial.polynomial import polyfit, polyval
    coeffs = polyfit(ncsnr_per_roi, mean_kappa_per_roi, deg=1)
    predicted_kappa = polyval(ncsnr_per_roi, coeffs)
    residual_kappa = mean_kappa_per_roi - predicted_kappa
    residual_variance = float(np.var(residual_kappa) / np.var(mean_kappa_per_roi))

    report = {
        "pearson_kappa_ncsnr": {"r": float(r_kappa_ncsnr), "p": float(p_kappa_ncsnr)},
        "spearman_kappa_ncsnr": {
            "rho": float(sp_kappa_ncsnr.statistic),
            "p": float(sp_kappa_ncsnr.pvalue),
        },
        "residual_variance_fraction": residual_variance,
        "interpretation": (
            "kappa captures information beyond NCSNR"
            if residual_variance > 0.1
            else "kappa largely tracks NCSNR"
        ),
        "per_roi": {
            name: {
                "mean_kappa": float(mean_kappa_per_roi[i]),
                "ncsnr": float(ncsnr_per_roi[i]),
                "residual": float(residual_kappa[i]),
            }
            for i, name in enumerate(result.roi_names)
            if i < len(ncsnr_per_roi)
        },
    }

    logger.info(
        "Kappa-NCSNR dissociation: Pearson r=%.3f (p=%.4f), "
        "residual variance=%.1f%% -> %s",
        r_kappa_ncsnr, p_kappa_ncsnr,
        residual_variance * 100,
        report["interpretation"],
    )

    return report


def save_roi_kappa_results(
    result: PerROIKappaResult,
    output_dir: Path,
    prefix: str = "roi_kappa",
) -> Dict[str, Path]:
    """Save per-ROI kappa extraction results to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    kappas_path = output_dir / f"{prefix}_per_roi_kappas.npy"
    np.save(kappas_path, result.per_roi_kappas)
    paths["per_roi_kappas"] = kappas_path

    consensus_path = output_dir / f"{prefix}_consensus_kappa.npy"
    np.save(consensus_path, result.consensus_kappa)
    paths["consensus_kappa"] = consensus_path

    delta_path = output_dir / f"{prefix}_delta.npy"
    np.save(delta_path, result.delta)
    paths["delta"] = delta_path

    alphas_path = output_dir / f"{prefix}_alphas.npy"
    np.save(alphas_path, result.alphas)
    paths["alphas"] = alphas_path

    if result.per_roi_mus is not None:
        mus_path = output_dir / f"{prefix}_per_roi_mus.npy"
        np.save(mus_path, result.per_roi_mus)
        paths["per_roi_mus"] = mus_path

    if result.mu_fused is not None:
        fused_path = output_dir / f"{prefix}_mu_fused.npy"
        np.save(fused_path, result.mu_fused)
        paths["mu_fused"] = fused_path

    if result.trial_ids is not None:
        ids_path = output_dir / f"{prefix}_trial_ids.npy"
        np.save(ids_path, result.trial_ids)
        paths["trial_ids"] = ids_path

    # Save metadata
    import json
    meta = {
        "n_trials": result.n_trials,
        "n_rois": result.n_rois,
        "roi_names": result.roi_names,
        "has_per_roi_mus": result.per_roi_mus is not None,
        "has_mu_fused": result.mu_fused is not None,
        "files": {k: str(v) for k, v in paths.items()},
    }
    meta_path = output_dir / f"{prefix}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    paths["meta"] = meta_path

    logger.info("Saved per-ROI kappa results to %s (%d files)", output_dir, len(paths))
    return paths


def load_roi_kappa_results(
    output_dir: Path,
    prefix: str = "roi_kappa",
) -> PerROIKappaResult:
    """Load previously saved per-ROI kappa results."""
    import json
    output_dir = Path(output_dir)

    meta_path = output_dir / f"{prefix}_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    per_roi_kappas = np.load(output_dir / f"{prefix}_per_roi_kappas.npy")
    consensus_kappa = np.load(output_dir / f"{prefix}_consensus_kappa.npy")
    delta = np.load(output_dir / f"{prefix}_delta.npy")
    alphas = np.load(output_dir / f"{prefix}_alphas.npy")

    per_roi_mus = None
    if meta.get("has_per_roi_mus"):
        mus_path = output_dir / f"{prefix}_per_roi_mus.npy"
        if mus_path.exists():
            per_roi_mus = np.load(mus_path)

    mu_fused = None
    if meta.get("has_mu_fused"):
        fused_path = output_dir / f"{prefix}_mu_fused.npy"
        if fused_path.exists():
            mu_fused = np.load(fused_path)

    trial_ids = None
    ids_path = output_dir / f"{prefix}_trial_ids.npy"
    if ids_path.exists():
        trial_ids = np.load(ids_path)

    return PerROIKappaResult(
        per_roi_kappas=per_roi_kappas,
        consensus_kappa=consensus_kappa,
        delta=delta,
        alphas=alphas,
        per_roi_mus=per_roi_mus,
        mu_fused=mu_fused,
        roi_names=meta.get("roi_names", list(DEFAULT_ROI_NAMES)),
        trial_ids=trial_ids,
    )
