"""Representational Similarity Analysis (RSA).

Scientific claim tested:
    "Decoded fMRI embeddings preserve representational geometry of CLIP
     image space beyond top-1 retrieval."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats as sp_stats

from analysis.manifold.figures import (
    create_heatmap,
    create_scatter,
    save_figure,
    setup_figure_style,
)
from analysis.manifold.utils import bootstrap_ci, cosine_similarity_matrix, safe_normalize, upper_triangle

logger = logging.getLogger(__name__)


def compute_rdm(embeddings: np.ndarray) -> np.ndarray:
    """Representational Dissimilarity Matrix: 1 - cosine similarity."""
    embeddings = safe_normalize(embeddings)
    sim = cosine_similarity_matrix(embeddings, embeddings)
    return 1.0 - sim


def compute_rsa(
    z_pred: np.ndarray,
    z_target: np.ndarray,
    max_samples: Optional[int] = None,
    seed: int = 42,
    methods: Sequence[str] = ("spearman", "pearson"),
    n_bootstrap: int = 0,
) -> Dict[str, Any]:
    """RSA between predicted and target embedding spaces.

    Args:
        z_pred: (N, D) predicted embeddings.
        z_target: (N, D) target embeddings.
        max_samples: Subsample to this many items to avoid O(N^2) memory.
        seed: Random seed for subsampling.
        methods: Correlation methods to compute.

    Returns:
        Dict with RSA results per method plus the two RDMs (subsampled).
    """
    n = z_pred.shape[0]
    if max_samples is not None and n > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, max_samples, replace=False)
        z_pred = z_pred[idx]
        z_target = z_target[idx]
        logger.info("RSA subsampled from %d to %d", n, max_samples)

    rdm_pred = compute_rdm(z_pred)
    rdm_target = compute_rdm(z_target)

    ut_pred = upper_triangle(rdm_pred)
    ut_target = upper_triangle(rdm_target)
    abs_err = np.abs(ut_pred - ut_target)

    results: Dict[str, Any] = {"n_samples": z_pred.shape[0]}
    for method in methods:
        if method == "spearman":
            rho, p = sp_stats.spearmanr(ut_pred, ut_target)
        elif method == "pearson":
            rho, p = sp_stats.pearsonr(ut_pred, ut_target)
        elif method == "kendall":
            rho, p = sp_stats.kendalltau(ut_pred, ut_target)
        else:
            continue
        results[f"{method}_rho"] = float(rho)
        results[f"{method}_p"] = float(p)
        if method == "spearman" and n_bootstrap > 0:
            paired = np.column_stack([ut_pred, ut_target])

            def _spearman(sample: np.ndarray) -> float:
                res = sp_stats.spearmanr(sample[:, 0], sample[:, 1])
                return float(res.statistic if hasattr(res, "statistic") else res[0])

            lo, hi = bootstrap_ci(paired, _spearman, n_bootstrap=n_bootstrap, seed=seed)
            results["spearman_ci_low"] = lo
            results["spearman_ci_high"] = hi

    results["mean_absolute_rdm_error"] = float(np.mean(abs_err))
    results["median_absolute_rdm_error"] = float(np.median(abs_err))
    results["std_absolute_rdm_error"] = float(np.std(abs_err))
    results["percentile_95_absolute_rdm_error"] = float(np.percentile(abs_err, 95))

    results["rdm_pred"] = rdm_pred
    results["rdm_target"] = rdm_target
    results["ut_pred"] = ut_pred
    results["ut_target"] = ut_target
    return results


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

def plot_rdm_heatmap(
    rdm: np.ndarray,
    title: str,
    output_path: Path,
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    fig = create_heatmap(rdm, title=title, cmap="viridis", vmin=0, vmax=2)
    return save_figure(fig, output_path, **save_kw)


def plot_rsa_difference(
    rdm_pred: np.ndarray,
    rdm_target: np.ndarray,
    output_path: Path,
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    diff = np.abs(rdm_pred - rdm_target)
    off_diag = diff[np.triu_indices_from(diff, k=1)]
    vmax = float(np.percentile(off_diag, 95)) if off_diag.size else None
    fig = create_heatmap(
        diff,
        title="RDM Absolute Difference (95th-percentile scale)",
        xlabel="Sample index",
        ylabel="Sample index",
        cmap="Reds",
        vmin=0,
        vmax=vmax,
    )
    return save_figure(fig, output_path, **save_kw)


def plot_rsa_scatter(
    ut_pred: np.ndarray,
    ut_target: np.ndarray,
    output_path: Path,
    subsample: int = 50000,
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    if len(ut_pred) > subsample:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(ut_pred), subsample, replace=False)
        ut_pred = ut_pred[idx]
        ut_target = ut_target[idx]
    fig = create_scatter(
        ut_target, ut_pred,
        title="RSA: Pairwise Similarity Scatter",
        xlabel="Target (CLIP) dissimilarity",
        ylabel="Predicted dissimilarity",
    )
    return save_figure(fig, output_path, **save_kw)


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------

def save_rsa_results(
    results: Dict[str, Any],
    output_dir: Path,
    save_formats: Sequence[str] = ("png",),
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {k: v for k, v in results.items()
               if k not in ("rdm_pred", "rdm_target", "ut_pred", "ut_target")}
    with open(output_dir / "rsa_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    figs_dir = output_dir / "figures"
    figs: List[Path] = []

    n = results["rdm_pred"].shape[0]
    max_heatmap = 500
    rdm_p = results["rdm_pred"][:max_heatmap, :max_heatmap]
    rdm_t = results["rdm_target"][:max_heatmap, :max_heatmap]

    figs += plot_rdm_heatmap(rdm_t, "CLIP Target RDM",
                             figs_dir / "figure_rsa_clip_matrix", formats=save_formats)
    figs += plot_rdm_heatmap(rdm_p, "Predicted RDM",
                             figs_dir / "figure_rsa_pred_matrix", formats=save_formats)
    figs += plot_rsa_difference(rdm_p, rdm_t,
                                figs_dir / "figure_rsa_difference_matrix", formats=save_formats)
    figs += plot_rsa_scatter(results["ut_pred"], results["ut_target"],
                             figs_dir / "figure_rsa_similarity_scatter", formats=save_formats)

    logger.info("RSA results saved to %s", output_dir)
    return {"metrics": metrics, "figures": [str(f) for f in figs]}
