"""Baseline retrieval evaluation: cosine and CSLS retrieval metrics.

Scientific claim tested:
    "CSLS improves neural-to-image retrieval compared to raw cosine retrieval."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from analysis.manifold.figures import (
    create_grouped_bar,
    create_histogram,
    save_figure,
    setup_figure_style,
)
from analysis.manifold.utils import (
    bootstrap_ci,
    cosine_similarity_matrix,
    csls_scores,
    safe_normalize,
    topk_entropy,
)

logger = logging.getLogger(__name__)


def _ranks_from_scores(
    scores: np.ndarray, target_indices: np.ndarray
) -> np.ndarray:
    """Per-query rank (0-based) of the target in *scores* (descending)."""
    n = scores.shape[0]
    sorted_idx = np.argsort(-scores, axis=1)
    ranks = np.zeros(n, dtype=np.int64)
    for i in range(n):
        (pos,) = np.where(sorted_idx[i] == target_indices[i])
        ranks[i] = pos[0] if len(pos) > 0 else scores.shape[1]
    return ranks


def _metrics_from_ranks(
    ranks: np.ndarray,
    ks: Sequence[int],
    n_bootstrap: int = 0,
    seed: int = 42,
) -> Dict[str, float]:
    """Aggregate retrieval metrics from rank array."""
    n = len(ranks)
    metrics: Dict[str, float] = {}
    for k in ks:
        hit = ranks < k
        metrics[f"R@{k}"] = float(np.mean(hit))
        if n_bootstrap > 0:
            lo, hi = bootstrap_ci(hit.astype(np.float64), np.mean, n_bootstrap, seed)
            metrics[f"R@{k}_ci_low"] = lo
            metrics[f"R@{k}_ci_high"] = hi
    metrics["median_rank"] = float(np.median(ranks + 1))
    metrics["mean_rank"] = float(np.mean(ranks + 1))
    metrics["mrr"] = float(np.mean(1.0 / (ranks + 1)))
    return metrics


def compute_cosine_retrieval(
    z_pred: np.ndarray,
    gallery: np.ndarray,
    target_indices: np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
    n_bootstrap: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Full cosine retrieval evaluation.

    Args:
        z_pred: (N, D) query embeddings (L2-normalised).
        gallery: (G, D) gallery embeddings (L2-normalised).
        target_indices: (N,) index of correct gallery item per query.
        ks: Recall-at-K values to compute.

    Returns:
        Dict with ``metrics``, ``per_trial`` (ranks, scores, correct),
        and ``cosine_scores`` matrix for downstream use.
    """
    cos_scores = cosine_similarity_matrix(z_pred, gallery)
    ranks = _ranks_from_scores(cos_scores, target_indices)
    top1_scores = cos_scores[np.arange(len(z_pred)), target_indices]

    sorted_scores = np.sort(cos_scores, axis=1)[:, ::-1]
    margin = sorted_scores[:, 0] - sorted_scores[:, 1]
    entropy = topk_entropy(cos_scores, k=min(10, gallery.shape[0]))

    metrics = _metrics_from_ranks(ranks, ks, n_bootstrap=n_bootstrap, seed=seed)
    metrics["top1_score_mean"] = float(np.mean(top1_scores))
    metrics["margin_mean"] = float(np.mean(margin))
    metrics["topk_entropy_mean"] = float(np.mean(entropy))

    per_trial = pd.DataFrame(
        {
            "rank": ranks + 1,
            "rank_0indexed": ranks,
            "top1_score": top1_scores,
            "margin": margin,
            "entropy": entropy,
            "correct": ranks == 0,
        }
    )

    return {
        "metrics": metrics,
        "per_trial": per_trial,
        "cosine_scores": cos_scores,
        "ranks": ranks,
    }


def compute_csls_retrieval(
    z_pred: np.ndarray,
    gallery: np.ndarray,
    target_indices: np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
    csls_k: int = 10,
    cosine_scores: Optional[np.ndarray] = None,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Full CSLS retrieval evaluation.

    Args:
        z_pred: (N, D) query embeddings (L2-normalised).
        gallery: (G, D) gallery embeddings (L2-normalised).
        target_indices: (N,) index of correct gallery item per query.
        ks: Recall-at-K values.
        csls_k: K for CSLS neighbourhood penalty.
        cosine_scores: Pre-computed cosine matrix to avoid recomputation.

    Returns:
        Dict with ``metrics``, ``per_trial``, ``csls_scores``, ``ranks``.
    """
    css = csls_scores(z_pred, gallery, k=csls_k, cosine_scores=cosine_scores)
    ranks = _ranks_from_scores(css, target_indices)
    top1_scores = css[np.arange(len(z_pred)), target_indices]

    sorted_scores = np.sort(css, axis=1)[:, ::-1]
    margin = sorted_scores[:, 0] - sorted_scores[:, 1]
    entropy = topk_entropy(css, k=min(10, gallery.shape[0]))

    metrics = _metrics_from_ranks(ranks, ks, n_bootstrap=n_bootstrap, seed=seed)
    metrics["top1_score_mean"] = float(np.mean(top1_scores))
    metrics["margin_mean"] = float(np.mean(margin))
    metrics["csls_margin_mean"] = float(np.mean(margin))
    metrics["topk_entropy_mean"] = float(np.mean(entropy))

    per_trial = pd.DataFrame(
        {
            "rank": ranks + 1,
            "rank_0indexed": ranks,
            "top1_score": top1_scores,
            "csls_margin": margin,
            "entropy": entropy,
            "correct": ranks == 0,
        }
    )

    return {
        "metrics": metrics,
        "per_trial": per_trial,
        "csls_scores": css,
        "ranks": ranks,
    }


def compute_retrieval_comparison(
    z_pred: np.ndarray,
    gallery: np.ndarray,
    target_indices: np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
    csls_k: int = 10,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run both cosine and CSLS retrieval and return combined results."""
    cos = compute_cosine_retrieval(
        z_pred, gallery, target_indices, ks, n_bootstrap=n_bootstrap, seed=seed
    )
    csl = compute_csls_retrieval(
        z_pred, gallery, target_indices, ks, csls_k,
        cosine_scores=cos["cosine_scores"], n_bootstrap=n_bootstrap, seed=seed
    )
    return {
        "cosine": cos,
        "csls": csl,
        "improvement": {
            k: csl["metrics"][k] - cos["metrics"][k]
            for k in cos["metrics"]
            if k in csl["metrics"] and not k.endswith(("_ci_low", "_ci_high"))
        },
    }


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

def plot_retrieval_comparison_bar(
    cosine_metrics: Dict[str, float],
    csls_metrics: Dict[str, float],
    output_path: Path,
    ks: Sequence[int] = (1, 5, 10),
    **save_kw: Any,
) -> List[Path]:
    """Bar chart comparing cosine vs CSLS R@K."""
    setup_figure_style()
    rk = {f"R@{k}": cosine_metrics.get(f"R@{k}", 0) for k in ks}
    ck = {f"R@{k}": csls_metrics.get(f"R@{k}", 0) for k in ks}
    fig = create_grouped_bar(
        {"Cosine": rk, "CSLS": ck},
        title="Retrieval: Cosine vs CSLS",
        ylabel="Recall",
    )
    return save_figure(fig, output_path, **save_kw)


def plot_rank_distribution(
    cosine_ranks: np.ndarray,
    csls_ranks: np.ndarray,
    output_path: Path,
    **save_kw: Any,
) -> List[Path]:
    """Histogram of retrieval ranks for cosine and CSLS."""
    setup_figure_style()
    cosine_user_ranks = cosine_ranks + 1
    csls_user_ranks = csls_ranks + 1
    max_r = int(max(cosine_user_ranks.max(), csls_user_ranks.max()))
    bins = min(max_r, 100)
    fig = create_histogram(
        {"Cosine": cosine_user_ranks, "CSLS": csls_user_ranks},
        title="Rank Distribution",
        xlabel="Retrieval rank (1 = correct top match)",
        bins=bins,
    )
    return save_figure(fig, output_path, **save_kw)


def plot_csls_margin_distribution(
    margins: np.ndarray,
    correct: np.ndarray,
    output_path: Path,
    **save_kw: Any,
) -> List[Path]:
    """CSLS margin histogram split by correct/incorrect."""
    setup_figure_style()
    fig = create_histogram(
        {
            "Correct": margins[correct],
            "Incorrect": margins[~correct],
        },
        title="CSLS Margin Distribution",
        xlabel="CSLS Margin (score_1 - score_2)",
    )
    return save_figure(fig, output_path, **save_kw)


# ------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------

def save_retrieval_results(
    results: Dict[str, Any],
    output_dir: Path,
    save_formats: Sequence[str] = ("png",),
) -> Dict[str, Any]:
    """Persist retrieval metrics, per-trial data, and figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cos = results["cosine"]
    csl = results["csls"]

    combined_metrics = {
        "cosine": cos["metrics"],
        "csls": csl["metrics"],
        "improvement": results["improvement"],
    }
    with open(output_dir / "retrieval_metrics.json", "w") as f:
        json.dump(combined_metrics, f, indent=2)

    cos["per_trial"].to_parquet(output_dir / "retrieval_cosine_per_trial.parquet")
    csl["per_trial"].to_parquet(output_dir / "retrieval_csls_per_trial.parquet")

    figs: List[Path] = []
    ks = [k for k in [1, 5, 10, 20, 50, 100] if f"R@{k}" in cos["metrics"]]
    figs += plot_retrieval_comparison_bar(
        cos["metrics"], csl["metrics"],
        output_dir / "figures" / "figure_retrieval_comparison",
        ks=ks, formats=save_formats,
    )
    figs += plot_rank_distribution(
        cos["ranks"], csl["ranks"],
        output_dir / "figures" / "figure_rank_distribution",
        formats=save_formats,
    )
    correct = csl["per_trial"]["correct"].values
    margins = csl["per_trial"]["csls_margin"].values
    if correct.any() and (~correct).any():
        figs += plot_csls_margin_distribution(
            margins, correct,
            output_dir / "figures" / "figure_csls_margin_distribution",
            formats=save_formats,
        )

    logger.info("Retrieval results saved to %s", output_dir)
    return {"metrics": combined_metrics, "figures": [str(f) for f in figs]}
