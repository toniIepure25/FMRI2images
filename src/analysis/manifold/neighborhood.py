"""Local manifold / neighbourhood preservation analysis.

Scientific claim tested:
    "High-performing fMRI-to-CLIP decoding preserves local semantic
     neighbourhoods, not just exact image identity."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from analysis.manifold.figures import (
    create_boxplot,
    create_line_plot,
    create_scatter,
    save_figure,
    setup_figure_style,
)
from analysis.manifold.utils import bootstrap_ci, cosine_similarity_matrix

logger = logging.getLogger(__name__)


def _knn_sets(
    score_matrix: np.ndarray, ks: Sequence[int]
) -> Dict[int, np.ndarray]:
    """Top-K indices per query for each K in *ks*."""
    max_k = max(ks)
    max_k = min(max_k, score_matrix.shape[1])
    all_topk = np.argpartition(score_matrix, -max_k, axis=1)[:, -max_k:]

    sorted_within = np.argsort(
        -np.take_along_axis(score_matrix, all_topk, axis=1), axis=1
    )
    all_topk_sorted = np.take_along_axis(all_topk, sorted_within, axis=1)

    result: Dict[int, np.ndarray] = {}
    for k in ks:
        k_ = min(k, max_k)
        result[k] = all_topk_sorted[:, :k_]
    return result


def compute_neighborhood_overlap(
    z_pred: np.ndarray,
    z_target: np.ndarray,
    gallery: np.ndarray,
    ks: Sequence[int] = (5, 10, 20, 50, 100),
    n_bootstrap: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Neighbourhood overlap between target-space and prediction-space KNN.

    For each sample *i*:
        overlap@K = |N_target(i) ∩ N_pred(i)| / K

    Args:
        z_pred: (N, D) predicted embeddings.
        z_target: (N, D) target embeddings.
        gallery: (G, D) gallery embeddings (same pool for both lookups).
        ks: K values.

    Returns:
        Dict with per-K mean/median overlap and per-trial array.
    """
    scores_pred = cosine_similarity_matrix(z_pred, gallery)
    scores_tgt = cosine_similarity_matrix(z_target, gallery)

    knn_pred = _knn_sets(scores_pred, ks)
    knn_tgt = _knn_sets(scores_tgt, ks)

    n = z_pred.shape[0]
    gallery_size = gallery.shape[0]
    results: Dict[str, Any] = {"ks": list(ks), "gallery_size": int(gallery_size)}
    per_trial_overlaps: Dict[int, np.ndarray] = {}

    for k in ks:
        overlaps = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s_pred = set(knn_pred[k][i])
            s_tgt = set(knn_tgt[k][i])
            overlaps[i] = len(s_pred & s_tgt) / k
        per_trial_overlaps[k] = overlaps
        results[f"overlap@{k}_mean"] = float(overlaps.mean())
        results[f"overlap@{k}_median"] = float(np.median(overlaps))
        random_baseline = min(float(k) / float(gallery_size), 1.0)
        results[f"overlap@{k}_random_baseline"] = random_baseline
        results[f"overlap@{k}_lift_over_random"] = (
            float(overlaps.mean() / random_baseline) if random_baseline > 0 else None
        )
        if n_bootstrap > 0:
            lo, hi = bootstrap_ci(overlaps, np.mean, n_bootstrap=n_bootstrap, seed=seed)
            results[f"overlap@{k}_ci_low"] = lo
            results[f"overlap@{k}_ci_high"] = hi

    results["per_trial_overlaps"] = per_trial_overlaps
    return results


def compute_trustworthiness_continuity(
    z_pred: np.ndarray,
    z_target: np.ndarray,
    k: int = 10,
) -> Dict[str, float]:
    """Trustworthiness and continuity via sklearn if available."""
    try:
        from sklearn.manifold import trustworthiness as tw_fn
    except ImportError:
        logger.warning("sklearn.manifold.trustworthiness not available; skipping")
        return {"trustworthiness": None, "continuity": None}

    k_ = min(k, z_pred.shape[0] - 2)
    trust = float(tw_fn(z_target, z_pred, n_neighbors=k_, metric="cosine"))
    cont = float(tw_fn(z_pred, z_target, n_neighbors=k_, metric="cosine"))
    return {"trustworthiness": trust, "continuity": cont}


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

def plot_overlap_vs_k(
    results: Dict[str, Any],
    output_path: Path,
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    ks = results["ks"]
    means = np.array([results[f"overlap@{k}_mean"] for k in ks])
    baseline = np.array([results.get(f"overlap@{k}_random_baseline", 0.0) for k in ks])
    fig = create_line_plot(
        {
            "Observed overlap": (np.array(ks, dtype=float), means),
            "Random baseline": (np.array(ks, dtype=float), baseline),
        },
        title="Neighbourhood Overlap vs K",
        xlabel="K",
        ylabel="Mean Overlap",
    )
    return save_figure(fig, output_path, **save_kw)


def plot_overlap_correct_vs_incorrect(
    overlaps: np.ndarray,
    correct: np.ndarray,
    output_path: Path,
    k: int = 10,
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    fig = create_boxplot(
        {"Correct": overlaps[correct], "Incorrect": overlaps[~correct]},
        title=f"Neighbourhood Overlap@{k}: Correct vs Incorrect",
        ylabel=f"Overlap@{k}",
    )
    return save_figure(fig, output_path, **save_kw)


def plot_overlap_vs_rank(
    overlaps: np.ndarray,
    ranks: np.ndarray,
    output_path: Path,
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    fig = create_scatter(
        ranks.astype(float) + 1, overlaps,
        title="Neighbourhood Overlap vs Retrieval Rank",
        xlabel="Retrieval rank (1 = correct top match)",
        ylabel="Neighbourhood Overlap",
    )
    return save_figure(fig, output_path, **save_kw)


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------

def save_neighborhood_results(
    results: Dict[str, Any],
    correct: np.ndarray,
    ranks: np.ndarray,
    output_dir: Path,
    save_formats: Sequence[str] = ("png",),
    k_for_detail: int = 10,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {k: v for k, v in results.items()
               if k != "per_trial_overlaps"}
    with open(output_dir / "neighborhood_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    import pandas as pd
    overlap_cols = {}
    for k_, arr in results.get("per_trial_overlaps", {}).items():
        overlap_cols[f"overlap@{k_}"] = arr
    if overlap_cols:
        pd.DataFrame(overlap_cols).to_csv(
            output_dir / "per_trial_neighborhood.csv", index=False
        )

    figs_dir = output_dir / "figures"
    figs: List[Path] = []
    figs += plot_overlap_vs_k(results, figs_dir / "figure_neighborhood_overlap_vs_k",
                              formats=save_formats)

    overlaps_detail = results.get("per_trial_overlaps", {}).get(k_for_detail)
    if overlaps_detail is not None and correct is not None:
        if correct.any() and (~correct).any():
            figs += plot_overlap_correct_vs_incorrect(
                overlaps_detail, correct,
                figs_dir / "figure_neighborhood_correct_vs_incorrect",
                k=k_for_detail, formats=save_formats,
            )
    if overlaps_detail is not None and ranks is not None:
        figs += plot_overlap_vs_rank(
            overlaps_detail, ranks,
            figs_dir / "figure_neighborhood_overlap_vs_rank",
            formats=save_formats,
        )

    logger.info("Neighbourhood results saved to %s", output_dir)
    return {"metrics": metrics, "figures": [str(f) for f in figs]}
