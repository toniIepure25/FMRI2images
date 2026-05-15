"""Repeat stability analysis across NSD stimulus repetitions.

Scientific claim tested:
    "Repeated presentations of the same stimulus produce stable clusters
     in decoded CLIP space."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats as sp_stats

from analysis.manifold.figures import (
    create_boxplot,
    create_histogram,
    create_scatter,
    save_figure,
    setup_figure_style,
)
from analysis.manifold.utils import cosine_similarity_matrix, safe_normalize

logger = logging.getLogger(__name__)

_UNAVAILABLE = {
    "status": "unavailable",
    "reason": (
        "Per-trial predictions with repeat metadata are required. "
        "Provide per_trial_pred_path and per_trial_nsd_ids_path, "
        "or save per-trial predictions during training "
        "(e.g. --save-train-preds without nsdId aggregation)."
    ),
}


def compute_repeat_clusters(
    z_pred_trials: np.ndarray,
    nsd_ids: np.ndarray,
    min_repeats: int = 2,
) -> Dict[int, np.ndarray]:
    """Group per-trial predicted embeddings by stimulus nsdId.

    Returns:
        Dict mapping nsdId -> (R, D) array of repeat embeddings.
    """
    clusters: Dict[int, list] = {}
    for i, nid in enumerate(nsd_ids):
        nid = int(nid)
        clusters.setdefault(nid, []).append(z_pred_trials[i])
    return {
        nid: np.stack(vecs)
        for nid, vecs in clusters.items()
        if len(vecs) >= min_repeats
    }


def compute_within_between_similarity(
    clusters: Dict[int, np.ndarray],
    max_between_pairs: int = 100_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Within-image and between-image cosine similarities.

    Args:
        clusters: nsdId -> (R, D) embedding arrays.
        max_between_pairs: Cap random between-image pairs for speed.
        seed: RNG seed.

    Returns:
        Dict with within/between similarity arrays and statistics.
    """
    within_sims: List[float] = []
    for nid, vecs in clusters.items():
        vecs = safe_normalize(vecs)
        sim = cosine_similarity_matrix(vecs, vecs)
        n = sim.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                within_sims.append(float(sim[i, j]))
    within_arr = np.array(within_sims)

    all_means = []
    for nid, vecs in clusters.items():
        all_means.append(safe_normalize(vecs.mean(axis=0, keepdims=True)).ravel())
    all_means = np.array(all_means)

    rng = np.random.RandomState(seed)
    n_imgs = len(all_means)
    n_pairs = min(max_between_pairs, n_imgs * (n_imgs - 1) // 2)
    between_sims: List[float] = []
    if n_imgs >= 2:
        for _ in range(n_pairs):
            i, j = rng.choice(n_imgs, 2, replace=False)
            between_sims.append(float(np.dot(all_means[i], all_means[j])))
    between_arr = np.array(between_sims) if between_sims else np.array([])

    within_mean = float(within_arr.mean()) if len(within_arr) else 0.0
    between_mean = float(between_arr.mean()) if len(between_arr) else 0.0
    within_var = float(within_arr.var()) if len(within_arr) else 0.0
    between_var = float(between_arr.var()) if len(between_arr) else 0.0

    stability_ratio = between_var / (within_var + 1e-12) if within_var > 0 else float("inf")

    return {
        "within_sims": within_arr,
        "between_sims": between_arr,
        "within_mean": within_mean,
        "between_mean": between_mean,
        "within_var": within_var,
        "between_var": between_var,
        "stability_ratio": stability_ratio,
        "n_images_with_repeats": len(clusters),
        "n_within_pairs": len(within_arr),
        "n_between_pairs": len(between_arr),
    }


def compute_same_different_auroc(
    within_sims: np.ndarray,
    between_sims: np.ndarray,
) -> float:
    """AUROC for classifying same-image vs different-image pairs."""
    if len(within_sims) == 0 or len(between_sims) == 0:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        logger.warning("sklearn not available for AUROC; skipping")
        return float("nan")
    labels = np.concatenate([np.ones(len(within_sims)), np.zeros(len(between_sims))])
    scores = np.concatenate([within_sims, between_sims])
    return float(roc_auc_score(labels, scores))


def run_repeat_stability(
    z_pred_trials: Optional[np.ndarray],
    nsd_ids: Optional[np.ndarray],
    seed: int = 42,
) -> Dict[str, Any]:
    """Top-level repeat stability entry point.

    Returns unavailable status if inputs are missing.
    """
    if z_pred_trials is None or nsd_ids is None:
        logger.info("Repeat stability: skipped — per-trial data unavailable")
        return dict(_UNAVAILABLE)

    clusters = compute_repeat_clusters(z_pred_trials, nsd_ids)
    if len(clusters) < 5:
        return {
            "status": "unavailable",
            "reason": f"Only {len(clusters)} images with >=2 repeats found.",
        }

    wb = compute_within_between_similarity(clusters, seed=seed)
    auroc = compute_same_different_auroc(wb["within_sims"], wb["between_sims"])

    metrics = {
        "status": "computed",
        "within_mean": wb["within_mean"],
        "between_mean": wb["between_mean"],
        "within_var": wb["within_var"],
        "between_var": wb["between_var"],
        "stability_ratio": wb["stability_ratio"],
        "same_different_auroc": auroc,
        "n_images_with_repeats": wb["n_images_with_repeats"],
        "n_within_pairs": wb["n_within_pairs"],
    }
    return {**metrics, "_within_sims": wb["within_sims"], "_between_sims": wb["between_sims"]}


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

def plot_same_vs_different(
    within_sims: np.ndarray,
    between_sims: np.ndarray,
    output_path: Path,
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    fig = create_histogram(
        {"Same image": within_sims, "Different images": between_sims},
        title="Repeat Stability: Same vs Different Image Similarity",
        xlabel="Cosine Similarity",
        bins=60,
    )
    return save_figure(fig, output_path, **save_kw)


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------

def save_repeat_stability_results(
    results: Dict[str, Any],
    output_dir: Path,
    save_formats: Sequence[str] = ("png",),
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {k: v for k, v in results.items() if not k.startswith("_")}
    with open(output_dir / "repeat_stability_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    figs: List[Path] = []
    within = results.get("_within_sims")
    between = results.get("_between_sims")
    if within is not None and between is not None and len(within) > 0:
        figs_dir = output_dir / "figures"
        figs += plot_same_vs_different(
            within, between,
            figs_dir / "figure_repeat_same_vs_different",
            formats=save_formats,
        )

    logger.info("Repeat stability results saved to %s", output_dir)
    return {"metrics": metrics, "figures": [str(f) for f in figs]}
