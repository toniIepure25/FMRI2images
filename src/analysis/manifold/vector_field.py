"""Semantic error vector field analysis.

Scientific claim tested:
    "Neural decoding errors follow structured trajectories on the CLIP
     semantic manifold rather than random noise."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from analysis.manifold.figures import (
    PALETTE,
    create_histogram,
    create_scatter,
    save_figure,
    setup_figure_style,
)
from analysis.manifold.utils import cosine_similarity_matrix, safe_normalize

logger = logging.getLogger(__name__)


def compute_error_vectors(
    z_pred: np.ndarray,
    z_target: np.ndarray,
) -> Dict[str, Any]:
    """Compute and analyse prediction error vectors.

    Args:
        z_pred: (N, D) predicted embeddings.
        z_target: (N, D) target embeddings.

    Returns:
        Dict with error magnitude statistics, direction similarity, etc.
    """
    errors = z_target - z_pred
    magnitudes = np.linalg.norm(errors, axis=1)

    error_dirs = safe_normalize(errors)
    dir_sim = cosine_similarity_matrix(error_dirs, error_dirs)
    mean_dir_sim = float(np.mean(dir_sim[np.triu_indices_from(dir_sim, k=1)]))

    return {
        "error_magnitude_mean": float(magnitudes.mean()),
        "error_magnitude_std": float(magnitudes.std()),
        "error_magnitude_median": float(np.median(magnitudes)),
        "mean_error_direction_similarity": mean_dir_sim,
        "_errors": errors,
        "_magnitudes": magnitudes,
    }


def analyze_error_structure(
    errors: np.ndarray,
    n_clusters: int = 5,
) -> Dict[str, Any]:
    """Cluster error directions and report structure."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        logger.warning("sklearn.cluster.KMeans not available; skipping clustering")
        return {"error_clusters": None}

    dirs = safe_normalize(errors)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(dirs)
    inertia = float(km.inertia_)
    cluster_sizes = [int((labels == i).sum()) for i in range(n_clusters)]
    return {
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "inertia": inertia,
        "_labels": labels,
    }


def project_to_2d(
    z_pred: np.ndarray,
    z_target: np.ndarray,
    method: str = "pca",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Project pred and target into shared 2D space for visualisation."""
    combined = np.concatenate([z_pred, z_target], axis=0)
    n = z_pred.shape[0]

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=seed, metric="cosine")
            proj = reducer.fit_transform(combined)
        except ImportError:
            logger.info("umap not installed; falling back to PCA")
            method = "pca"

    if method == "pca":
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2, random_state=seed).fit_transform(combined)

    return {"pred_2d": proj[:n], "target_2d": proj[n:]}


def run_vector_field(
    z_pred: np.ndarray,
    z_target: np.ndarray,
    correct: Optional[np.ndarray] = None,
    projection_method: str = "pca",
    n_clusters: int = 5,
) -> Dict[str, Any]:
    """Top-level semantic error vector analysis."""
    ev = compute_error_vectors(z_pred, z_target)
    es = analyze_error_structure(ev["_errors"], n_clusters)
    proj = project_to_2d(z_pred, z_target, method=projection_method)

    metrics = {k: v for k, v in ev.items() if not k.startswith("_")}
    metrics.update({k: v for k, v in es.items() if not k.startswith("_")})
    metrics["status"] = "computed"

    return {
        **metrics,
        "_errors": ev["_errors"],
        "_magnitudes": ev["_magnitudes"],
        "_pred_2d": proj["pred_2d"],
        "_target_2d": proj["target_2d"],
        "_labels": es.get("_labels"),
    }


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

def plot_error_vector_field_2d(
    pred_2d: np.ndarray,
    target_2d: np.ndarray,
    correct: Optional[np.ndarray],
    output_path: Path,
    max_arrows: int = 500,
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    n = pred_2d.shape[0]
    if n > max_arrows:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_arrows, replace=False)
        pred_2d = pred_2d[idx]
        target_2d = target_2d[idx]
        correct = correct[idx] if correct is not None else None

    fig, ax = plt.subplots(figsize=(8, 7))
    dx = target_2d[:, 0] - pred_2d[:, 0]
    dy = target_2d[:, 1] - pred_2d[:, 1]

    if correct is not None:
        colors = np.where(correct, PALETTE[2], PALETTE[1])
    else:
        colors = PALETTE[0]

    ax.quiver(pred_2d[:, 0], pred_2d[:, 1], dx, dy,
              color=colors, alpha=0.5, scale_units="xy", angles="xy", scale=1,
              headwidth=3, headlength=4, width=0.002)
    ax.set_title("Semantic Error Vector Field (2D projection)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    return save_figure(fig, output_path, **save_kw)


def plot_error_magnitude_distribution(
    magnitudes: np.ndarray,
    correct: Optional[np.ndarray],
    output_path: Path,
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    groups = {"All": magnitudes}
    if correct is not None and correct.any() and (~correct).any():
        groups = {"Correct": magnitudes[correct], "Incorrect": magnitudes[~correct]}
    fig = create_histogram(groups, title="Error Magnitude Distribution",
                           xlabel="L2 error magnitude", bins=50)
    return save_figure(fig, output_path, **save_kw)


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------

def save_vector_field_results(
    results: Dict[str, Any],
    correct: Optional[np.ndarray],
    output_dir: Path,
    save_formats: Sequence[str] = ("png",),
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_2d = results.pop("_pred_2d", None)
    target_2d = results.pop("_target_2d", None)
    magnitudes = results.pop("_magnitudes", None)
    results.pop("_errors", None)
    results.pop("_labels", None)
    metrics = {k: v for k, v in results.items() if not k.startswith("_")}
    with open(output_dir / "vector_field_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    figs: List[Path] = []
    figs_dir = output_dir / "figures"
    if pred_2d is not None and target_2d is not None:
        figs += plot_error_vector_field_2d(
            pred_2d, target_2d, correct,
            figs_dir / "figure_semantic_error_vector_field",
            formats=save_formats,
        )
    if magnitudes is not None:
        figs += plot_error_magnitude_distribution(
            magnitudes, correct,
            figs_dir / "figure_error_magnitude_distribution",
            formats=save_formats,
        )

    logger.info("Vector field results saved to %s", output_dir)
    return {"metrics": metrics, "figures": [str(f) for f in figs]}
