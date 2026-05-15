"""Manifold density / off-manifold detection.

Scientific claim tested:
    "Reliable decoded embeddings tend to lie in dense regions of the
     natural image CLIP manifold."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from analysis.manifold.figures import (
    create_boxplot,
    create_histogram,
    create_scatter,
    save_figure,
    setup_figure_style,
)
from analysis.manifold.utils import cosine_similarity_matrix

logger = logging.getLogger(__name__)


def compute_manifold_density(
    z_pred: np.ndarray,
    gallery: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Per-sample manifold density: mean cosine similarity to K-NN in gallery.

    Args:
        z_pred: (N, D) predicted embeddings (L2-normalised).
        gallery: (G, D) gallery CLIP embeddings (L2-normalised).
        k: Number of nearest gallery neighbours.

    Returns:
        (N,) density scores in (0, 1].
    """
    scores = cosine_similarity_matrix(z_pred, gallery)
    k = min(k, scores.shape[1])
    topk = np.partition(scores, -k, axis=1)[:, -k:]
    return np.mean(topk, axis=1)


def compute_reliability_quadrants(
    kappa: np.ndarray,
    density: np.ndarray,
    correct: np.ndarray,
) -> Dict[str, Any]:
    """Split predictions into reliability quadrants (high/low kappa x density)."""
    k_med = float(np.median(kappa))
    d_med = float(np.median(density))

    quads = {
        "high_k_high_d": (kappa >= k_med) & (density >= d_med),
        "high_k_low_d": (kappa >= k_med) & (density < d_med),
        "low_k_high_d": (kappa < k_med) & (density >= d_med),
        "low_k_low_d": (kappa < k_med) & (density < d_med),
    }
    results: Dict[str, Any] = {"kappa_median": k_med, "density_median": d_med}
    for name, mask in quads.items():
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n > 0 else None
        results[name] = {"n": n, "accuracy": acc}
    return results


def run_manifold_density(
    z_pred: np.ndarray,
    gallery: np.ndarray,
    correct: np.ndarray,
    kappa: Optional[np.ndarray] = None,
    k: int = 20,
) -> Dict[str, Any]:
    """Top-level manifold density entry point."""
    density = compute_manifold_density(z_pred, gallery, k)
    metrics: Dict[str, Any] = {
        "status": "computed",
        "density_mean": float(density.mean()),
        "density_std": float(density.std()),
        "density_median": float(np.median(density)),
        "density_correct_mean": float(density[correct].mean()) if correct.any() else None,
        "density_incorrect_mean": float(density[~correct].mean()) if (~correct).any() else None,
    }
    if kappa is not None:
        from scipy import stats as sp_stats
        rho, p = sp_stats.spearmanr(kappa, density)
        metrics["kappa_density_spearman"] = float(rho)
        metrics["kappa_density_p"] = float(p)
        metrics["quadrants"] = compute_reliability_quadrants(kappa, density, correct)
    return {**metrics, "_density": density}


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

def plot_density_distribution(
    density: np.ndarray, output_path: Path, **save_kw: Any
) -> List[Path]:
    setup_figure_style()
    fig = create_histogram({"Manifold density": density},
                           title="Manifold Density Distribution",
                           xlabel="Mean cosine to K-NN", bins=50)
    return save_figure(fig, output_path, **save_kw)


def plot_density_correct_vs_incorrect(
    density: np.ndarray, correct: np.ndarray, output_path: Path, **save_kw: Any
) -> List[Path]:
    setup_figure_style()
    fig = create_boxplot(
        {"Correct": density[correct], "Incorrect": density[~correct]},
        title="Manifold Density: Correct vs Incorrect",
        ylabel="Density",
    )
    return save_figure(fig, output_path, **save_kw)


def plot_kappa_density_quadrants(
    kappa: np.ndarray,
    density: np.ndarray,
    correct: np.ndarray,
    output_path: Path,
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    fig = create_scatter(
        kappa, density, color=correct.astype(float),
        title="Kappa vs Manifold Density (colour = correct)",
        xlabel="kappa", ylabel="Manifold density", cmap="coolwarm",
    )
    return save_figure(fig, output_path, **save_kw)


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------

def save_manifold_density_results(
    results: Dict[str, Any],
    correct: np.ndarray,
    kappa: Optional[np.ndarray],
    output_dir: Path,
    save_formats: Sequence[str] = ("png",),
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    density = results.pop("_density", None)
    metrics = {k: v for k, v in results.items() if not k.startswith("_")}
    with open(output_dir / "manifold_density_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    if density is not None:
        import pandas as pd
        pd.DataFrame({"density": density}).to_csv(
            output_dir / "per_trial_density.csv", index=False
        )

    figs: List[Path] = []
    if density is not None:
        figs_dir = output_dir / "figures"
        figs += plot_density_distribution(density, figs_dir / "figure_density_distribution",
                                          formats=save_formats)
        if correct.any() and (~correct).any():
            figs += plot_density_correct_vs_incorrect(
                density, correct, figs_dir / "figure_density_correct_vs_incorrect",
                formats=save_formats,
            )
        if kappa is not None:
            figs += plot_kappa_density_quadrants(
                kappa, density, correct,
                figs_dir / "figure_kappa_density_quadrants",
                formats=save_formats,
            )

    logger.info("Manifold density results saved to %s", output_dir)
    return {"metrics": metrics, "figures": [str(f) for f in figs]}
