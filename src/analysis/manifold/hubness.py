"""Hubness analysis for neural-to-image retrieval.

Scientific claim tested:
    "CSLS improves retrieval partly by reducing hubness in neural-to-CLIP search."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats as sp_stats

from analysis.manifold.figures import (
    create_bar_chart,
    create_histogram,
    create_line_plot,
    save_figure,
    setup_figure_style,
)
from analysis.manifold.utils import gini_coefficient

logger = logging.getLogger(__name__)


def compute_hub_counts(
    score_matrix: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Count how often each gallery item appears in the top-K of any query.

    Args:
        score_matrix: (N_queries, N_gallery) similarity scores.
        k: Top-K for neighbourhood.

    Returns:
        (N_gallery,) occurrence counts.
    """
    n_gallery = score_matrix.shape[1]
    k = min(k, n_gallery)
    topk = np.argpartition(score_matrix, -k, axis=1)[:, -k:]
    counts = np.bincount(topk.ravel(), minlength=n_gallery)
    return counts


def compute_hubness_metrics(
    cosine_counts: np.ndarray,
    csls_counts: np.ndarray,
    correct_indices: Optional[np.ndarray] = None,
    csls_topk_indices: Optional[np.ndarray] = None,
    k: int = 10,
) -> Dict[str, Any]:
    """Compute hubness statistics for cosine and CSLS retrievals.

    Args:
        cosine_counts: (G,) hub counts under cosine retrieval.
        csls_counts: (G,) hub counts under CSLS retrieval.
        correct_indices: (N,) gallery index of correct answer per query.
        csls_topk_indices: (N, k) top-k indices from CSLS retrieval.
        k: The K used for hub counts.

    Returns:
        Dict of hubness metrics for both scoring modes.
    """
    def _stats(counts: np.ndarray, label: str) -> Dict[str, float]:
        n = len(counts)
        s: Dict[str, float] = {}
        s[f"{label}_max_hub"] = int(counts.max())
        s[f"{label}_mean_hub"] = float(counts.mean())
        s[f"{label}_std_hub"] = float(counts.std())
        s[f"{label}_skewness"] = float(sp_stats.skew(counts))
        s[f"{label}_gini"] = gini_coefficient(counts)
        sorted_c = np.sort(counts)[::-1]
        cumsum = np.cumsum(sorted_c) / sorted_c.sum()
        for pct in [1, 5, 10]:
            top_n = max(1, int(n * pct / 100))
            s[f"{label}_top{pct}pct_share"] = float(cumsum[top_n - 1])
        return s

    metrics = {}
    metrics.update(_stats(cosine_counts, "cosine"))
    metrics.update(_stats(csls_counts, "csls"))
    metrics["k"] = k
    return metrics


def compute_lorenz_curve(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lorenz curve: cumulative share of retrievals vs cumulative share of gallery items."""
    sorted_c = np.sort(counts)
    cumsum = np.cumsum(sorted_c) / sorted_c.sum()
    x = np.linspace(0, 1, len(cumsum))
    return x, cumsum


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

def plot_hubness_histogram(
    cosine_counts: np.ndarray,
    csls_counts: np.ndarray,
    output_path: Path,
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    fig = create_histogram(
        {"Cosine": cosine_counts, "CSLS": csls_counts},
        title="Hub Count Distribution",
        xlabel=f"Times appearing in top-K",
        bins=50,
    )
    return save_figure(fig, output_path, **save_kw)


def plot_lorenz_curve(
    cosine_counts: np.ndarray,
    csls_counts: np.ndarray,
    output_path: Path,
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    x_cos, y_cos = compute_lorenz_curve(cosine_counts)
    x_csl, y_csl = compute_lorenz_curve(csls_counts)
    fig = create_line_plot(
        {"Cosine": (x_cos, y_cos), "CSLS": (x_csl, y_csl), "Equality": (np.array([0, 1]), np.array([0, 1]))},
        title="Lorenz Curve — Retrieval Concentration",
        xlabel="Cumulative share of gallery images",
        ylabel="Cumulative share of retrievals",
    )
    return save_figure(fig, output_path, **save_kw)


def plot_top_hubs_bar(
    counts: np.ndarray,
    output_path: Path,
    n_top: int = 20,
    label: str = "CSLS",
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    top_idx = np.argsort(counts)[::-1][:n_top]
    data = {f"img_{idx}": int(counts[idx]) for idx in top_idx}
    fig = create_bar_chart(data, title=f"Top {n_top} Hubs ({label})", ylabel="Hub count")
    return save_figure(fig, output_path, **save_kw)


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------

def save_hubness_results(
    cosine_counts: np.ndarray,
    csls_counts: np.ndarray,
    metrics: Dict[str, Any],
    output_dir: Path,
    save_formats: Sequence[str] = ("png",),
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "hubness_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    top_idx = np.argsort(csls_counts)[::-1][:50]
    import pandas as pd
    pd.DataFrame({"gallery_index": top_idx, "csls_hub_count": csls_counts[top_idx],
                   "cosine_hub_count": cosine_counts[top_idx]}).to_csv(
        output_dir / "hubness_top_images.csv", index=False
    )

    figs_dir = output_dir / "figures"
    figs: List[Path] = []
    figs += plot_hubness_histogram(cosine_counts, csls_counts,
                                   figs_dir / "figure_hubness_distribution", formats=save_formats)
    figs += plot_lorenz_curve(cosine_counts, csls_counts,
                              figs_dir / "figure_hubness_lorenz", formats=save_formats)
    figs += plot_top_hubs_bar(csls_counts, figs_dir / "figure_hubness_top_hubs",
                              label="CSLS", formats=save_formats)
    logger.info("Hubness results saved to %s", output_dir)
    return {"metrics": metrics, "figures": [str(f) for f in figs]}
