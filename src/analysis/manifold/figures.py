"""Publication-quality figure utilities for manifold analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

PALETTE = [
    "#2176AE",  # blue
    "#E85D75",  # rose
    "#57B894",  # green
    "#FBB13C",  # amber
    "#8B5CF6",  # violet
    "#F97316",  # orange
    "#06B6D4",  # cyan
    "#EF4444",  # red
    "#6366F1",  # indigo
    "#10B981",  # emerald
]


def setup_figure_style() -> None:
    """Set matplotlib defaults for publication-quality figures."""
    matplotlib.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.figsize": (6, 4),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.constrained_layout.use": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    formats: Sequence[str] = ("png",),
    close: bool = True,
) -> List[Path]:
    """Save *fig* in each requested format, return paths."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for fmt in formats:
        out = p.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt, bbox_inches="tight")
        saved.append(out)
        logger.info("Saved figure: %s", out)
    if close:
        plt.close(fig)
    return saved


def create_bar_chart(
    data: Dict[str, float],
    title: str = "",
    ylabel: str = "",
    colors: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (7, 4),
) -> plt.Figure:
    """Horizontal bar chart from a label->value dict."""
    fig, ax = plt.subplots(figsize=figsize)
    labels = list(data.keys())
    values = list(data.values())
    c = colors or PALETTE[: len(labels)]
    bars = ax.barh(labels, values, color=c)
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.invert_yaxis()
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)
    return fig


def create_grouped_bar(
    groups: Dict[str, Dict[str, float]],
    title: str = "",
    ylabel: str = "",
    figsize: Tuple[float, float] = (8, 4),
) -> plt.Figure:
    """Grouped vertical bar chart (e.g. cosine vs CSLS for R@1, R@5, ...)."""
    fig, ax = plt.subplots(figsize=figsize)
    group_names = list(groups.keys())
    metrics = list(next(iter(groups.values())).keys())
    n_groups = len(group_names)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_groups
    for i, gname in enumerate(group_names):
        vals = [groups[gname].get(m, 0) for m in metrics]
        offset = (i - n_groups / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=gname, color=PALETTE[i % len(PALETTE)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig


def create_histogram(
    arrays: Dict[str, np.ndarray],
    title: str = "",
    xlabel: str = "",
    bins: int = 50,
    alpha: float = 0.6,
    figsize: Tuple[float, float] = (7, 4),
) -> plt.Figure:
    """Overlaid histograms for multiple arrays."""
    fig, ax = plt.subplots(figsize=figsize)
    for i, (label, arr) in enumerate(arrays.items()):
        ax.hist(arr, bins=bins, alpha=alpha, label=label, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    return fig


def create_scatter(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    color: Optional[np.ndarray] = None,
    cmap: str = "viridis",
    alpha: float = 0.4,
    figsize: Tuple[float, float] = (6, 5),
) -> plt.Figure:
    """Scatter plot with optional colour mapping."""
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(x, y, c=color, cmap=cmap, alpha=alpha, s=8, edgecolors="none")
    if color is not None:
        fig.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig


def create_heatmap(
    matrix: np.ndarray,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[float, float] = (6, 5),
) -> plt.Figure:
    """Image-style heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig


def create_boxplot(
    groups: Dict[str, np.ndarray],
    title: str = "",
    ylabel: str = "",
    figsize: Tuple[float, float] = (6, 4),
) -> plt.Figure:
    """Side-by-side box plots for named groups."""
    fig, ax = plt.subplots(figsize=figsize)
    labels = list(groups.keys())
    data = [groups[l] for l in labels]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig


def create_line_plot(
    series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: Tuple[float, float] = (7, 4),
) -> plt.Figure:
    """Line plot for multiple (x, y) series."""
    fig, ax = plt.subplots(figsize=figsize)
    for i, (label, (x, y)) in enumerate(series.items()):
        ax.plot(x, y, marker="o", markersize=4, label=label, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig


def create_radar_chart(
    values: Dict[str, float],
    title: str = "",
    figsize: Tuple[float, float] = (6, 6),
) -> plt.Figure:
    """Radar / spider chart for multi-dimensional scores."""
    labels = list(values.keys())
    vals = np.array([values[l] for l in labels])
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    vals_closed = np.concatenate([vals, [vals[0]]])
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    ax.fill(angles_closed, vals_closed, alpha=0.25, color=PALETTE[0])
    ax.plot(angles_closed, vals_closed, color=PALETTE[0], linewidth=2)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, y=1.08)
    return fig
