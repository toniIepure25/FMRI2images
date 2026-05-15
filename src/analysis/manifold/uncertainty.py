"""vMF directional uncertainty (kappa) analysis.

Scientific claims tested:
    1. "vMF kappa provides meaningful directional uncertainty for
        fMRI-to-CLIP retrieval."
    2. "High retrieval accuracy does not necessarily imply calibrated
        uncertainty."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from analysis.manifold.figures import (
    PALETTE,
    create_boxplot,
    create_histogram,
    create_scatter,
    save_figure,
    setup_figure_style,
)

logger = logging.getLogger(__name__)

_UNAVAILABLE = {
    "status": "unavailable",
    "reason": "Kappa values not provided (kappa_path is null).",
}


def compute_kappa_retrieval_correlation(
    kappa: np.ndarray,
    correct: np.ndarray,
    margins: Optional[np.ndarray] = None,
    entropy: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Correlations between kappa and retrieval outcomes."""
    results: Dict[str, Any] = {}
    results["kappa_mean"] = float(kappa.mean())
    results["kappa_std"] = float(kappa.std())
    results["kappa_min"] = float(kappa.min())
    results["kappa_max"] = float(kappa.max())
    results["kappa_median"] = float(np.median(kappa))

    corr_correct = sp_stats.pointbiserialr(correct.astype(float), kappa)
    results["kappa_correct_rpb"] = float(corr_correct.correlation)
    results["kappa_correct_p"] = float(corr_correct.pvalue)
    results["kappa_correct_mean"] = float(kappa[correct].mean()) if correct.any() else None
    results["kappa_incorrect_mean"] = float(kappa[~correct].mean()) if (~correct).any() else None

    if margins is not None:
        rho, p = sp_stats.spearmanr(kappa, margins)
        results["kappa_margin_spearman"] = float(rho)
        results["kappa_margin_p"] = float(p)

    if entropy is not None:
        rho, p = sp_stats.spearmanr(kappa, entropy)
        results["kappa_entropy_spearman"] = float(rho)
        results["kappa_entropy_p"] = float(p)

    return results


def compute_kappa_auroc(
    kappa: np.ndarray,
    correct: np.ndarray,
) -> float:
    """AUROC of kappa for predicting retrieval correctness."""
    if not correct.any() or not (~correct).any():
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(correct.astype(int), kappa))
    except ImportError:
        logger.warning("sklearn not available for AUROC")
        return float("nan")


def compute_calibration_curve(
    kappa: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Binned calibration: mean kappa vs mean accuracy per bin."""
    sorted_idx = np.argsort(kappa)
    bins = np.array_split(sorted_idx, n_bins)
    bin_kappa = [float(kappa[b].mean()) for b in bins if len(b)]
    bin_acc = [float(correct[b].mean()) for b in bins if len(b)]
    bin_count = [len(b) for b in bins if len(b)]
    return {
        "bin_kappa": bin_kappa,
        "bin_accuracy": bin_acc,
        "bin_count": bin_count,
    }


def compute_ece(
    kappa: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error using normalised kappa as confidence proxy."""
    kappa_norm = (kappa - kappa.min()) / (kappa.max() - kappa.min() + 1e-12)
    sorted_idx = np.argsort(kappa_norm)
    bins = np.array_split(sorted_idx, n_bins)
    ece = 0.0
    n = len(kappa)
    for b in bins:
        if len(b) == 0:
            continue
        conf = kappa_norm[b].mean()
        acc = correct[b].mean()
        ece += len(b) / n * abs(acc - conf)
    return float(ece)


def run_uncertainty_analysis(
    kappa: Optional[np.ndarray],
    correct: np.ndarray,
    margins: Optional[np.ndarray] = None,
    entropy: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Top-level uncertainty analysis entry point."""
    if kappa is None:
        logger.info("Uncertainty analysis: skipped — kappa unavailable")
        return dict(_UNAVAILABLE)

    corr = compute_kappa_retrieval_correlation(kappa, correct, margins, entropy)
    auroc = compute_kappa_auroc(kappa, correct)
    cal = compute_calibration_curve(kappa, correct, n_bins)
    ece = compute_ece(kappa, correct, n_bins)

    return {
        "status": "computed",
        **corr,
        "kappa_auroc": auroc,
        "calibration": cal,
        "ece": ece,
    }


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

def plot_kappa_distribution(
    kappa: np.ndarray, output_path: Path, **save_kw: Any
) -> List[Path]:
    setup_figure_style()
    fig = create_histogram({"kappa": kappa}, title="vMF Kappa Distribution",
                           xlabel="kappa", bins=60)
    return save_figure(fig, output_path, **save_kw)


def plot_kappa_correct_vs_incorrect(
    kappa: np.ndarray, correct: np.ndarray, output_path: Path, **save_kw: Any
) -> List[Path]:
    setup_figure_style()
    fig = create_boxplot(
        {"Correct": kappa[correct], "Incorrect": kappa[~correct]},
        title="Kappa: Correct vs Incorrect",
        ylabel="kappa",
    )
    return save_figure(fig, output_path, **save_kw)


def plot_kappa_margin_scatter(
    kappa: np.ndarray, margins: np.ndarray, output_path: Path, **save_kw: Any
) -> List[Path]:
    setup_figure_style()
    fig = create_scatter(kappa, margins, title="Kappa vs CSLS Margin",
                         xlabel="kappa", ylabel="CSLS Margin")
    return save_figure(fig, output_path, **save_kw)


def plot_calibration_diagram(
    cal: Dict[str, Any], output_path: Path, **save_kw: Any
) -> List[Path]:
    setup_figure_style()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(cal["bin_kappa"], cal["bin_accuracy"], "o-", color=PALETTE[0], label="Model")
    lo = min(min(cal["bin_kappa"]), min(cal["bin_accuracy"]))
    hi = max(max(cal["bin_kappa"]), max(cal["bin_accuracy"]))
    ax.plot([lo, hi], [lo, hi], "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean normalised kappa (bin)")
    ax.set_ylabel("Mean accuracy (bin)")
    ax.set_title("Kappa Calibration / Reliability Diagram")
    ax.legend()
    return save_figure(fig, output_path, **save_kw)


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------

def save_uncertainty_results(
    results: Dict[str, Any],
    kappa: Optional[np.ndarray],
    correct: np.ndarray,
    margins: Optional[np.ndarray],
    output_dir: Path,
    save_formats: Sequence[str] = ("png",),
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {k: v for k, v in results.items() if k != "calibration"}
    cal = results.get("calibration")
    if cal:
        metrics["calibration"] = cal
    with open(output_dir / "uncertainty_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    figs: List[Path] = []
    if kappa is not None:
        figs_dir = output_dir / "figures"
        figs += plot_kappa_distribution(kappa, figs_dir / "figure_kappa_distribution",
                                        formats=save_formats)
        if correct.any() and (~correct).any():
            figs += plot_kappa_correct_vs_incorrect(
                kappa, correct, figs_dir / "figure_kappa_correct_vs_incorrect",
                formats=save_formats,
            )
        if margins is not None:
            figs += plot_kappa_margin_scatter(
                kappa, margins, figs_dir / "figure_kappa_margin_scatter",
                formats=save_formats,
            )
        if cal and cal.get("bin_kappa"):
            figs += plot_calibration_diagram(
                cal, figs_dir / "figure_kappa_calibration", formats=save_formats,
            )

    logger.info("Uncertainty results saved to %s", output_dir)
    return {"metrics": metrics, "figures": [str(f) for f in figs]}
