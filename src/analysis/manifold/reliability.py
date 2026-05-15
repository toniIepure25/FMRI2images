"""Reliability-aware selective decoding.

Scientific claim tested:
    "The decoder knows when it knows: reliability-aware selective decoding
     improves accuracy at lower coverage."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from analysis.manifold.figures import (
    create_bar_chart,
    create_boxplot,
    create_line_plot,
    save_figure,
    setup_figure_style,
)
from analysis.manifold.utils import bootstrap_ci, percentile_normalize, zscore_normalize

logger = logging.getLogger(__name__)


class ReliabilityScorer:
    """Composite reliability score from heterogeneous signal arrays.

    Each signal is z-scored or percentile-normalised, weighted, and
    summed.  Missing signals are silently excluded, and the report lists
    which components were used.
    """

    def __init__(
        self,
        weights: Dict[str, float],
        norm_method: str = "percentile",
    ):
        self.weights = dict(weights)
        self.norm_method = norm_method

    def compute(
        self,
        signals: Dict[str, Optional[np.ndarray]],
    ) -> tuple[np.ndarray, Dict[str, float]]:
        """Compute composite reliability score.

        Args:
            signals: name -> (N,) array.  ``None`` entries are skipped.

        Returns:
            (N,) composite score and dict of actually-used weights.
        """
        norm_fn = percentile_normalize if self.norm_method == "percentile" else zscore_normalize
        used: Dict[str, float] = {}
        components: list[np.ndarray] = []
        n = None

        for name, arr in signals.items():
            if arr is None:
                continue
            w = self.weights.get(name, 0.0)
            if w == 0.0:
                continue
            if n is None:
                n = len(arr)
            if len(arr) != n:
                logger.warning("Signal '%s' length %d != %d; skipping", name, len(arr), n)
                continue
            normed = norm_fn(arr.astype(np.float64))
            components.append(w * normed)
            used[name] = w

        if not components:
            logger.warning("No valid signals for reliability score")
            return np.zeros(1), {}

        total_w = sum(used.values())
        score = sum(components) / total_w
        return score, used


def compute_selective_prediction(
    scores: np.ndarray,
    correct: np.ndarray,
    coverages: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
    n_bootstrap: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Accuracy at various coverage levels, sorted by reliability descending."""
    order = np.argsort(-scores)
    n = len(scores)
    result: Dict[str, Any] = {
        "coverage": [], "accuracy": [], "n_selected": [],
        "accuracy_ci_low": [], "accuracy_ci_high": [],
    }
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        sel = order[:k]
        acc = float(correct[sel].mean())
        result["coverage"].append(float(cov))
        result["accuracy"].append(acc)
        result["n_selected"].append(k)
        if n_bootstrap > 0:
            lo, hi = bootstrap_ci(
                correct[sel].astype(np.float64), np.mean,
                n_bootstrap=n_bootstrap, seed=seed
            )
        else:
            lo, hi = (None, None)
        result["accuracy_ci_low"].append(lo)
        result["accuracy_ci_high"].append(hi)
    return result


def compute_risk_coverage_curve(
    scores: np.ndarray,
    errors: np.ndarray,
    n_points: int = 100,
) -> Dict[str, Any]:
    """Risk-coverage curve and AURC."""
    order = np.argsort(-scores)
    n = len(scores)
    coverages = np.linspace(1.0 / n, 1.0, n_points)
    risks = []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        risks.append(float(errors[order[:k]].mean()))
    risks_arr = np.array(risks)
    aurc = float(np.trapz(risks_arr, coverages))
    return {"coverage": coverages.tolist(), "risk": risks_arr.tolist(), "aurc": aurc}


def run_reliability_analysis(
    correct: np.ndarray,
    kappa: Optional[np.ndarray] = None,
    csls_margin: Optional[np.ndarray] = None,
    manifold_density: Optional[np.ndarray] = None,
    topk_entropy: Optional[np.ndarray] = None,
    hubness_score: Optional[np.ndarray] = None,
    semantic_probe_agreement: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None,
    coverages: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
    n_bootstrap: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Top-level reliability analysis entry point."""
    if weights is None:
        weights = {
            "kappa": 1.0, "csls_margin": 1.0, "manifold_density": 1.0,
            "topk_entropy": -1.0, "hubness_score": -0.5,
            "semantic_probe_agreement": 0.5,
        }

    signals: Dict[str, Optional[np.ndarray]] = {
        "kappa": kappa,
        "csls_margin": csls_margin,
        "manifold_density": manifold_density,
        "topk_entropy": topk_entropy,
        "hubness_score": hubness_score,
        "semantic_probe_agreement": semantic_probe_agreement,
    }

    scorer = ReliabilityScorer(weights)
    score, used_weights = scorer.compute(signals)

    sel = compute_selective_prediction(
        score, correct, coverages, n_bootstrap=n_bootstrap, seed=seed
    )
    errors = (~correct).astype(np.float64)
    rc = compute_risk_coverage_curve(score, errors)

    metrics: Dict[str, Any] = {
        "status": "computed",
        "used_components": list(used_weights.keys()),
        "used_weights": used_weights,
        "selective_prediction": sel,
        "aurc": rc["aurc"],
    }
    return {**metrics, "_score": score, "_risk_coverage": rc}


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

def plot_coverage_accuracy(
    sel: Dict[str, Any], output_path: Path, **save_kw: Any
) -> List[Path]:
    setup_figure_style()
    fig = create_line_plot(
        {"Selective R@1": (np.array(sel["coverage"]), np.array(sel["accuracy"]))},
        title="Coverage vs Accuracy (Selective Decoding)",
        xlabel="Coverage", ylabel="Accuracy (R@1)",
    )
    return save_figure(fig, output_path, **save_kw)


def plot_reliability_distribution(
    score: np.ndarray, correct: np.ndarray, output_path: Path, **save_kw: Any
) -> List[Path]:
    setup_figure_style()
    fig = create_boxplot(
        {"Correct": score[correct], "Incorrect": score[~correct]},
        title="Reliability Score: Correct vs Incorrect",
        ylabel="Reliability score",
    )
    return save_figure(fig, output_path, **save_kw)


def plot_component_importance(
    used_weights: Dict[str, float], output_path: Path, **save_kw: Any
) -> List[Path]:
    setup_figure_style()
    fig = create_bar_chart(used_weights, title="Reliability Component Weights",
                           ylabel="Weight")
    return save_figure(fig, output_path, **save_kw)


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------

def save_reliability_results(
    results: Dict[str, Any],
    correct: np.ndarray,
    output_dir: Path,
    save_formats: Sequence[str] = ("png",),
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score = results.pop("_score", None)
    rc = results.pop("_risk_coverage", None)
    metrics = {k: v for k, v in results.items() if not k.startswith("_")}
    with open(output_dir / "reliability_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    if score is not None:
        import pandas as pd
        pd.DataFrame({"reliability_score": score}).to_csv(
            output_dir / "reliability_per_trial.csv", index=False
        )

    figs: List[Path] = []
    figs_dir = output_dir / "figures"
    sel = metrics.get("selective_prediction")
    if sel:
        figs += plot_coverage_accuracy(sel, figs_dir / "figure_coverage_accuracy",
                                       formats=save_formats)
    if score is not None and correct.any() and (~correct).any():
        figs += plot_reliability_distribution(
            score, correct, figs_dir / "figure_reliability_correct_vs_incorrect",
            formats=save_formats,
        )
    used = metrics.get("used_weights", {})
    if used:
        figs += plot_component_importance(
            used, figs_dir / "figure_reliability_components",
            formats=save_formats,
        )

    logger.info("Reliability results saved to %s", output_dir)
    return {"metrics": metrics, "figures": [str(f) for f in figs]}
