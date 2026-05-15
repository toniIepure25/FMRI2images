"""Neural-Manifold Alignment Score (NMAS).

Scientific claim tested:
    "R@1 alone is insufficient; NMAS captures complementary aspects of
     neural-semantic alignment."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from analysis.manifold.figures import (
    create_bar_chart,
    create_radar_chart,
    save_figure,
    setup_figure_style,
)
from analysis.manifold.utils import weighted_harmonic_mean

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: Dict[str, float] = {
    "retrieval": 1.0,
    "neighborhood": 1.0,
    "rsa": 1.0,
    "repeat_stability": 1.0,
    "on_manifold": 1.0,
    "calibration": 1.0,
}


def compute_nmas(
    components: Dict[str, Optional[float]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute aggregate Neural-Manifold Alignment Score.

    Args:
        components: name -> score in (0, 1].  ``None`` for unavailable.
        weights: Per-component weights (defaults to uniform).

    Returns:
        Dict with ``nmas`` score, included/excluded components, and
        leave-one-out sensitivity analysis.
    """
    if weights is None:
        weights = dict(DEFAULT_WEIGHTS)

    valid = {k: v for k, v in components.items() if v is not None and np.isfinite(v) and v > 0}
    excluded = [k for k in components if k not in valid]

    nmas, used = weighted_harmonic_mean(valid, weights)

    sensitivity: Dict[str, float] = {}
    for drop in valid:
        sub = {k: v for k, v in valid.items() if k != drop}
        s, _ = weighted_harmonic_mean(sub, weights)
        sensitivity[drop] = nmas - s

    return {
        "nmas": nmas,
        "components": {k: float(v) for k, v in valid.items()},
        "weights_used": used,
        "excluded": excluded,
        "sensitivity": sensitivity,
    }


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

def plot_nmas_radar(
    components: Dict[str, float],
    output_path: Path,
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    fig = create_radar_chart(components, title="NMAS Components")
    return save_figure(fig, output_path, **save_kw)


def plot_nmas_bar(
    components: Dict[str, float],
    nmas: float,
    output_path: Path,
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    data = dict(components)
    data["NMAS (aggregate)"] = nmas
    fig = create_bar_chart(data, title="NMAS Component Breakdown", ylabel="Score")
    return save_figure(fig, output_path, **save_kw)


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------

def save_nmas_results(
    results: Dict[str, Any],
    output_dir: Path,
    save_formats: Sequence[str] = ("png",),
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "nmas_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    figs: List[Path] = []
    figs_dir = output_dir / "figures"
    comps = results.get("components", {})
    if comps:
        figs += plot_nmas_radar(comps, figs_dir / "figure_nmas_radar", formats=save_formats)
        figs += plot_nmas_bar(comps, results.get("nmas", 0),
                              figs_dir / "figure_nmas_components", formats=save_formats)

    logger.info("NMAS results saved to %s", output_dir)
    return {"metrics": results, "figures": [str(f) for f in figs]}
