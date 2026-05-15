"""ROI-to-semantic-axis mapping via ablation (optional advanced module).

Scientific claim tested:
    "Cortical ROI ablations produce structured perturbations along CLIP
     semantic axes, suggesting that the decoder can probe visual-semantic
     organisation."

This module provides the interface and stub implementation.  Full execution
requires a trained model, per-ROI voxel indices, and the ability to run
forward passes with ablated inputs — resources that may not be available
at analysis time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from analysis.manifold.figures import (
    create_heatmap,
    save_figure,
    setup_figure_style,
)

logger = logging.getLogger(__name__)

_UNAVAILABLE = {
    "status": "unavailable",
    "reason": (
        "ROI ablation analysis requires:\n"
        "  1. A trained model checkpoint loadable via UnifiedModel\n"
        "  2. ROI voxel index mapping (from build_roi_index)\n"
        "  3. Raw or pre-extracted fMRI features for the subject\n"
        "  4. Semantic axis vectors\n"
        "Provide roi_masks_path in the config and implement the "
        "forward-pass hook to enable this module."
    ),
}

ROI_NAMES = [
    "V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "V3A", "V3B", "V4",
    "FFA1", "FFA2", "PPA", "EBA", "OFA", "OPA", "RSC", "nsdgeneral_other",
]


class ROIAblationAnalyzer:
    """Placeholder for ROI ablation analysis.

    Subclass or monkey-patch ``_forward`` to supply actual model inference.
    """

    def __init__(
        self,
        roi_index: Optional[Dict[str, np.ndarray]] = None,
        axes: Optional[np.ndarray] = None,
        axis_names: Optional[List[str]] = None,
    ):
        self.roi_index = roi_index
        self.axes = axes
        self.axis_names = axis_names or []

    def _forward(self, fmri: np.ndarray) -> np.ndarray:
        """Run model forward pass; override in subclass."""
        raise NotImplementedError(
            "ROIAblationAnalyzer._forward must be overridden with actual "
            "model inference logic."
        )

    def compute_roi_axis_contributions(
        self,
        fmri: np.ndarray,
        sample_indices: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Ablate each ROI and measure delta along semantic axes.

        For each ROI:
            z_full = forward(fmri)
            z_ablated = forward(fmri with ROI zeroed)
            delta_z = z_full - z_ablated
            contribution(ROI, axis) = dot(delta_z, axis)

        Returns:
            Dict with ROI x axis contribution matrix and retrieval drops.
        """
        if self.roi_index is None or self.axes is None:
            return dict(_UNAVAILABLE)

        if sample_indices is not None:
            fmri = fmri[sample_indices]

        try:
            z_full = self._forward(fmri)
        except NotImplementedError:
            return dict(_UNAVAILABLE)

        rois = list(self.roi_index.keys())
        n_rois = len(rois)
        n_axes = self.axes.shape[0]
        contributions = np.zeros((n_rois, n_axes))

        for ri, roi_name in enumerate(rois):
            mask = self.roi_index[roi_name]
            fmri_ablated = fmri.copy()
            fmri_ablated[:, mask] = 0.0
            z_ablated = self._forward(fmri_ablated)
            delta = z_full - z_ablated
            mean_delta = delta.mean(axis=0)
            for ai in range(n_axes):
                contributions[ri, ai] = float(np.dot(mean_delta, self.axes[ai]))

        return {
            "status": "computed",
            "roi_names": rois,
            "axis_names": self.axis_names,
            "contributions": contributions.tolist(),
            "_contributions_matrix": contributions,
        }


def run_roi_axes(
    roi_index: Optional[Dict[str, np.ndarray]] = None,
    axes: Optional[np.ndarray] = None,
    axis_names: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Top-level ROI axes entry point — returns unavailable unless wired."""
    if roi_index is None or axes is None:
        return dict(_UNAVAILABLE)
    analyzer = ROIAblationAnalyzer(roi_index, axes, axis_names)
    return dict(_UNAVAILABLE)


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

def plot_roi_axis_heatmap(
    contributions: np.ndarray,
    roi_names: List[str],
    axis_names: List[str],
    output_path: Path,
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(6, len(axis_names) * 0.8), max(5, len(roi_names) * 0.4)))
    im = ax.imshow(contributions, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(axis_names)))
    ax.set_xticklabels(axis_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(roi_names)))
    ax.set_yticklabels(roi_names, fontsize=8)
    ax.set_title("ROI → Semantic Axis Contributions")
    fig.colorbar(im, ax=ax, shrink=0.7)
    return save_figure(fig, output_path, **save_kw)


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------

def save_roi_axes_results(
    results: Dict[str, Any],
    output_dir: Path,
    save_formats: Sequence[str] = ("png",),
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contrib_matrix = results.pop("_contributions_matrix", None)
    metrics = {k: v for k, v in results.items() if not k.startswith("_")}
    with open(output_dir / "roi_axis_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    figs: List[Path] = []
    if contrib_matrix is not None:
        roi_names = results.get("roi_names", [])
        axis_names = results.get("axis_names", [])
        figs_dir = output_dir / "figures"
        figs += plot_roi_axis_heatmap(
            contrib_matrix, roi_names, axis_names,
            figs_dir / "figure_roi_axis_heatmap", formats=save_formats,
        )

    logger.info("ROI axes results saved to %s", output_dir)
    return {"metrics": metrics, "figures": [str(f) for f in figs]}
