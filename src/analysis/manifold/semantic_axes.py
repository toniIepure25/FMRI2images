"""Semantic axes / concept directions in CLIP space.

Semantic axes are CLIP text-space directions such as
``text("animal") - text("object")``. They are used to probe whether decoded
visual-stimulus-related fMRI embeddings preserve interpretable concept
dimensions. This is not mind reading; it is CLIP-space interpretability.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from analysis.manifold.figures import (
    PALETTE,
    create_boxplot,
    create_histogram,
    save_figure,
    setup_figure_style,
)
from analysis.manifold.utils import safe_normalize

logger = logging.getLogger(__name__)

DEFAULT_AXIS_DEFINITIONS: List[Dict[str, str]] = [
    {"name": "animal_vs_inanimate", "positive": "animal", "negative": "object"},
    {"name": "indoor_vs_outdoor", "positive": "indoor room", "negative": "outdoor landscape"},
    {"name": "face_vs_landscape", "positive": "human face", "negative": "natural scene"},
    {"name": "natural_vs_urban", "positive": "natural scene", "negative": "urban scene"},
    {"name": "vehicle_vs_animal", "positive": "vehicle", "negative": "animal"},
    {"name": "food_vs_object", "positive": "food", "negative": "object"},
    {"name": "water_vs_land", "positive": "water", "negative": "road"},
    {"name": "person_vs_empty_scene", "positive": "person", "negative": "outdoor landscape"},
    {"name": "road_vs_nature", "positive": "road", "negative": "forest"},
    {"name": "building_vs_natural_landscape", "positive": "building", "negative": "natural scene"},
]

CONCEPT_ALIASES: Dict[str, List[str]] = {
    "inanimate object": ["inanimate object", "object"],
    "object": ["object", "inanimate object"],
    "landscape": ["landscape", "natural scene", "outdoor landscape"],
    "dry land": ["dry land", "road"],
    "empty scene": ["empty scene", "outdoor landscape"],
    "indoor scene": ["indoor scene", "indoor room"],
    "outdoor scene": ["outdoor scene", "outdoor landscape"],
    "road or street": ["road or street", "road", "city street"],
    "nature": ["nature", "forest", "natural scene"],
    "natural landscape": ["natural landscape", "natural scene", "outdoor landscape"],
}

_UNAVAILABLE = {
    "status": "unavailable",
    "reason": (
        "Semantic axes require real CLIP text embeddings. Provide a text "
        "embedding cache via artifacts.text_embeddings_path or precomputed axis "
        "vectors via artifacts.semantic_axes_path. No axes were fabricated."
    ),
}


def normalize_axis_definitions(axis_definitions: Any = None) -> List[Dict[str, str]]:
    """Normalize config axis definitions to a list of name/positive/negative dicts."""
    if axis_definitions is None:
        return list(DEFAULT_AXIS_DEFINITIONS)
    if isinstance(axis_definitions, list):
        out = []
        for item in axis_definitions:
            if not isinstance(item, dict):
                continue
            if {"name", "positive", "negative"} <= set(item):
                out.append(
                    {
                        "name": str(item["name"]),
                        "positive": str(item["positive"]),
                        "negative": str(item["negative"]),
                    }
                )
        return out
    if isinstance(axis_definitions, dict):
        out = []
        for name, pair in axis_definitions.items():
            if isinstance(pair, dict):
                pos = pair.get("positive")
                neg = pair.get("negative")
            else:
                pos, neg = pair[:2] if len(pair) >= 2 else (None, None)
            if pos is not None and neg is not None:
                out.append({"name": str(name), "positive": str(pos), "negative": str(neg)})
        return out
    return list(DEFAULT_AXIS_DEFINITIONS)


def _concept_lookup(concepts: Sequence[str]) -> Dict[str, int]:
    return {c.strip().lower(): i for i, c in enumerate(concepts)}


def resolve_concept(
    requested: str,
    concepts: Sequence[str],
    aliases: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Optional[int], Optional[str], List[str]]:
    """Resolve a requested concept name to an available cache concept."""
    aliases = aliases or CONCEPT_ALIASES
    lookup = _concept_lookup(concepts)
    candidates = [requested] + aliases.get(requested.strip().lower(), [])
    seen: set[str] = set()
    unique = []
    for c in candidates:
        key = c.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    for cand in unique:
        idx = lookup.get(cand.strip().lower())
        if idx is not None:
            return idx, concepts[idx], unique
    return None, None, unique


def build_axes_from_text_embeddings(
    text_embeddings: np.ndarray,
    concepts: Sequence[str],
    axis_definitions: Any = None,
) -> Dict[str, Any]:
    """Construct normalized semantic axes from concept text embeddings."""
    if text_embeddings is None or concepts is None:
        return dict(_UNAVAILABLE)

    defs = normalize_axis_definitions(axis_definitions)
    text_embeddings = safe_normalize(text_embeddings.astype(np.float32))
    axes: List[np.ndarray] = []
    axis_names: List[str] = []
    available: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, Any]] = []

    for axis_def in defs:
        pos_idx, pos_used, pos_candidates = resolve_concept(axis_def["positive"], concepts)
        neg_idx, neg_used, neg_candidates = resolve_concept(axis_def["negative"], concepts)
        if pos_idx is None or neg_idx is None:
            unavailable.append(
                {
                    "name": axis_def["name"],
                    "positive": axis_def["positive"],
                    "negative": axis_def["negative"],
                    "reason": "missing positive or negative concept embedding",
                    "positive_candidates": pos_candidates,
                    "negative_candidates": neg_candidates,
                    "resolved_positive": pos_used,
                    "resolved_negative": neg_used,
                }
            )
            continue
        direction = text_embeddings[pos_idx] - text_embeddings[neg_idx]
        direction = safe_normalize(direction.reshape(1, -1))[0]
        if not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
            unavailable.append(
                {
                    "name": axis_def["name"],
                    "positive": axis_def["positive"],
                    "negative": axis_def["negative"],
                    "reason": "axis direction was zero or non-finite",
                    "resolved_positive": pos_used,
                    "resolved_negative": neg_used,
                }
            )
            continue
        axes.append(direction)
        axis_names.append(axis_def["name"])
        available.append(
            {
                "name": axis_def["name"],
                "positive": axis_def["positive"],
                "negative": axis_def["negative"],
                "resolved_positive": pos_used,
                "resolved_negative": neg_used,
            }
        )

    if not axes:
        return {
            "status": "unavailable",
            "reason": "No semantic axes could be built from the provided text concepts.",
            "available_axes": available,
            "unavailable_axes": unavailable,
        }
    return {
        "status": "computed",
        "axes": np.stack(axes).astype(np.float32),
        "axis_names": axis_names,
        "available_axes": available,
        "unavailable_axes": unavailable,
    }


def project_onto_axes(
    embeddings: np.ndarray,
    axes: np.ndarray,
) -> np.ndarray:
    """Project embeddings onto normalized semantic axes."""
    return embeddings @ safe_normalize(axes).T


def _safe_corr(a: np.ndarray, b: np.ndarray, method: str) -> Tuple[float, float]:
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan"), float("nan")
    if method == "pearson":
        r, p = sp_stats.pearsonr(a, b)
    elif method == "spearman":
        res = sp_stats.spearmanr(a, b)
        r, p = res.correlation, res.pvalue
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    return float(r) if np.isfinite(r) else float("nan"), float(p) if np.isfinite(p) else float("nan")


def compute_axis_preservation(
    z_pred: np.ndarray,
    z_target: np.ndarray,
    axes: np.ndarray,
    axis_names: Optional[List[str]] = None,
    retrieval_correct: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute per-axis preservation/distortion metrics."""
    axes = safe_normalize(axes.astype(np.float32))
    proj_pred = project_onto_axes(z_pred, axes)
    proj_target = project_onto_axes(z_target, axes)
    n_axes = axes.shape[0]
    if axis_names is None:
        axis_names = [f"axis_{i}" for i in range(n_axes)]

    correct = None
    if retrieval_correct is not None:
        c = np.asarray(retrieval_correct, dtype=bool)
        if c.shape[0] == z_pred.shape[0]:
            correct = c

    per_axis: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(axis_names):
        pred = proj_pred[:, i]
        target = proj_target[:, i]
        err = pred - target
        abs_err = np.abs(err)
        pearson_r, pearson_p = _safe_corr(pred, target, "pearson")
        spearman_r, spearman_p = _safe_corr(pred, target, "spearman")
        sign_agreement = float(np.mean(np.sign(pred) == np.sign(target)))
        preservation = float(
            np.nanmean(
                [
                    np.clip((pearson_r + 1.0) / 2.0, 0.0, 1.0),
                    np.clip((spearman_r + 1.0) / 2.0, 0.0, 1.0),
                    sign_agreement,
                ]
            )
        )
        row: Dict[str, float] = {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "mae": float(np.mean(abs_err)),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "bias": float(np.mean(err)),
            "sign_agreement": sign_agreement,
            "preservation_score": preservation,
        }
        if correct is not None and correct.any() and (~correct).any():
            row["mae_correct"] = float(np.mean(abs_err[correct]))
            row["mae_incorrect"] = float(np.mean(abs_err[~correct]))
        per_axis[name] = row

    pearsons = np.array([v["pearson_r"] for v in per_axis.values()], dtype=float)
    spearmans = np.array([v["spearman_r"] for v in per_axis.values()], dtype=float)
    maes = np.array([v["mae"] for v in per_axis.values()], dtype=float)
    signs = np.array([v["sign_agreement"] for v in per_axis.values()], dtype=float)
    preservations = np.array([v["preservation_score"] for v in per_axis.values()], dtype=float)

    best_axis = max(per_axis.items(), key=lambda kv: kv[1]["preservation_score"])[0]
    weakest_axis = min(per_axis.items(), key=lambda kv: kv[1]["preservation_score"])[0]

    return {
        "status": "computed",
        "axis_names": axis_names,
        "n_axes_built": int(n_axes),
        "per_axis": per_axis,
        "mean_axis_pearson": float(np.nanmean(pearsons)),
        "mean_axis_spearman": float(np.nanmean(spearmans)),
        "mean_axis_mae": float(np.nanmean(maes)),
        "mean_sign_agreement": float(np.nanmean(signs)),
        "mean_preservation": float(np.nanmean(preservations)),
        "best_axis": best_axis,
        "best_axis_preservation": float(per_axis[best_axis]["preservation_score"]),
        "weakest_axis": weakest_axis,
        "weakest_axis_preservation": float(per_axis[weakest_axis]["preservation_score"]),
        "preservation_definition": (
            "Per-axis preservation_score is the mean of normalized Pearson, "
            "normalized Spearman, and sign agreement. Correlations are mapped "
            "from [-1, 1] to [0, 1] before averaging."
        ),
        "_proj_pred": proj_pred,
        "_proj_target": proj_target,
        "_abs_errors": np.abs(proj_pred - proj_target),
        "_retrieval_correct": correct,
    }


def run_semantic_axes(
    z_pred: np.ndarray,
    z_target: np.ndarray,
    axes: Optional[np.ndarray] = None,
    axis_names: Optional[List[str]] = None,
    axis_definitions: Any = None,
    text_embeddings: Optional[np.ndarray] = None,
    text_concepts: Optional[List[str]] = None,
    retrieval_correct: Optional[np.ndarray] = None,
    max_examples: int = 12,
) -> Dict[str, Any]:
    """Top-level semantic axes entry point."""
    available_axes: List[Dict[str, Any]] = []
    unavailable_axes: List[Dict[str, Any]] = []

    if axes is None:
        built = build_axes_from_text_embeddings(text_embeddings, text_concepts or [], axis_definitions)
        if built.get("status") != "computed":
            return built
        axes = built["axes"]
        axis_names = built["axis_names"]
        available_axes = built.get("available_axes", [])
        unavailable_axes = built.get("unavailable_axes", [])

    if axis_names is None:
        axis_names = [f"axis_{i}" for i in range(axes.shape[0])]

    results = compute_axis_preservation(
        z_pred,
        z_target,
        axes,
        axis_names,
        retrieval_correct=retrieval_correct,
    )
    results["_axes"] = axes
    results["available_axes"] = available_axes
    results["unavailable_axes"] = unavailable_axes
    results["n_axes_unavailable"] = int(len(unavailable_axes))
    results["max_examples"] = int(max_examples)
    return results


def _plot_axis_preservation_bar(
    per_axis: Dict[str, Dict[str, float]],
    output_path: Path,
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    names = list(per_axis)
    x = np.arange(len(names))
    width = 0.36
    pearson = [per_axis[n]["pearson_r"] for n in names]
    spearman = [per_axis[n]["spearman_r"] for n in names]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width / 2, pearson, width, label="Pearson", color=PALETTE[0])
    ax.bar(x + width / 2, spearman, width, label="Spearman", color=PALETTE[2])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Correlation")
    ax.set_title("Semantic Axis Preservation")
    ax.legend()
    return save_figure(fig, output_path, **save_kw)


def _plot_axis_projection_scatter(
    proj_pred: np.ndarray,
    proj_target: np.ndarray,
    axis_names: List[str],
    output_path: Path,
    max_axes: int = 6,
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    n_axes = min(proj_pred.shape[1], max_axes)
    cols = min(3, n_axes)
    rows = (n_axes + cols - 1) // cols
    fig, axes_arr = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes_arr = np.atleast_1d(axes_arr).reshape(rows, cols)
    for i in range(n_axes):
        r, c = divmod(i, cols)
        ax = axes_arr[r, c]
        ax.scatter(proj_target[:, i], proj_pred[:, i], s=7, alpha=0.35,
                   color=PALETTE[i % len(PALETTE)], edgecolors="none")
        lo = min(float(proj_target[:, i].min()), float(proj_pred[:, i].min()))
        hi = max(float(proj_target[:, i].max()), float(proj_pred[:, i].max()))
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=0.8)
        ax.set_title(axis_names[i], fontsize=9)
        ax.set_xlabel("Target projection")
        ax.set_ylabel("Decoded projection")
    for i in range(n_axes, rows * cols):
        r, c = divmod(i, cols)
        axes_arr[r, c].set_visible(False)
    fig.suptitle("Semantic Axis Projections", fontsize=12)
    return save_figure(fig, output_path, **save_kw)


def _plot_radar_examples(
    proj_pred: np.ndarray,
    proj_target: np.ndarray,
    axis_names: List[str],
    output_path: Path,
    max_examples: int = 4,
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    n_examples = min(max_examples, proj_pred.shape[0])
    n_axes = len(axis_names)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]
    fig, axs = plt.subplots(
        1,
        n_examples,
        subplot_kw={"projection": "polar"},
        figsize=(3.3 * n_examples, 3.8),
    )
    axs = np.atleast_1d(axs)
    for i in range(n_examples):
        pred = np.r_[proj_pred[i], proj_pred[i, 0]]
        target = np.r_[proj_target[i], proj_target[i, 0]]
        ax = axs[i]
        ax.plot(angles, target, color=PALETTE[0], label="target", linewidth=1.8)
        ax.plot(angles, pred, color=PALETTE[1], label="decoded", linewidth=1.8)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([n.replace("_vs_", "\nvs\n") for n in axis_names], fontsize=6)
        ax.set_title(f"sample {i}", fontsize=9)
    axs[0].legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=8)
    return save_figure(fig, output_path, **save_kw)


def save_semantic_axes_results(
    results: Dict[str, Any],
    output_dir: Path,
    save_formats: Sequence[str] = ("png",),
) -> Dict[str, Any]:
    """Persist semantic axis metrics, tables, available-axis audit, and figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proj_pred = results.pop("_proj_pred", None)
    proj_target = results.pop("_proj_target", None)
    abs_errors = results.pop("_abs_errors", None)
    retrieval_correct = results.pop("_retrieval_correct", None)
    metrics = {k: v for k, v in results.items() if not k.startswith("_")}
    with open(output_dir / "semantic_axes_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    with open(output_dir / "semantic_axes_available_axes.json", "w") as f:
        json.dump(
            {
                "available_axes": metrics.get("available_axes", []),
                "unavailable_axes": metrics.get("unavailable_axes", []),
            },
            f,
            indent=2,
            default=str,
        )

    figs: List[Path] = []
    if results.get("status") != "computed":
        logger.info("Semantic axes skipped: %s", results.get("reason", "unavailable"))
        return {"metrics": metrics, "figures": []}

    axis_names = results.get("axis_names", [])
    if proj_pred is not None and proj_target is not None:
        cols: Dict[str, Any] = {"sample_index": np.arange(proj_pred.shape[0])}
        for i, name in enumerate(axis_names):
            cols[f"{name}_pred"] = proj_pred[:, i]
            cols[f"{name}_target"] = proj_target[:, i]
            cols[f"{name}_error"] = proj_pred[:, i] - proj_target[:, i]
            cols[f"{name}_abs_error"] = np.abs(proj_pred[:, i] - proj_target[:, i])
        pd.DataFrame(cols).to_csv(output_dir / "semantic_axes_per_trial.csv", index=False)

    per_axis = results.get("per_axis", {})
    if per_axis:
        rows = []
        for name, vals in per_axis.items():
            rows.append({"axis": name, **vals})
        pd.DataFrame(rows).to_csv(output_dir / "semantic_axes_axis_summary.csv", index=False)

    figs_dir = output_dir / "figures"
    if per_axis:
        figs += _plot_axis_preservation_bar(
            per_axis,
            figs_dir / "figure_semantic_axes_preservation_bar",
            formats=save_formats,
        )
    if proj_pred is not None and proj_target is not None:
        figs += _plot_axis_projection_scatter(
            proj_pred,
            proj_target,
            axis_names,
            figs_dir / "figure_semantic_axes_projection_scatter",
            formats=save_formats,
        )
        if abs_errors is not None:
            figs += save_figure(
                create_histogram(
                    {"absolute axis error": abs_errors.ravel()},
                    title="Semantic Axis Absolute Error",
                    xlabel="Absolute projection error",
                    bins=50,
                    figsize=(6.5, 3.8),
                ),
                figs_dir / "figure_semantic_axes_error_distribution",
                formats=save_formats,
            )
        if retrieval_correct is not None and abs_errors is not None:
            correct = np.asarray(retrieval_correct, dtype=bool)
            if correct.any() and (~correct).any():
                figs += save_figure(
                    create_boxplot(
                        {
                            "correct": abs_errors[correct].mean(axis=1),
                            "incorrect": abs_errors[~correct].mean(axis=1),
                        },
                        title="Semantic Axis Error: Correct vs Incorrect Retrieval",
                        ylabel="Mean absolute axis error",
                    ),
                    figs_dir / "figure_semantic_axes_correct_vs_incorrect",
                    formats=save_formats,
                )
        figs += _plot_radar_examples(
            proj_pred,
            proj_target,
            axis_names,
            figs_dir / "figure_semantic_axes_radar_examples",
            max_examples=min(int(results.get("max_examples", 12)), 4),
            formats=save_formats,
        )

    logger.info("Semantic axes results saved to %s", output_dir)
    return {"metrics": metrics, "figures": [str(f) for f in figs]}
