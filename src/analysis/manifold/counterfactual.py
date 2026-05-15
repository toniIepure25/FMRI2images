"""Counterfactual latent editing in decoded CLIP space.

These analyses perturb decoded embeddings along CLIP semantic axes and observe
changes in retrieval and text-probe readouts. They are latent-space analyses,
not brain manipulation or causal neural control.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.manifold.figures import (
    PALETTE,
    create_bar_chart,
    create_histogram,
    create_line_plot,
    save_figure,
    setup_figure_style,
)
from analysis.manifold.utils import cosine_similarity_matrix, safe_normalize

logger = logging.getLogger(__name__)

_UNAVAILABLE = {
    "status": "unavailable",
    "reason": (
        "Counterfactual latent editing requires semantic axes and CLIP text "
        "probe embeddings. Missing prerequisites are not fabricated."
    ),
}


def make_counterfactual_grid(
    z: np.ndarray,
    axis: np.ndarray,
    alphas: Sequence[float],
) -> np.ndarray:
    """Return normalized edits ``normalize(z + alpha * axis)`` for all alphas."""
    z = np.asarray(z, dtype=np.float32).reshape(1, -1)
    axis = safe_normalize(np.asarray(axis, dtype=np.float32).reshape(1, -1))
    edits = np.concatenate([z + float(alpha) * axis for alpha in alphas], axis=0)
    return safe_normalize(edits)


def retrieve_ranks(
    queries: np.ndarray,
    gallery: np.ndarray,
    target_index: Optional[int] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Retrieve top-K gallery indices and optional target ranks for queries."""
    scores = cosine_similarity_matrix(safe_normalize(queries), safe_normalize(gallery))
    order = np.argsort(-scores, axis=1)
    topk = order[:, : min(top_k, gallery.shape[0])]
    out: Dict[str, Any] = {
        "topk_indices": topk,
        "topk_scores": np.take_along_axis(scores, topk, axis=1),
    }
    if target_index is not None and 0 <= int(target_index) < gallery.shape[0]:
        target_index = int(target_index)
        ranks = np.array([int(np.where(row == target_index)[0][0]) + 1 for row in order])
        out["target_ranks"] = ranks
    return out


def text_top_concepts(
    queries: np.ndarray,
    text_embeddings: np.ndarray,
    text_concepts: Optional[Sequence[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Top text concepts for query embeddings."""
    scores = cosine_similarity_matrix(safe_normalize(queries), safe_normalize(text_embeddings))
    order = np.argsort(-scores, axis=1)[:, : min(top_k, scores.shape[1])]
    names = None
    if text_concepts:
        names = [[text_concepts[int(i)] for i in row] for row in order]
    return {
        "scores": scores,
        "top_indices": order,
        "top_scores": np.take_along_axis(scores, order, axis=1),
        "top_concepts": names,
    }


def detect_first_transition(
    alphas: Sequence[float],
    labels: Sequence[Any],
    baseline_label: Any,
) -> Optional[float]:
    """Smallest non-zero ``|alpha|`` where label differs from baseline."""
    transitions = [
        abs(float(alpha))
        for alpha, label in zip(alphas, labels)
        if float(alpha) != 0.0 and label != baseline_label
    ]
    return min(transitions) if transitions else None


def select_counterfactual_cases(
    z_pred: np.ndarray,
    z_target: Optional[np.ndarray] = None,
    correct: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
    max_cases: int = 24,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Select a compact set of cases across confidence/correctness regimes."""
    rng = np.random.RandomState(seed)
    n = z_pred.shape[0]
    max_cases = min(int(max_cases), n)
    selected: Dict[int, str] = {}

    conf = None if confidence is None else np.asarray(confidence, dtype=float)
    if conf is not None and conf.shape[0] != n:
        conf = None
    corr = None if correct is None else np.asarray(correct, dtype=bool)
    if corr is not None and corr.shape[0] != n:
        corr = None

    def add(indices: Sequence[int], label: str, limit: int) -> None:
        for idx in indices:
            if len(selected) >= max_cases:
                return
            selected.setdefault(int(idx), label)
            if sum(1 for v in selected.values() if v == label) >= limit:
                return

    per_bucket = max(1, max_cases // 5)
    if corr is not None and conf is not None:
        correct_idx = np.where(corr)[0]
        incorrect_idx = np.where(~corr)[0]
        add(correct_idx[np.argsort(-conf[correct_idx])] if len(correct_idx) else [], "high_confidence_correct", per_bucket)
        add(incorrect_idx[np.argsort(-conf[incorrect_idx])] if len(incorrect_idx) else [], "high_confidence_incorrect", per_bucket)
        add(np.argsort(conf), "low_confidence", per_bucket)
    elif conf is not None:
        add(np.argsort(-conf), "high_confidence", per_bucket)
        add(np.argsort(conf), "low_confidence", per_bucket)

    if z_target is not None:
        err = np.linalg.norm(safe_normalize(z_pred) - safe_normalize(z_target), axis=1)
        add(np.argsort(-err), "high_semantic_error", per_bucket)

    remaining = [i for i in rng.permutation(n) if int(i) not in selected]
    add(remaining, "random", max_cases)
    return [{"index": idx, "case_type": label} for idx, label in selected.items()]


def compute_counterfactual_edits(
    z_pred: np.ndarray,
    axes: Optional[np.ndarray],
    axis_names: Optional[List[str]],
    gallery: np.ndarray,
    alphas: Sequence[float] = (-2, -1, -0.5, 0, 0.5, 1, 2),
    case_indices: Optional[np.ndarray] = None,
    text_embeddings: Optional[np.ndarray] = None,
    text_concepts: Optional[List[str]] = None,
    z_target: Optional[np.ndarray] = None,
    target_indices: Optional[np.ndarray] = None,
    correct: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
    max_cases: int = 24,
    seed: int = 42,
) -> Dict[str, Any]:
    """Perform counterfactual latent edits along semantic axes."""
    if axes is None or axis_names is None or len(axis_names) == 0:
        return {**_UNAVAILABLE, "reason": "Semantic axes are unavailable."}
    if text_embeddings is None:
        return {**_UNAVAILABLE, "reason": "CLIP text probe embeddings are unavailable."}

    axes = safe_normalize(np.asarray(axes, dtype=np.float32))
    z_pred = safe_normalize(z_pred)
    gallery = safe_normalize(gallery)
    text_embeddings = safe_normalize(text_embeddings)
    alphas = [float(a) for a in alphas]

    if case_indices is None:
        cases_to_run = select_counterfactual_cases(
            z_pred,
            z_target=z_target,
            correct=correct,
            confidence=confidence,
            max_cases=max_cases,
            seed=seed,
        )
    else:
        cases_to_run = [{"index": int(i), "case_type": "provided"} for i in case_indices[:max_cases]]

    cases: List[Dict[str, Any]] = []
    edit_rows: List[Dict[str, Any]] = []
    margins: List[float] = []
    transitions_by_axis: Dict[str, int] = {name: 0 for name in axis_names}
    rank_changes_by_axis: Dict[str, List[float]] = {name: [] for name in axis_names}
    concept_transition_counts: Dict[str, int] = {}

    for case in cases_to_run:
        idx = int(case["index"])
        target_index = None
        if target_indices is not None and idx < len(target_indices):
            target_index = int(target_indices[idx])
        base_text = text_top_concepts(z_pred[idx : idx + 1], text_embeddings, text_concepts, top_k=5)
        base_concept_idx = int(base_text["top_indices"][0, 0])
        base_concept = (
            text_concepts[base_concept_idx] if text_concepts else str(base_concept_idx)
        )
        base_ret = retrieve_ranks(z_pred[idx : idx + 1], gallery, target_index=target_index, top_k=5)
        base_rank = int(base_ret["target_ranks"][0]) if "target_ranks" in base_ret else None
        case_out: Dict[str, Any] = {
            "sample_index": idx,
            "case_type": case["case_type"],
            "baseline_top_concept": base_concept,
            "baseline_target_rank": base_rank,
            "axes": {},
        }

        for ai, axis_name in enumerate(axis_names):
            grid = make_counterfactual_grid(z_pred[idx], axes[ai], alphas)
            ret = retrieve_ranks(grid, gallery, target_index=target_index, top_k=5)
            txt = text_top_concepts(grid, text_embeddings, text_concepts, top_k=5)
            top_concepts = (
                [row[0] for row in txt["top_concepts"]]
                if txt["top_concepts"] is not None
                else [str(int(row[0])) for row in txt["top_indices"]]
            )
            margin = detect_first_transition(alphas, top_concepts, base_concept)
            if margin is not None:
                margins.append(float(margin))
                transitions_by_axis[axis_name] += 1

            axis_steps = []
            for step_i, alpha in enumerate(alphas):
                target_rank = (
                    int(ret["target_ranks"][step_i]) if "target_ranks" in ret else None
                )
                rank_change = None
                if base_rank is not None and target_rank is not None:
                    rank_change = int(target_rank - base_rank)
                    if float(alpha) != 0.0:
                        rank_changes_by_axis[axis_name].append(float(rank_change))
                top_concept = top_concepts[step_i]
                if top_concept != base_concept:
                    key = f"{base_concept} -> {top_concept}"
                    concept_transition_counts[key] = concept_transition_counts.get(key, 0) + 1
                row = {
                    "sample_index": idx,
                    "case_type": case["case_type"],
                    "axis": axis_name,
                    "alpha": float(alpha),
                    "top1_gallery_idx": int(ret["topk_indices"][step_i, 0]),
                    "top5_gallery_indices": ret["topk_indices"][step_i].tolist(),
                    "target_rank": target_rank,
                    "target_rank_change": rank_change,
                    "top_concept": top_concept,
                    "top_concept_changed": bool(top_concept != base_concept),
                    "semantic_robustness_margin": margin,
                }
                axis_steps.append(row)
                edit_rows.append(row)
            case_out["axes"][axis_name] = {
                "semantic_robustness_margin": margin,
                "steps": axis_steps,
            }
        cases.append(case_out)

    n_case_axis = max(len(cases_to_run) * len(axis_names), 1)
    margins_arr = np.asarray(margins, dtype=float)
    mean_rank_change_by_axis = {
        name: float(np.mean(vals)) if vals else None
        for name, vals in rank_changes_by_axis.items()
    }
    axis_sensitivity_ranked = sorted(
        [
            {
                "axis": name,
                "transition_rate": transitions_by_axis[name] / max(len(cases_to_run), 1),
                "mean_rank_change": mean_rank_change_by_axis[name],
            }
            for name in axis_names
        ],
        key=lambda row: (row["transition_rate"], abs(row["mean_rank_change"] or 0.0)),
        reverse=True,
    )

    return {
        "status": "computed",
        "n_cases": len(cases_to_run),
        "n_axes": len(axis_names),
        "alphas": alphas,
        "robustness_margin_mean": float(margins_arr.mean()) if len(margins_arr) else float("nan"),
        "robustness_margin_median": float(np.median(margins_arr)) if len(margins_arr) else float("nan"),
        "transition_rate": float(len(margins_arr) / n_case_axis),
        "n_transitions_observed": int(len(margins_arr)),
        "mean_rank_change_by_axis": mean_rank_change_by_axis,
        "concept_transition_counts": concept_transition_counts,
        "axis_sensitivity_ranked": axis_sensitivity_ranked,
        "cases": cases,
        "_per_edit_rows": edit_rows,
        "_robustness_margins": margins_arr,
    }


def _plot_alpha_concept_paths(rows: List[Dict[str, Any]], output_path: Path, **save_kw: Any) -> List[Path]:
    setup_figure_style()
    if not rows:
        return []
    df = pd.DataFrame(rows)
    subset = df.groupby(["sample_index", "axis"]).head(999)
    groups = list(subset.groupby(["sample_index", "axis"]))[:6]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for gi, ((sample, axis), g) in enumerate(groups):
        changed = g["top_concept_changed"].astype(int).to_numpy()
        ax.plot(g["alpha"], changed + gi * 1.25, marker="o", label=f"{sample}:{axis}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("concept changed path")
    ax.set_yticks([])
    ax.set_title("Counterfactual Alpha Concept Paths")
    ax.legend(fontsize=7, ncol=2)
    return save_figure(fig, output_path, **save_kw)


def _plot_axis_sensitivity(axis_rows: List[Dict[str, Any]], output_path: Path, **save_kw: Any) -> List[Path]:
    if not axis_rows:
        return []
    data = {row["axis"]: row["transition_rate"] for row in axis_rows}
    fig = create_bar_chart(data, title="Counterfactual Axis Sensitivity", ylabel="Transition rate")
    return save_figure(fig, output_path, **save_kw)


def _plot_case_grid(cases: List[Dict[str, Any]], output_path: Path, **save_kw: Any) -> List[Path]:
    setup_figure_style()
    rows = []
    for case in cases[:8]:
        axis_summaries = []
        for axis, data in list(case.get("axes", {}).items())[:3]:
            axis_summaries.append(f"{axis}: {data.get('semantic_robustness_margin')}")
        rows.append([
            case["sample_index"],
            case["case_type"],
            case.get("baseline_top_concept"),
            "; ".join(axis_summaries),
        ])
    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(rows) + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["idx", "case type", "baseline concept", "axis margins"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    ax.set_title("Counterfactual Case Grid")
    return save_figure(fig, output_path, **save_kw)


def save_counterfactual_results(
    results: Dict[str, Any],
    output_dir: Path,
    save_formats: Sequence[str] = ("png",),
) -> Dict[str, Any]:
    """Persist counterfactual metrics, cases, per-edit table, and figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    margins = results.pop("_robustness_margins", np.array([]))
    rows = results.pop("_per_edit_rows", [])
    cases = results.get("cases", [])
    metrics = {k: v for k, v in results.items() if not k.startswith("_") and k != "cases"}
    with open(output_dir / "counterfactual_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(output_dir / "counterfactual_cases.json", "w") as f:
        json.dump(cases, f, indent=2, default=str)
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "counterfactual_per_edit.csv", index=False)

    figs: List[Path] = []
    if results.get("status") == "computed":
        figs_dir = output_dir / "figures"
        figs += _plot_alpha_concept_paths(
            rows,
            figs_dir / "figure_counterfactual_alpha_concept_paths",
            formats=save_formats,
        )
        if len(margins) > 0:
            figs += save_figure(
                create_histogram(
                    {"Robustness margin": margins},
                    title="Counterfactual Semantic Robustness Margin",
                    xlabel="minimum |alpha| changing top concept",
                    bins=20,
                ),
                figs_dir / "figure_counterfactual_robustness_margin",
                formats=save_formats,
            )
        figs += _plot_axis_sensitivity(
            results.get("axis_sensitivity_ranked", []),
            figs_dir / "figure_counterfactual_axis_sensitivity",
            formats=save_formats,
        )
        figs += _plot_case_grid(
            cases,
            figs_dir / "figure_counterfactual_case_grid",
            formats=save_formats,
        )

    logger.info("Counterfactual results saved to %s", output_dir)
    return {"metrics": metrics, "figures": [str(f) for f in figs]}
