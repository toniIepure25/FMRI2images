"""CLIP text probe / neural semantic probe analysis.

This module interprets decoded visual-stimulus-related fMRI embeddings by
measuring where they fall relative to a fixed library of CLIP text concepts.
It does not decode thoughts, dreams, or consciousness.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.special import rel_entr, softmax

from analysis.manifold.figures import (
    PALETTE,
    create_bar_chart,
    create_boxplot,
    create_histogram,
    create_scatter,
    save_figure,
    setup_figure_style,
)
from analysis.manifold.utils import cosine_similarity_matrix, safe_normalize

logger = logging.getLogger(__name__)

_EPS = 1e-12

_UNAVAILABLE = {
    "status": "unavailable",
    "reason": (
        "Semantic probes require a real CLIP text embedding cache. Build one "
        "with scripts/build/build_text_probe_embeddings.py and set "
        "artifacts.text_embeddings_path. No text embeddings were fabricated."
    ),
}


def compute_text_probe_scores(
    embeddings: np.ndarray,
    concept_embeddings: np.ndarray,
) -> np.ndarray:
    """Return cosine scores between image/brain embeddings and text concepts."""
    return cosine_similarity_matrix(
        safe_normalize(embeddings),
        safe_normalize(concept_embeddings),
    )


def scores_to_probabilities(scores: np.ndarray, temperature: float = 0.07) -> np.ndarray:
    """Convert text-probe scores to concept distributions."""
    if temperature <= 0:
        raise ValueError("semantic probe temperature must be positive")
    return softmax(scores / temperature, axis=1)


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), scores.shape[1])
    return np.argsort(-scores, axis=1)[:, :k]


def _safe_corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    if method == "spearman":
        val = sp_stats.spearmanr(a, b).correlation
    elif method == "pearson":
        val = sp_stats.pearsonr(a, b)[0]
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    return float(val) if np.isfinite(val) else float("nan")


def _entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(probs + _EPS), axis=1)


def compute_probe_agreement(
    pred_scores: np.ndarray,
    target_scores: np.ndarray,
    pred_probs: np.ndarray,
    target_probs: np.ndarray,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, Any]:
    """Compute text-probe agreement and distributional alignment metrics.

    ``agreement@K`` is the fraction of samples where the target embedding's
    top-1 concept appears in the decoded embedding's top-K concept list.
    """
    n, n_concepts = pred_scores.shape
    if target_scores.shape != pred_scores.shape:
        raise ValueError("pred_scores and target_scores must have the same shape")

    pred_order = np.argsort(-pred_scores, axis=1)
    target_order = np.argsort(-target_scores, axis=1)
    pred_top1 = pred_order[:, 0]
    target_top1 = target_order[:, 0]

    target_top1_rank = np.empty(n, dtype=np.int64)
    for i in range(n):
        target_top1_rank[i] = int(np.where(pred_order[i] == target_top1[i])[0][0]) + 1

    results: Dict[str, Any] = {
        "top1_agreement": float(np.mean(pred_top1 == target_top1)),
        "target_top_concept_rank_mean": float(target_top1_rank.mean()),
        "target_top_concept_rank_median": float(np.median(target_top1_rank)),
    }

    per_trial_agreement: Dict[int, np.ndarray] = {}
    per_trial_overlap: Dict[int, np.ndarray] = {}
    for k in ks:
        k_ = min(int(k), n_concepts)
        agree = np.array(
            [target_top1[i] in pred_order[i, :k_] for i in range(n)],
            dtype=bool,
        )
        overlap = np.array(
            [
                len(set(pred_order[i, :k_]) & set(target_order[i, :k_])) / k_
                for i in range(n)
            ],
            dtype=np.float64,
        )
        results[f"agreement@{k}"] = float(agree.mean())
        results[f"concept_overlap@{k}"] = float(overlap.mean())
        per_trial_agreement[int(k)] = agree
        per_trial_overlap[int(k)] = overlap

    pred_safe = np.clip(pred_probs, _EPS, 1.0)
    target_safe = np.clip(target_probs, _EPS, 1.0)
    midpoint = 0.5 * (pred_safe + target_safe)
    kl_divs = np.sum(rel_entr(target_safe, pred_safe), axis=1)
    js_divs = 0.5 * np.sum(rel_entr(target_safe, midpoint), axis=1) + 0.5 * np.sum(
        rel_entr(pred_safe, midpoint), axis=1
    )

    spearman = np.array(
        [_safe_corr(pred_scores[i], target_scores[i], "spearman") for i in range(n)],
        dtype=np.float64,
    )
    pearson = np.array(
        [_safe_corr(pred_scores[i], target_scores[i], "pearson") for i in range(n)],
        dtype=np.float64,
    )

    results.update(
        {
            "score_vector_spearman": float(np.nanmean(spearman)),
            "score_vector_spearman_median": float(np.nanmedian(spearman)),
            "score_vector_pearson": float(np.nanmean(pearson)),
            "score_vector_pearson_median": float(np.nanmedian(pearson)),
            "js_divergence_mean": float(np.mean(js_divs)),
            "js_divergence_median": float(np.median(js_divs)),
            "kl_divergence_mean": float(np.mean(kl_divs)),
            "kl_divergence_median": float(np.median(kl_divs)),
            "pred_entropy_mean": float(np.mean(_entropy(pred_probs))),
            "target_entropy_mean": float(np.mean(_entropy(target_probs))),
            "_per_trial_agreement": per_trial_agreement,
            "_per_trial_overlap": per_trial_overlap,
            "_per_trial_spearman": spearman,
            "_per_trial_pearson": pearson,
            "_per_trial_js": js_divs,
            "_per_trial_kl": kl_divs,
            "_target_top1_rank": target_top1_rank,
            "_pred_order": pred_order,
            "_target_order": target_order,
            "_pred_top1": pred_top1,
            "_target_top1": target_top1,
        }
    )
    return results


def _top_concepts(
    order: np.ndarray,
    scores: np.ndarray,
    concepts: Sequence[str],
    row: int,
    k: int = 5,
) -> List[Dict[str, Any]]:
    out = []
    for idx in order[row, : min(k, len(concepts))]:
        out.append({"concept": concepts[int(idx)], "score": float(scores[row, int(idx)])})
    return out


def run_semantic_probes(
    z_pred: np.ndarray,
    z_target: np.ndarray,
    text_embeddings: Optional[np.ndarray] = None,
    concepts: Optional[List[str]] = None,
    temperature: float = 0.07,
    agreement_ks: Sequence[int] = (1, 3, 5, 10),
    retrieval_correct: Optional[np.ndarray] = None,
    max_examples: int = 12,
    text_cache_metadata: Optional[Dict[str, Any]] = None,
    text_embeddings_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run semantic probe analysis against precomputed CLIP text embeddings."""
    if text_embeddings is None:
        res = dict(_UNAVAILABLE)
        if text_embeddings_path:
            res["text_embeddings_path"] = text_embeddings_path
        return res

    text_embeddings = safe_normalize(text_embeddings.astype(np.float32))
    if concepts is None:
        concepts = [f"concept_{i}" for i in range(text_embeddings.shape[0])]
    if len(concepts) != text_embeddings.shape[0]:
        raise ValueError(
            f"Number of concepts ({len(concepts)}) must match text embeddings "
            f"({text_embeddings.shape[0]})."
        )

    pred_scores = compute_text_probe_scores(z_pred, text_embeddings)
    target_scores = compute_text_probe_scores(z_target, text_embeddings)
    pred_probs = scores_to_probabilities(pred_scores, temperature)
    target_probs = scores_to_probabilities(target_scores, temperature)

    agreement = compute_probe_agreement(
        pred_scores,
        target_scores,
        pred_probs,
        target_probs,
        ks=agreement_ks,
    )

    metrics: Dict[str, Any] = {
        "status": "computed",
        "n_samples": int(z_pred.shape[0]),
        "n_concepts": int(len(concepts)),
        "concepts": list(concepts),
        "temperature": float(temperature),
        "text_embeddings_path": text_embeddings_path,
        "text_cache": text_cache_metadata or {},
        **agreement,
        "_pred_scores": pred_scores,
        "_target_scores": target_scores,
        "_pred_probs": pred_probs,
        "_target_probs": target_probs,
    }

    if retrieval_correct is not None:
        correct = np.asarray(retrieval_correct, dtype=bool)
        if correct.shape[0] == z_pred.shape[0] and correct.any() and (~correct).any():
            metrics["_retrieval_correct"] = correct
            for name, values in [
                ("agreement@5", agreement.get("_per_trial_agreement", {}).get(5)),
                ("score_spearman", agreement.get("_per_trial_spearman")),
                ("js_divergence", agreement.get("_per_trial_js")),
            ]:
                if values is None:
                    continue
                arr = np.asarray(values, dtype=float)
                metrics[f"{name}_correct_mean"] = float(np.nanmean(arr[correct]))
                metrics[f"{name}_incorrect_mean"] = float(np.nanmean(arr[~correct]))
            metrics["correct_vs_incorrect_status"] = "computed"
        else:
            metrics["correct_vs_incorrect_status"] = "unavailable"
            metrics["correct_vs_incorrect_reason"] = (
                "Retrieval correctness was missing, all-correct, all-incorrect, "
                "or shape-incompatible."
            )
    else:
        metrics["correct_vs_incorrect_status"] = "unavailable"
        metrics["correct_vs_incorrect_reason"] = "Retrieval correctness unavailable."

    pred_order = agreement["_pred_order"]
    target_order = agreement["_target_order"]
    n_examples = min(int(max_examples), z_pred.shape[0])
    examples = []
    for i in range(n_examples):
        examples.append(
            {
                "sample_index": int(i),
                "predicted_top_concepts": _top_concepts(pred_order, pred_scores, concepts, i, 5),
                "target_top_concepts": _top_concepts(target_order, target_scores, concepts, i, 5),
                "target_top_concept_rank_in_prediction": int(
                    agreement["_target_top1_rank"][i]
                ),
                "js_divergence": float(agreement["_per_trial_js"][i]),
                "score_vector_spearman": float(agreement["_per_trial_spearman"][i]),
            }
        )
    metrics["_examples"] = examples
    return metrics


def _json_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for k, v in results.items():
        if k.startswith("_"):
            continue
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, dict) and k != "text_cache":
            continue
        metrics[k] = v
    return metrics


def _plot_examples(examples: List[Dict[str, Any]], output_path: Path, **save_kw: Any) -> List[Path]:
    setup_figure_style()
    rows = []
    for ex in examples[: min(len(examples), 8)]:
        rows.append(
            [
                ex["sample_index"],
                ", ".join(c["concept"] for c in ex["predicted_top_concepts"][:3]),
                ", ".join(c["concept"] for c in ex["target_top_concepts"][:3]),
                ex["target_top_concept_rank_in_prediction"],
            ]
        )
    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(rows) + 1.4)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["idx", "decoded top concepts", "target top concepts", "target rank"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    ax.set_title("Semantic Probe Examples")
    return save_figure(fig, output_path, **save_kw)


def _plot_concept_confusion(
    pred_top1: np.ndarray,
    target_top1: np.ndarray,
    concepts: Sequence[str],
    output_path: Path,
    **save_kw: Any,
) -> List[Path]:
    setup_figure_style()
    n_c = len(concepts)
    conf = np.zeros((n_c, n_c), dtype=np.float64)
    for p, t in zip(pred_top1, target_top1):
        conf[int(t), int(p)] += 1
    row_sums = conf.sum(axis=1, keepdims=True)
    conf_norm = np.divide(conf, row_sums, out=np.zeros_like(conf), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(conf_norm, cmap="Blues", vmin=0, vmax=1)
    trunc = [c[:14] for c in concepts]
    ax.set_xticks(range(n_c))
    ax.set_yticks(range(n_c))
    ax.set_xticklabels(trunc, rotation=90, fontsize=6)
    ax.set_yticklabels(trunc, fontsize=6)
    ax.set_xlabel("Decoded embedding top-1 text concept")
    ax.set_ylabel("Target embedding top-1 text concept")
    ax.set_title("Semantic Concept Confusion")
    fig.colorbar(im, ax=ax, shrink=0.7)
    return save_figure(fig, output_path, **save_kw)


def save_semantic_probe_results(
    results: Dict[str, Any],
    output_dir: Path,
    save_formats: Sequence[str] = ("png",),
) -> Dict[str, Any]:
    """Persist semantic probe metrics, per-trial tables, examples, and figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = output_dir / "figures"

    metrics = _json_metrics(results)
    with open(output_dir / "semantic_probe_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    figs: List[Path] = []
    if results.get("status") != "computed":
        logger.info("Semantic probe skipped: %s", results.get("reason", "unavailable"))
        return {"metrics": metrics, "figures": []}

    concepts = list(results["concepts"])
    pred_scores = results["_pred_scores"]
    target_scores = results["_target_scores"]
    pred_probs = results["_pred_probs"]
    target_probs = results["_target_probs"]
    pred_order = results["_pred_order"]
    target_order = results["_target_order"]
    pred_top1 = results["_pred_top1"]
    target_top1 = results["_target_top1"]

    per_trial: Dict[str, Any] = {
        "sample_index": np.arange(pred_scores.shape[0]),
        "pred_top1_concept": [concepts[int(i)] for i in pred_top1],
        "target_top1_concept": [concepts[int(i)] for i in target_top1],
        "target_top_concept_rank_in_prediction": results["_target_top1_rank"],
        "score_vector_spearman": results["_per_trial_spearman"],
        "score_vector_pearson": results["_per_trial_pearson"],
        "js_divergence": results["_per_trial_js"],
        "kl_divergence": results["_per_trial_kl"],
        "pred_entropy": _entropy(pred_probs),
        "target_entropy": _entropy(target_probs),
    }
    for k, arr in results["_per_trial_agreement"].items():
        per_trial[f"agreement@{k}"] = arr.astype(int)
    for k, arr in results["_per_trial_overlap"].items():
        per_trial[f"concept_overlap@{k}"] = arr
    pd.DataFrame(per_trial).to_csv(output_dir / "semantic_probe_per_trial.csv", index=False)

    top_rows = []
    for idx, concept in enumerate(concepts):
        top_rows.append(
            {
                "concept": concept,
                "pred_top1_count": int(np.sum(pred_top1 == idx)),
                "target_top1_count": int(np.sum(target_top1 == idx)),
                "pred_mean_score": float(np.mean(pred_scores[:, idx])),
                "target_mean_score": float(np.mean(target_scores[:, idx])),
                "pred_mean_probability": float(np.mean(pred_probs[:, idx])),
                "target_mean_probability": float(np.mean(target_probs[:, idx])),
            }
        )
    pd.DataFrame(top_rows).to_csv(output_dir / "semantic_probe_top_concepts.csv", index=False)

    examples = results.get("_examples", [])
    with open(output_dir / "semantic_probe_examples.json", "w") as f:
        json.dump(examples, f, indent=2, default=str)

    show = {
        k: float(results[k])
        for k in ["agreement@1", "agreement@3", "agreement@5", "agreement@10"]
        if k in results
    }
    if show:
        figs += save_figure(
            create_bar_chart(
                show,
                title="Semantic Probe Agreement",
                ylabel="Fraction of samples",
                figsize=(6, 3.5),
            ),
            figs_dir / "figure_semantic_probe_agreement_bar",
            formats=save_formats,
        )

    figs += save_figure(
        create_scatter(
            results["_per_trial_spearman"],
            results["_per_trial_js"],
            title="Text-Probe Score Alignment",
            xlabel="Per-sample score-vector Spearman",
            ylabel="JS divergence",
            color=results["_target_top1_rank"],
            alpha=0.55,
            figsize=(6, 4),
        ),
        figs_dir / "figure_semantic_probe_score_correlation",
        formats=save_formats,
    )

    figs += save_figure(
        create_histogram(
            {"JS divergence": results["_per_trial_js"]},
            title="Semantic Probe JS Divergence",
            xlabel="JS divergence",
            bins=40,
            figsize=(6, 3.5),
        ),
        figs_dir / "figure_semantic_probe_js_distribution",
        formats=save_formats,
    )

    if results.get("correct_vs_incorrect_status") == "computed":
        correct_vals = {
            "correct": results["_per_trial_js"][
                np.asarray(results.get("_retrieval_correct", []), dtype=bool)
            ],
            "incorrect": results["_per_trial_js"][
                ~np.asarray(results.get("_retrieval_correct", []), dtype=bool)
            ],
        }
        if len(correct_vals["correct"]) and len(correct_vals["incorrect"]):
            figs += save_figure(
                create_boxplot(
                    correct_vals,
                    title="Semantic Probe JS: Correct vs Incorrect Retrieval",
                    ylabel="JS divergence",
                ),
                figs_dir / "figure_semantic_probe_correct_vs_incorrect",
                formats=save_formats,
            )

    figs += _plot_examples(
        examples,
        figs_dir / "figure_semantic_probe_examples",
        formats=save_formats,
    )
    figs += _plot_concept_confusion(
        pred_top1,
        target_top1,
        concepts,
        figs_dir / "figure_semantic_concept_confusion",
        formats=save_formats,
    )

    logger.info("Semantic probe results saved to %s", output_dir)
    return {"metrics": metrics, "figures": [str(f) for f in figs]}
