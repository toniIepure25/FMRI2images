#!/usr/bin/env python3
"""Sweep post-hoc compact+rerrank score fusion on saved experiment outputs.

This script performs VAL-only hyperparameter selection for two-stage score
fusion, then evaluates the frozen best setting on SHARED1000.

Usage:
    python scripts/evaluation/sweep_fusion_retrieval.py \
        experimental_results/V30e_rerank_head_2048/subj01
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SHORTLIST_KS = [10, 20, 50, 100, 200]
ALPHAS = [round(x * 0.05, 2) for x in range(21)]
COMPACT_SCORE_VARIANTS = ["raw_cosine", "csls"]
NORMALIZATION_MODES = ["none", "zscore", "minmax", "stdscale"]
RRF_K = 60.0


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _l2_normalize(a) @ _l2_normalize(b).T


def _csls_scores(preds: np.ndarray, gallery: np.ndarray, k: int = 10) -> np.ndarray:
    sim = _cosine_sim(preds, gallery)
    top_k_pred = np.sort(sim, axis=1)[:, -k:].mean(axis=1, keepdims=True)
    top_k_gal = np.sort(sim, axis=0)[-k:, :].mean(axis=0, keepdims=True)
    return sim - 0.5 * (top_k_pred + top_k_gal)


def _metrics_from_gt_rank(gt_rank: np.ndarray) -> dict[str, float]:
    gt_rank = gt_rank.astype(np.int32)
    return {
        "R@1": float(np.mean(gt_rank <= 1)),
        "R@5": float(np.mean(gt_rank <= 5)),
        "R@10": float(np.mean(gt_rank <= 10)),
        "median_rank": float(np.median(gt_rank)),
        "MRR": float(np.mean(1.0 / gt_rank)),
    }


def _gt_rank_from_scores(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, axis=1)
    n = scores.shape[0]
    gt_rank = np.argmax(order == np.arange(n)[:, None], axis=1) + 1
    return order, gt_rank.astype(np.int32)


def _normalize_shortlist_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return scores
    if mode == "zscore":
        mean = scores.mean(axis=1, keepdims=True)
        std = scores.std(axis=1, keepdims=True)
        return (scores - mean) / np.maximum(std, 1e-8)
    if mode == "minmax":
        s_min = scores.min(axis=1, keepdims=True)
        s_max = scores.max(axis=1, keepdims=True)
        return (scores - s_min) / np.maximum(s_max - s_min, 1e-8)
    if mode == "stdscale":
        std = scores.std(axis=1, keepdims=True)
        return scores / np.maximum(std, 1e-8)
    raise ValueError(f"Unknown normalization mode: {mode}")


def _rank_within_shortlist(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(scores.shape[0])[:, None], order] = np.arange(scores.shape[1])[None, :] + 1
    return ranks


def _build_split_scores(compact_preds: np.ndarray, compact_gts: np.ndarray,
                        rerank_preds: np.ndarray, rerank_gts: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "compact_raw": _cosine_sim(compact_preds, compact_gts),
        "compact_csls": _csls_scores(compact_preds, compact_gts, k=10),
        "rerank": _cosine_sim(rerank_preds, rerank_gts),
    }


def _compute_baselines(scores: dict[str, np.ndarray]) -> dict[str, Any]:
    baselines: dict[str, Any] = {}
    for key, score_name in [
        ("compact_raw", "compact_raw"),
        ("compact_csls", "compact_csls"),
        ("rerank_only", "rerank"),
    ]:
        _, gt_rank = _gt_rank_from_scores(scores[score_name])
        baselines[key] = _metrics_from_gt_rank(gt_rank)
    return baselines


def _sweep_single_setting(
    compact_scores: np.ndarray,
    rerank_scores: np.ndarray,
    compact_order: np.ndarray,
    compact_gt_rank: np.ndarray,
    shortlist_k: int,
    family: str,
    normalization: str,
    alpha: float | None,
) -> dict[str, Any]:
    n = compact_scores.shape[0]
    shortlist = compact_order[:, :shortlist_k]
    row_idx = np.arange(n)[:, None]

    compact_sl = compact_scores[row_idx, shortlist]
    rerank_sl = rerank_scores[row_idx, shortlist]

    if family == "weighted":
        fused = alpha * compact_sl + (1.0 - alpha) * rerank_sl
    elif family == "normalized_weighted":
        compact_norm = _normalize_shortlist_scores(compact_sl, normalization)
        rerank_norm = _normalize_shortlist_scores(rerank_sl, normalization)
        fused = alpha * compact_norm + (1.0 - alpha) * rerank_norm
    elif family == "rrf":
        compact_rank = _rank_within_shortlist(compact_sl)
        rerank_rank = _rank_within_shortlist(rerank_sl)
        fused = 1.0 / (RRF_K + compact_rank) + 1.0 / (RRF_K + rerank_rank)
    else:
        raise ValueError(f"Unknown fusion family: {family}")

    local_order = np.argsort(-fused, axis=1)
    reranked_shortlist = shortlist[row_idx, local_order]

    gt_rank = compact_gt_rank.copy()
    gt_ids = np.arange(n)
    hit_mask = np.any(shortlist == gt_ids[:, None], axis=1)
    if np.any(hit_mask):
        hit_rows = np.where(hit_mask)[0]
        local_gt_rank = np.argmax(
            reranked_shortlist[hit_rows] == hit_rows[:, None],
            axis=1,
        ) + 1
        gt_rank[hit_rows] = local_gt_rank.astype(np.int32)

    metrics = _metrics_from_gt_rank(gt_rank)
    return {
        "shortlist_k": shortlist_k,
        "fusion_family": family,
        "normalization": normalization,
        "alpha": alpha,
        **metrics,
    }


def _best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        results,
        key=lambda r: (
            r["R@1"],
            r["R@5"],
            r["R@10"],
            r["MRR"],
            -r["median_rank"],
        ),
    )


def _load_split(metrics_dir: Path, prefix: str) -> dict[str, np.ndarray]:
    compact_pred = metrics_dir / f"{prefix}_predictions_compact.npy"
    compact_gt = metrics_dir / f"{prefix}_ground_truth_compact.npy"
    rerank_pred = metrics_dir / f"{prefix}_predictions_rerank.npy"
    rerank_gt = metrics_dir / f"{prefix}_ground_truth_rerank.npy"
    for path in [compact_pred, compact_gt, rerank_pred, rerank_gt]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    return {
        "compact_preds": np.load(compact_pred),
        "compact_gts": np.load(compact_gt),
        "rerank_preds": np.load(rerank_pred),
        "rerank_gts": np.load(rerank_gt),
    }


def _evaluate_split(split_arrays: dict[str, np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores = _build_split_scores(
        split_arrays["compact_preds"],
        split_arrays["compact_gts"],
        split_arrays["rerank_preds"],
        split_arrays["rerank_gts"],
    )
    baselines = _compute_baselines(scores)

    compact_orders = {}
    compact_gt_ranks = {}
    for variant_name, score_name in [
        ("raw_cosine", "compact_raw"),
        ("csls", "compact_csls"),
    ]:
        compact_orders[variant_name], compact_gt_ranks[variant_name] = _gt_rank_from_scores(scores[score_name])

    sweep_results: list[dict[str, Any]] = []
    for compact_variant in COMPACT_SCORE_VARIANTS:
        compact_scores = scores["compact_raw"] if compact_variant == "raw_cosine" else scores["compact_csls"]
        compact_order = compact_orders[compact_variant]
        compact_gt_rank = compact_gt_ranks[compact_variant]
        rerank_scores = scores["rerank"]

        for shortlist_k in SHORTLIST_KS:
            for alpha in ALPHAS:
                result = _sweep_single_setting(
                    compact_scores=compact_scores,
                    rerank_scores=rerank_scores,
                    compact_order=compact_order,
                    compact_gt_rank=compact_gt_rank,
                    shortlist_k=shortlist_k,
                    family="weighted",
                    normalization="none",
                    alpha=alpha,
                )
                result["compact_score_variant"] = compact_variant
                sweep_results.append(result)

            for normalization in [m for m in NORMALIZATION_MODES if m != "none"]:
                for alpha in ALPHAS:
                    result = _sweep_single_setting(
                        compact_scores=compact_scores,
                        rerank_scores=rerank_scores,
                        compact_order=compact_order,
                        compact_gt_rank=compact_gt_rank,
                        shortlist_k=shortlist_k,
                        family="normalized_weighted",
                        normalization=normalization,
                        alpha=alpha,
                    )
                    result["compact_score_variant"] = compact_variant
                    sweep_results.append(result)

            result = _sweep_single_setting(
                compact_scores=compact_scores,
                rerank_scores=rerank_scores,
                compact_order=compact_order,
                compact_gt_rank=compact_gt_rank,
                shortlist_k=shortlist_k,
                family="rrf",
                normalization="none",
                alpha=None,
            )
            result["compact_score_variant"] = compact_variant
            sweep_results.append(result)

    baselines["rerank_replacement_raw"] = _sweep_single_setting(
        compact_scores=scores["compact_raw"],
        rerank_scores=scores["rerank"],
        compact_order=compact_orders["raw_cosine"],
        compact_gt_rank=compact_gt_ranks["raw_cosine"],
        shortlist_k=100,
        family="weighted",
        normalization="none",
        alpha=0.0,
    )
    baselines["rerank_replacement_raw"]["compact_score_variant"] = "raw_cosine"
    baselines["rerank_replacement_csls"] = _sweep_single_setting(
        compact_scores=scores["compact_csls"],
        rerank_scores=scores["rerank"],
        compact_order=compact_orders["csls"],
        compact_gt_rank=compact_gt_ranks["csls"],
        shortlist_k=100,
        family="weighted",
        normalization="none",
        alpha=0.0,
    )
    baselines["rerank_replacement_csls"]["compact_score_variant"] = "csls"

    return baselines, sweep_results


def _apply_frozen_setting(split_arrays: dict[str, np.ndarray], setting: dict[str, Any]) -> dict[str, Any]:
    scores = _build_split_scores(
        split_arrays["compact_preds"],
        split_arrays["compact_gts"],
        split_arrays["rerank_preds"],
        split_arrays["rerank_gts"],
    )
    compact_variant = setting["compact_score_variant"]
    compact_scores = scores["compact_raw"] if compact_variant == "raw_cosine" else scores["compact_csls"]
    compact_order, compact_gt_rank = _gt_rank_from_scores(compact_scores)
    result = _sweep_single_setting(
        compact_scores=compact_scores,
        rerank_scores=scores["rerank"],
        compact_order=compact_order,
        compact_gt_rank=compact_gt_rank,
        shortlist_k=int(setting["shortlist_k"]),
        family=setting["fusion_family"],
        normalization=setting["normalization"],
        alpha=setting["alpha"],
    )
    result["compact_score_variant"] = compact_variant
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep post-hoc compact+rerrank score fusion")
    parser.add_argument("results_dir", type=str, help="Experiment directory, e.g. experimental_results/V30e_rerank_head_2048/subj01")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metrics_dir = results_dir / "metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    diagnostics_dir = results_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    val_arrays = _load_split(metrics_dir, "val")
    shared_arrays = _load_split(metrics_dir, "shared1000")

    val_baselines, val_results = _evaluate_split(val_arrays)
    best_val = _best_result(val_results)
    shared_best = _apply_frozen_setting(shared_arrays, best_val)
    shared_baselines, _ = _evaluate_split(shared_arrays)

    best_compact_val = max(
        val_baselines["compact_raw"]["R@1"],
        val_baselines["compact_csls"]["R@1"],
    )
    best_fusion_gain = best_val["R@1"] - best_compact_val

    report = {
        "search_space": {
            "shortlist_k": SHORTLIST_KS,
            "fusion_families": ["weighted", "normalized_weighted", "rrf"],
            "normalization_modes": NORMALIZATION_MODES,
            "alpha_sweep": ALPHAS,
            "compact_score_variants": COMPACT_SCORE_VARIANTS,
            "rerank_score_variant": "cosine",
            "csls_included_for_compact": True,
        },
        "val": {
            "baselines": val_baselines,
            "best_setting": best_val,
            "all_results": val_results,
        },
        "shared1000": {
            "baselines": shared_baselines,
            "frozen_best_setting": shared_best,
        },
    }

    json_path = diagnostics_dir / "fusion_sweep.json"
    csv_path = diagnostics_dir / "fusion_sweep_val.csv"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    _write_csv(csv_path, val_results)

    print("=" * 72)
    print("FUSION SWEEP SUMMARY")
    print("=" * 72)
    print(f"Results dir: {results_dir}")
    print("")
    print("VAL baselines")
    print(f"  Compact raw R@1:         {val_baselines['compact_raw']['R@1']:.1%}")
    print(f"  Compact CSLS R@1:        {val_baselines['compact_csls']['R@1']:.1%}")
    print(f"  Rerank-only R@1:         {val_baselines['rerank_only']['R@1']:.1%}")
    print(f"  Rerank replacement R@1:  {val_baselines['rerank_replacement_csls']['R@1']:.1%} (k=100, csls, alpha=0)")
    print("")
    print("Best VAL fusion setting")
    print(f"  compact score:           {best_val['compact_score_variant']}")
    print(f"  family:                  {best_val['fusion_family']}")
    print(f"  normalization:           {best_val['normalization']}")
    print(f"  shortlist_k:             {best_val['shortlist_k']}")
    print(f"  alpha:                   {best_val['alpha']}")
    print(f"  VAL R@1 / R@5 / R@10:    {best_val['R@1']:.1%} / {best_val['R@5']:.1%} / {best_val['R@10']:.1%}")
    print(f"  VAL MedR / MRR:          {best_val['median_rank']:.1f} / {best_val['MRR']:.4f}")
    print(f"  Gain over best compact:  {best_fusion_gain:+.1%}")
    print("")
    print("Frozen SHARED1000 result")
    print(f"  SHARED R@1 / R@5 / R@10: {shared_best['R@1']:.1%} / {shared_best['R@5']:.1%} / {shared_best['R@10']:.1%}")
    print(f"  SHARED MedR / MRR:       {shared_best['median_rank']:.1f} / {shared_best['MRR']:.4f}")
    print("")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
