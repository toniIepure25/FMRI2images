#!/usr/bin/env python3
"""Measure union-shortlist oracle coverage for V39 headroom audit.

Given saved predictions from two or three experts (compact, legacy, optional
rerank), this script measures the oracle recall of the union shortlist — i.e.,
if a perfect reranker existed, what fraction of queries could it answer
correctly given only the candidates in the union of experts' top-K lists?

This is the absolute ceiling for any candidate-level reranker.

Usage:
    python scripts/evaluation/measure_union_shortlist_oracle.py \
        experimental_results/V35_legacy_teacher_distill/subj01 \
        experimental_results/N1v28a_dual_head/subj01

Outputs:
    {tri_results_dir}/diagnostics/union_shortlist_oracle.json
    {tri_results_dir}/diagnostics/union_shortlist_oracle.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ── Import shared helpers from sweep_tri_fusion_retrieval ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_tri_fusion_retrieval import (  # noqa: E402
    _align_common_ids,
    _assert_split_disjointness,
    _build_split_scores,
    _cosine_sim,
    _csls_scores,
    _gt_rank_from_scores,
    _load_legacy_split,
    _load_tri_split,
    _metrics_from_gt_rank,
    _nearby_experiment_dirs,
    _save_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SHORTLIST_KS = [25, 50, 100, 150, 200]


def _union_shortlist(
    compact_order: np.ndarray,
    legacy_order: np.ndarray,
    rerank_order: np.ndarray | None,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-query union shortlist from top-K of each expert.

    Returns:
        shortlists: list of 1-D int arrays (one per query, variable length)
        sizes: (N,) int array of union sizes
    """
    n = compact_order.shape[0]
    shortlists = []
    sizes = np.empty(n, dtype=np.int32)
    for i in range(n):
        candidates = set(compact_order[i, :k].tolist())
        candidates.update(legacy_order[i, :k].tolist())
        if rerank_order is not None:
            candidates.update(rerank_order[i, :k].tolist())
        sl = np.array(sorted(candidates), dtype=np.int32)
        shortlists.append(sl)
        sizes[i] = len(sl)
    return shortlists, sizes


def _measure_oracle(
    shortlists: list[np.ndarray],
    n: int,
) -> dict[str, float]:
    """Oracle: GT is at position i for query i. Check if i is in shortlist."""
    hits = 0
    for i in range(n):
        if i in shortlists[i]:
            hits += 1
    return {
        "oracle_recall": float(hits / n),
        "oracle_hit_count": int(hits),
        "oracle_miss_count": int(n - hits),
    }


def _expert_overlap_stats(
    compact_order: np.ndarray,
    legacy_order: np.ndarray,
    rerank_order: np.ndarray | None,
    k: int,
) -> dict[str, float]:
    """Measure overlap and disagreement between experts."""
    n = compact_order.shape[0]

    compact_sets = [set(compact_order[i, :k].tolist()) for i in range(n)]
    legacy_sets = [set(legacy_order[i, :k].tolist()) for i in range(n)]

    # Jaccard overlap
    jaccard_cl = []
    for i in range(n):
        intersection = len(compact_sets[i] & legacy_sets[i])
        union_size = len(compact_sets[i] | legacy_sets[i])
        jaccard_cl.append(intersection / max(union_size, 1))

    # Top-1 disagreement
    compact_top1 = compact_order[:, 0]
    legacy_top1 = legacy_order[:, 0]
    top1_disagree = float(np.mean(compact_top1 != legacy_top1))

    # GT source: which expert has GT in top-K?
    compact_has_gt = sum(1 for i in range(n) if i in compact_sets[i])
    legacy_has_gt = sum(1 for i in range(n) if i in legacy_sets[i])
    both_have_gt = sum(1 for i in range(n) if i in compact_sets[i] and i in legacy_sets[i])
    neither_has_gt = sum(1 for i in range(n) if i not in compact_sets[i] and i not in legacy_sets[i])
    only_compact = compact_has_gt - both_have_gt
    only_legacy = legacy_has_gt - both_have_gt

    stats = {
        "jaccard_compact_legacy_mean": float(np.mean(jaccard_cl)),
        "jaccard_compact_legacy_std": float(np.std(jaccard_cl)),
        "top1_disagree_fraction": top1_disagree,
        "compact_has_gt_rate": float(compact_has_gt / n),
        "legacy_has_gt_rate": float(legacy_has_gt / n),
        "both_have_gt_rate": float(both_have_gt / n),
        "only_compact_has_gt_rate": float(only_compact / n),
        "only_legacy_has_gt_rate": float(only_legacy / n),
        "neither_has_gt_rate": float(neither_has_gt / n),
    }

    if rerank_order is not None:
        rerank_sets = [set(rerank_order[i, :k].tolist()) for i in range(n)]
        rerank_has_gt = sum(1 for i in range(n) if i in rerank_sets[i])
        stats["rerank_has_gt_rate"] = float(rerank_has_gt / n)
        jaccard_cr = []
        for i in range(n):
            intersection = len(compact_sets[i] & rerank_sets[i])
            union_size = len(compact_sets[i] | rerank_sets[i])
            jaccard_cr.append(intersection / max(union_size, 1))
        stats["jaccard_compact_rerank_mean"] = float(np.mean(jaccard_cr))

    return stats


def _run_audit_for_split(
    split_arrays: dict[str, np.ndarray],
    split_name: str,
    include_rerank: bool,
) -> dict[str, Any]:
    """Run full headroom audit for one split."""
    logger.info("Computing scores for %s split (%d queries)...", split_name, split_arrays["compact_preds"].shape[0])
    scores = _build_split_scores(split_arrays)

    # Get expert orderings
    compact_csls_order, compact_csls_gt_rank = _gt_rank_from_scores(scores["compact_csls"])
    legacy_csls_order, legacy_csls_gt_rank = _gt_rank_from_scores(scores["legacy_csls"])
    rerank_order, _ = _gt_rank_from_scores(scores["rerank"]) if include_rerank else (None, None)

    results_per_k: list[dict[str, Any]] = []
    for k in SHORTLIST_KS:
        logger.info("  K=%d ...", k)
        shortlists, sizes = _union_shortlist(
            compact_csls_order, legacy_csls_order, rerank_order if include_rerank else None, k
        )
        oracle = _measure_oracle(shortlists, scores["compact_csls"].shape[0])
        overlap = _expert_overlap_stats(
            compact_csls_order, legacy_csls_order, rerank_order if include_rerank else None, k
        )

        row = {
            "K": int(k),
            "include_rerank": include_rerank,
            "union_oracle_recall": oracle["oracle_recall"],
            "union_oracle_hit": oracle["oracle_hit_count"],
            "union_oracle_miss": oracle["oracle_miss_count"],
            "mean_union_size": float(np.mean(sizes)),
            "median_union_size": float(np.median(sizes)),
            "min_union_size": int(np.min(sizes)),
            "max_union_size": int(np.max(sizes)),
            **overlap,
        }
        results_per_k.append(row)

    # Baselines
    compact_csls_metrics = _metrics_from_gt_rank(compact_csls_gt_rank)
    legacy_csls_metrics = _metrics_from_gt_rank(legacy_csls_gt_rank)

    return {
        "split": split_name,
        "n_queries": int(split_arrays["compact_preds"].shape[0]),
        "baselines": {
            "compact_csls": compact_csls_metrics,
            "legacy_csls": legacy_csls_metrics,
        },
        "per_k": results_per_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure union-shortlist oracle coverage for V39 headroom audit"
    )
    parser.add_argument(
        "tri_results_dir",
        type=str,
        help="Triple-head results dir (V32/V33/V35/V36/V38), "
        "e.g. experimental_results/V35_legacy_teacher_distill/subj01",
    )
    parser.add_argument(
        "legacy_results_dir",
        type=str,
        help="Legacy N1v28a results dir, "
        "e.g. experimental_results/N1v28a_dual_head/subj01",
    )
    parser.add_argument(
        "--include-rerank",
        action="store_true",
        help="Also include the rerank expert in the union shortlist",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "shared1000"],
        choices=["val", "shared1000"],
        help="Which splits to audit (default: val shared1000)",
    )
    args = parser.parse_args()

    tri_results_dir = Path(args.tri_results_dir)
    legacy_results_dir = Path(args.legacy_results_dir)

    if not tri_results_dir.exists():
        nearby = _nearby_experiment_dirs(tri_results_dir)
        hint = f" Nearby: {nearby}" if nearby else ""
        raise FileNotFoundError(f"Tri results dir not found: {tri_results_dir}.{hint}")
    if not legacy_results_dir.exists():
        nearby = _nearby_experiment_dirs(legacy_results_dir)
        hint = f" Nearby: {nearby}" if nearby else ""
        raise FileNotFoundError(f"Legacy results dir not found: {legacy_results_dir}.{hint}")

    tri_metrics = tri_results_dir / "metrics"
    legacy_metrics = legacy_results_dir / "metrics"
    if not tri_metrics.exists():
        raise FileNotFoundError(f"No metrics/ dir in {tri_results_dir}")
    if not legacy_metrics.exists():
        raise FileNotFoundError(f"No metrics/ dir in {legacy_results_dir}")

    diagnostics_dir = tri_results_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    all_results: dict[str, Any] = {"splits": {}}

    val_tri, val_legacy, val_aligned = None, None, None
    if "val" in args.splits:
        logger.info("Loading VAL split...")
        val_tri = _load_tri_split(tri_metrics, "val")
        val_legacy = _load_legacy_split(legacy_metrics, legacy_results_dir, "val", reference_split=None)
        val_aligned = _align_common_ids(val_tri, val_legacy, "val")

        val_results = _run_audit_for_split(val_aligned, "val", include_rerank=args.include_rerank)
        all_results["splits"]["val"] = val_results

    if "shared1000" in args.splits:
        logger.info("Loading SHARED1000 split...")
        s1000_tri = _load_tri_split(tri_metrics, "shared1000")
        ref = val_tri if val_tri is not None else None
        s1000_legacy = _load_legacy_split(
            legacy_metrics, legacy_results_dir, "shared1000", reference_split=ref
        )
        s1000_aligned = _align_common_ids(s1000_tri, s1000_legacy, "shared1000")

        if val_aligned is not None:
            _assert_split_disjointness(val_aligned["nsd_ids"], s1000_aligned["nsd_ids"], "val vs shared1000")

        s1000_results = _run_audit_for_split(s1000_aligned, "shared1000", include_rerank=args.include_rerank)
        all_results["splits"]["shared1000"] = s1000_results

    # Print summary
    print("\n" + "=" * 70)
    print("UNION SHORTLIST ORACLE HEADROOM AUDIT")
    print("=" * 70)
    for split_name, split_data in all_results["splits"].items():
        print(f"\n--- {split_name.upper()} ({split_data['n_queries']} queries) ---")
        print(f"  Compact CSLS R@1: {split_data['baselines']['compact_csls']['R@1']:.1%}")
        print(f"  Legacy  CSLS R@1: {split_data['baselines']['legacy_csls']['R@1']:.1%}")
        print()
        for row in split_data["per_k"]:
            rerank_flag = " +rerank" if row["include_rerank"] else ""
            print(
                f"  K={row['K']:>3d}{rerank_flag}:  "
                f"union_oracle={row['union_oracle_recall']:.1%}  "
                f"mean_size={row['mean_union_size']:.1f}  "
                f"top1_disagree={row['top1_disagree_fraction']:.1%}  "
                f"only_compact={row['only_compact_has_gt_rate']:.1%}  "
                f"only_legacy={row['only_legacy_has_gt_rate']:.1%}  "
                f"neither={row['neither_has_gt_rate']:.1%}"
            )
    print()

    # Assess 85% feasibility
    for split_name, split_data in all_results["splits"].items():
        best_oracle = max(r["union_oracle_recall"] for r in split_data["per_k"])
        if best_oracle >= 0.85:
            print(f"[{split_name}] 85%+ IS REACHABLE: best union oracle = {best_oracle:.1%}")
        elif best_oracle >= 0.80:
            print(f"[{split_name}] 85% is TIGHT: best union oracle = {best_oracle:.1%} — needs near-perfect reranking")
        else:
            print(f"[{split_name}] 85% is UNREACHABLE with current experts: best union oracle = {best_oracle:.1%}")

    # Save
    out_json = diagnostics_dir / "union_shortlist_oracle.json"
    _save_json(out_json, all_results)
    logger.info("Saved JSON report to %s", out_json)

    # Save CSV (flat table)
    csv_rows = []
    for split_name, split_data in all_results["splits"].items():
        for row in split_data["per_k"]:
            csv_row = {"split": split_name, **row}
            csv_rows.append(csv_row)
    out_csv = diagnostics_dir / "union_shortlist_oracle.csv"
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info("Saved CSV to %s", out_csv)


if __name__ == "__main__":
    main()
