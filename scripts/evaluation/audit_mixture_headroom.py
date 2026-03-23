#!/usr/bin/env python3
"""Audit mixture-vMF headroom on frozen saved experiment outputs.

This script is the zero-retrain forensic wave for V43b-style runs. It answers:
- does mixture-CSLS beat consensus compact CSLS?
- does frozen tri-fusion improve if mixture-CSLS is used as the compact expert?
- how much oracle headroom remains at shortlist K?
- among fixed-tri misses, which ones are recoverable by mixture, legacy, or the union shortlist?
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from sweep_tri_fusion_retrieval import (
    _align_common_ids,
    _apply_frozen_setting,
    _assert_split_disjointness,
    _build_split_scores,
    _compute_baselines,
    _gt_rank_from_scores,
    _load_legacy_split,
    _load_tri_split,
    _nearby_experiment_dirs,
    _normalize_shortlist_scores,
    _rank_within_shortlist,
    _save_json,
)

DEFAULT_FIXED_TRI_SETTING = {
    "compact_score_variant": "csls",
    "legacy_score_variant": "csls",
    "fusion_family": "normalized_weighted",
    "normalization": "zscore",
    "shortlist_k": 150,
    "alpha": 0.3,
    "beta": 0.0,
    "gamma": 0.7,
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _compact_scores_from_variant(scores: dict[str, np.ndarray], variant: str) -> np.ndarray:
    mapping = {
        "raw_cosine": "compact_raw",
        "csls": "compact_csls",
        "mixture_raw": "mixture_raw",
        "mixture_csls": "mixture_csls",
    }
    key = mapping[variant]
    if key not in scores:
        raise ValueError(f"Requested compact variant {variant} is unavailable")
    return scores[key]


def _union_oracle_summary(scores: dict[str, np.ndarray], expert_keys: list[str], k: int) -> dict[str, Any]:
    orders = {key: _gt_rank_from_scores(scores[key])[0] for key in expert_keys}
    n = next(iter(orders.values())).shape[0]
    hits = 0
    union_sizes = []
    for i in range(n):
        cand = set()
        for key in expert_keys:
            cand.update(orders[key][i, :k].tolist())
        union_sizes.append(len(cand))
        if i in cand:
            hits += 1
    return {
        "experts": expert_keys,
        "k": int(k),
        "oracle_recall": float(hits / n),
        "oracle_hit_count": int(hits),
        "oracle_miss_count": int(n - hits),
        "mean_union_size": float(np.mean(union_sizes)),
        "median_union_size": float(np.median(union_sizes)),
    }


def _fixed_tri_details(scores: dict[str, np.ndarray], setting: dict[str, Any]) -> dict[str, Any]:
    compact_scores = _compact_scores_from_variant(scores, setting["compact_score_variant"])
    legacy_scores = scores["legacy_csls"] if setting["legacy_score_variant"] == "csls" else scores["legacy_raw"]
    rerank_scores = scores["rerank"]
    compact_order, compact_gt_rank = _gt_rank_from_scores(compact_scores)
    shortlist_k = int(setting["shortlist_k"])
    shortlist = compact_order[:, :shortlist_k]
    row_idx = np.arange(compact_scores.shape[0])[:, None]
    compact_sl = compact_scores[row_idx, shortlist]
    rerank_sl = rerank_scores[row_idx, shortlist]
    legacy_sl = legacy_scores[row_idx, shortlist]
    family = str(setting["fusion_family"])
    normalization = str(setting["normalization"])
    alpha = float(setting["alpha"])
    beta = float(setting["beta"])
    gamma = float(setting["gamma"])

    if family == "weighted":
        fused_scores = alpha * compact_sl + beta * rerank_sl + gamma * legacy_sl
    elif family == "normalized_weighted":
        fused_scores = (
            alpha * _normalize_shortlist_scores(compact_sl, normalization)
            + beta * _normalize_shortlist_scores(rerank_sl, normalization)
            + gamma * _normalize_shortlist_scores(legacy_sl, normalization)
        )
    elif family == "rrf":
        fused_scores = (
            alpha / (60.0 + _rank_within_shortlist(compact_sl))
            + beta / (60.0 + _rank_within_shortlist(rerank_sl))
            + gamma / (60.0 + _rank_within_shortlist(legacy_sl))
        )
    elif family == "rank_average":
        fused_scores = -(
            alpha * _rank_within_shortlist(compact_sl)
            + beta * _rank_within_shortlist(rerank_sl)
            + gamma * _rank_within_shortlist(legacy_sl)
        )
    else:
        raise ValueError(f"Unsupported family for audit: {family}")

    local_order = np.argsort(-fused_scores, axis=1)
    reranked_shortlist = shortlist[row_idx, local_order]
    gt_rank = compact_gt_rank.copy()
    gt_ids = np.arange(compact_scores.shape[0])
    hit_mask = np.any(shortlist == gt_ids[:, None], axis=1)
    if np.any(hit_mask):
        hit_rows = np.where(hit_mask)[0]
        local_gt_rank = np.argmax(
            reranked_shortlist[hit_rows] == hit_rows[:, None],
            axis=1,
        ) + 1
        gt_rank[hit_rows] = local_gt_rank.astype(np.int32)
    top1 = reranked_shortlist[:, 0]
    return {
        "gt_rank": gt_rank.astype(np.int32),
        "top1": top1.astype(np.int32),
        "shortlist": shortlist.astype(np.int32),
    }


def _miss_taxonomy(scores: dict[str, np.ndarray], split_name: str, topk: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if "mixture_csls" not in scores:
        return {
            "available": False,
            "reason": "mixture_csls unavailable for this split",
        }, []

    fixed_details = _fixed_tri_details(scores, DEFAULT_FIXED_TRI_SETTING)
    mixture_order, mixture_gt_rank = _gt_rank_from_scores(scores["mixture_csls"])
    compact_order, compact_gt_rank = _gt_rank_from_scores(scores["compact_csls"])
    legacy_order, legacy_gt_rank = _gt_rank_from_scores(scores["legacy_csls"])

    rows: list[dict[str, Any]] = []
    counts = {
        "fixed_tri_hits": 0,
        "fixed_tri_misses": 0,
        "recoverable_by_mixture_only": 0,
        "recoverable_by_legacy_only": 0,
        "recoverable_by_union_but_misranked": 0,
        "not_recoverable_with_current_experts": 0,
    }

    n = scores["compact_csls"].shape[0]
    for i in range(n):
        fixed_hit = int(fixed_details["top1"][i] == i)
        mixture_hit = int(mixture_order[i, 0] == i)
        legacy_hit = int(legacy_order[i, 0] == i)
        compact_hit = int(compact_order[i, 0] == i)
        union_candidates = set(mixture_order[i, :topk].tolist())
        union_candidates.update(compact_order[i, :topk].tolist())
        union_candidates.update(legacy_order[i, :topk].tolist())
        union_has_gt = int(i in union_candidates)

        if fixed_hit:
            category = "fixed_tri_hit"
            counts["fixed_tri_hits"] += 1
        else:
            counts["fixed_tri_misses"] += 1
            if mixture_hit and not legacy_hit:
                category = "recoverable_by_mixture_only"
                counts[category] += 1
            elif legacy_hit and not mixture_hit:
                category = "recoverable_by_legacy_only"
                counts[category] += 1
            elif union_has_gt:
                category = "recoverable_by_union_but_misranked"
                counts[category] += 1
            else:
                category = "not_recoverable_with_current_experts"
                counts[category] += 1

        rows.append({
            "split": split_name,
            "query_index": int(i),
            "fixed_tri_hit": fixed_hit,
            "fixed_tri_gt_rank": int(fixed_details["gt_rank"][i]),
            "compact_hit": compact_hit,
            "compact_gt_rank": int(compact_gt_rank[i]),
            "mixture_hit": mixture_hit,
            "mixture_gt_rank": int(mixture_gt_rank[i]),
            "legacy_hit": legacy_hit,
            "legacy_gt_rank": int(legacy_gt_rank[i]),
            "union_topk": int(topk),
            "union_has_gt": union_has_gt,
            "category": category,
        })

    miss_rows = [r for r in rows if r["fixed_tri_hit"] == 0]
    return {
        "available": True,
        **counts,
        "recoverable_fraction_among_misses": float(
            (counts["recoverable_by_mixture_only"] + counts["recoverable_by_legacy_only"] + counts["recoverable_by_union_but_misranked"]) / max(counts["fixed_tri_misses"], 1)
        ),
    }, miss_rows


def _run_split_audit(split_arrays: dict[str, np.ndarray], split_name: str, topk: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores = _build_split_scores(split_arrays)
    baselines = _compute_baselines(scores)
    fixed_consensus = _apply_frozen_setting(split_arrays, DEFAULT_FIXED_TRI_SETTING, shared_has_kappa=False)
    fixed_mixture = None
    if "mixture_csls" in scores:
        mixture_setting = dict(DEFAULT_FIXED_TRI_SETTING)
        mixture_setting["compact_score_variant"] = "mixture_csls"
        fixed_mixture = _apply_frozen_setting(split_arrays, mixture_setting, shared_has_kappa=False)

    union_reports = {
        "compact_legacy": _union_oracle_summary(scores, ["compact_csls", "legacy_csls"], topk),
    }
    if "mixture_csls" in scores:
        union_reports["mixture_legacy"] = _union_oracle_summary(scores, ["mixture_csls", "legacy_csls"], topk)
        union_reports["mixture_compact_legacy"] = _union_oracle_summary(scores, ["mixture_csls", "compact_csls", "legacy_csls"], topk)

    taxonomy, per_query_rows = _miss_taxonomy(scores, split_name, topk)
    return {
        "split": split_name,
        "n_queries": int(split_arrays["nsd_ids"].shape[0]),
        "baselines": baselines,
        "fixed_tri_consensus": fixed_consensus,
        "fixed_tri_mixture": fixed_mixture,
        "union_oracle": union_reports,
        "miss_taxonomy": taxonomy,
        "mixture_gain_over_compact_csls": (
            float(baselines["mixture_csls"]["R@1"] - baselines["compact_csls"]["R@1"])
            if "mixture_csls" in baselines else None
        ),
        "fixed_mixture_gain_over_fixed_consensus": (
            float(fixed_mixture["R@1"] - fixed_consensus["R@1"])
            if fixed_mixture is not None else None
        ),
    }, per_query_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit mixture-vMF headroom on frozen results")
    parser.add_argument("tri_results_dir", type=str)
    parser.add_argument("legacy_results_dir", type=str)
    parser.add_argument("--oracle-topk", type=int, default=50)
    args = parser.parse_args()

    tri_results_dir = Path(args.tri_results_dir)
    legacy_results_dir = Path(args.legacy_results_dir)
    if not tri_results_dir.exists():
        nearby = _nearby_experiment_dirs(tri_results_dir)
        hint = f" Nearby experiment dirs: {nearby}" if nearby else ""
        raise FileNotFoundError(f"Tri results directory not found: {tri_results_dir}.{hint}")
    if not legacy_results_dir.exists():
        nearby = _nearby_experiment_dirs(legacy_results_dir)
        hint = f" Nearby experiment dirs: {nearby}" if nearby else ""
        raise FileNotFoundError(f"Legacy results directory not found: {legacy_results_dir}.{hint}")

    tri_metrics_dir = tri_results_dir / "metrics"
    legacy_metrics_dir = legacy_results_dir / "metrics"
    tri_val = _load_tri_split(tri_metrics_dir, "val")
    tri_shared = _load_tri_split(tri_metrics_dir, "shared1000")
    legacy_val = _load_legacy_split(legacy_metrics_dir, legacy_results_dir, "val", reference_split=tri_val)
    legacy_shared = _load_legacy_split(legacy_metrics_dir, legacy_results_dir, "shared1000", reference_split=tri_shared)

    _assert_split_disjointness(tri_val["nsd_ids"], tri_shared["nsd_ids"], "tri_results_dir")
    _assert_split_disjointness(legacy_val["nsd_ids"], legacy_shared["nsd_ids"], "legacy_results_dir")

    val_arrays = _align_common_ids(tri_val, legacy_val, "val")
    shared_arrays = _align_common_ids(tri_shared, legacy_shared, "shared1000")
    _assert_split_disjointness(val_arrays["nsd_ids"], shared_arrays["nsd_ids"], "aligned_tri_legacy")

    diagnostics_dir = tri_results_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    val_report, val_rows = _run_split_audit(val_arrays, "val", args.oracle_topk)
    shared_report, shared_rows = _run_split_audit(shared_arrays, "shared1000", args.oracle_topk)

    payload = {
        "topk": int(args.oracle_topk),
        "frozen_consensus_setting": DEFAULT_FIXED_TRI_SETTING,
        "val": val_report,
        "shared1000": shared_report,
    }
    summary_rows = [
        {
            "split": report["split"],
            "n_queries": report["n_queries"],
            "compact_csls_r1": report["baselines"].get("compact_csls", {}).get("R@1"),
            "mixture_csls_r1": report["baselines"].get("mixture_csls", {}).get("R@1"),
            "legacy_csls_r1": report["baselines"].get("legacy_csls", {}).get("R@1"),
            "fixed_tri_consensus_r1": report["fixed_tri_consensus"]["R@1"],
            "fixed_tri_mixture_r1": None if report["fixed_tri_mixture"] is None else report["fixed_tri_mixture"]["R@1"],
            "compact_legacy_union_oracle": report["union_oracle"]["compact_legacy"]["oracle_recall"],
            "mixture_legacy_union_oracle": None if "mixture_legacy" not in report["union_oracle"] else report["union_oracle"]["mixture_legacy"]["oracle_recall"],
            "mixture_compact_legacy_union_oracle": None if "mixture_compact_legacy" not in report["union_oracle"] else report["union_oracle"]["mixture_compact_legacy"]["oracle_recall"],
            "mixture_gain_over_compact_csls": report["mixture_gain_over_compact_csls"],
            "fixed_mixture_gain_over_fixed_consensus": report["fixed_mixture_gain_over_fixed_consensus"],
            "recoverable_fraction_among_misses": report["miss_taxonomy"].get("recoverable_fraction_among_misses"),
        }
        for report in [val_report, shared_report]
    ]

    json_path = diagnostics_dir / "mixture_headroom_audit.json"
    csv_path = diagnostics_dir / "mixture_headroom_audit_summary.csv"
    per_query_path = diagnostics_dir / "mixture_headroom_audit_per_query.csv"
    _save_json(json_path, payload)
    _write_csv(csv_path, summary_rows)
    _write_csv(per_query_path, val_rows + shared_rows)

    print("=" * 78)
    print("MIXTURE HEADROOM AUDIT")
    print("=" * 78)
    print(f"Tri results dir:    {tri_results_dir}")
    print(f"Legacy results dir: {legacy_results_dir}")
    print("")
    for report in [val_report, shared_report]:
        print(f"{report['split'].upper()} summary")
        print(f"  compact CSLS R@1:          {report['baselines']['compact_csls']['R@1']:.1%}")
        if 'mixture_csls' in report['baselines']:
            print(f"  mixture CSLS R@1:          {report['baselines']['mixture_csls']['R@1']:.1%}")
        print(f"  legacy CSLS R@1:           {report['baselines']['legacy_csls']['R@1']:.1%}")
        print(f"  fixed tri (consensus):     {report['fixed_tri_consensus']['R@1']:.1%}")
        if report['fixed_tri_mixture'] is not None:
            print(f"  fixed tri (mixture):       {report['fixed_tri_mixture']['R@1']:.1%}")
        print(f"  union oracle compact+legacy@{args.oracle_topk}: {report['union_oracle']['compact_legacy']['oracle_recall']:.1%}")
        if 'mixture_legacy' in report['union_oracle']:
            print(f"  union oracle mixture+legacy@{args.oracle_topk}: {report['union_oracle']['mixture_legacy']['oracle_recall']:.1%}")
        print("")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved per-query CSV: {per_query_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
