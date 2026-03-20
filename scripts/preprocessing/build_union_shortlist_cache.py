#!/usr/bin/env python3
"""Build offline union-shortlist cache for V39 residual reranker training.

For each query image in a split (train / val / shared1000), this script:
1. Loads expert predictions and ground-truth embeddings
2. Computes cosine and CSLS score matrices
3. Builds union shortlists from compact CSLS + legacy CSLS top-K
4. Extracts per-candidate feature vectors
5. Saves the cache as a .npz file

The TRAIN cache is built from the model's own training-set predictions.
This requires that train_unified.py saved train-split predictions during
the final epoch or at checkpoint time. If train predictions are unavailable,
the script will attempt to use val predictions as a proxy (with a warning).

Usage:
    python scripts/preprocessing/build_union_shortlist_cache.py \
        experimental_results/V35_legacy_teacher_distill/subj01 \
        experimental_results/N1v28a_dual_head/subj01 \
        --shortlist-k 100 \
        --splits train val shared1000

Outputs:
    {tri_results_dir}/cache/union_shortlist_train_k100.npz
    {tri_results_dir}/cache/union_shortlist_val_k100.npz
    {tri_results_dir}/cache/union_shortlist_shared1000_k100.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from sweep_tri_fusion_retrieval import (  # noqa: E402
    _align_common_ids,
    _build_split_scores,
    _cosine_sim,
    _csls_scores,
    _gt_rank_from_scores,
    _load_legacy_split,
    _load_tri_split,
    _metrics_from_gt_rank,
    _nearby_experiment_dirs,
    _rank_within_shortlist,
    _save_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _top_margin(local_scores: np.ndarray, top_n: int) -> np.ndarray:
    """Score gap between rank-1 and rank-top_n within each row."""
    if local_scores.shape[1] < 2:
        return np.zeros(local_scores.shape[0], dtype=np.float32)
    sorted_scores = np.sort(local_scores, axis=1)[:, ::-1]
    idx = min(max(top_n - 1, 1), sorted_scores.shape[1] - 1)
    return (sorted_scores[:, 0] - sorted_scores[:, idx]).astype(np.float32)


def _build_union_shortlist(
    compact_order: np.ndarray,
    legacy_order: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed-width union shortlists padded to max_union_size.

    Returns:
        shortlists: (N, max_size) int32 array, -1 padded
        sizes: (N,) int32 array of actual sizes
        sources: (N, max_size, 2) bool array — [compact_source, legacy_source]
    """
    n = compact_order.shape[0]
    # Build variable-length shortlists
    raw_shortlists = []
    raw_sources = []
    max_size = 0
    for i in range(n):
        compact_set = set(compact_order[i, :k].tolist())
        legacy_set = set(legacy_order[i, :k].tolist())
        union = sorted(compact_set | legacy_set)
        raw_shortlists.append(np.array(union, dtype=np.int32))
        src = np.zeros((len(union), 2), dtype=bool)
        for j, cand in enumerate(union):
            src[j, 0] = cand in compact_set
            src[j, 1] = cand in legacy_set
        raw_sources.append(src)
        max_size = max(max_size, len(union))

    # Pad to fixed width
    shortlists = np.full((n, max_size), -1, dtype=np.int32)
    sources = np.zeros((n, max_size, 2), dtype=bool)
    sizes = np.empty(n, dtype=np.int32)
    for i in range(n):
        sl = raw_shortlists[i]
        shortlists[i, : len(sl)] = sl
        sources[i, : len(sl)] = raw_sources[i]
        sizes[i] = len(sl)

    return shortlists, sizes, sources


def _safe_argmax(vals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """argmax over axis=1, ignoring positions where mask is False.

    Uses -inf masking so negative scores are handled correctly
    (unlike vals * mask which treats 0 as > negative scores).
    """
    masked = np.where(mask, vals, -np.inf)
    return np.argmax(masked, axis=1)


def _shortlist_zscore(vals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-query z-score normalization within valid shortlist candidates."""
    masked = np.where(mask, vals, np.nan)
    mu = np.nanmean(masked, axis=1, keepdims=True)
    sigma = np.nanstd(masked, axis=1, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    result = (vals - mu) / sigma
    return np.where(mask, result, 0.0)


def _shortlist_margin_to_top1(vals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-candidate margin to the query's top-1 score."""
    masked = np.where(mask, vals, -np.inf)
    top1_score = np.max(masked, axis=1, keepdims=True)
    return np.where(mask, vals - top1_score, 0.0)


def _extract_candidate_features(
    scores: dict[str, np.ndarray],
    shortlists: np.ndarray,
    sizes: np.ndarray,
    sources: np.ndarray,
    kappas: np.ndarray | None,
) -> np.ndarray:
    """Extract per-candidate feature vectors for the union shortlist.

    Returns:
        features: (N, max_size, F) float32 array, F = number of features
    """
    n, max_size = shortlists.shape
    row_idx = np.arange(n)[:, None]

    # Gather local scores for each candidate in the shortlist
    # For padded positions (-1), we index into column 0 (arbitrary, masked later)
    safe_sl = np.where(shortlists >= 0, shortlists, 0)
    mask = shortlists >= 0

    local_compact_raw = scores["compact_raw"][row_idx, safe_sl]
    local_compact_csls = scores["compact_csls"][row_idx, safe_sl]
    local_rerank = scores["rerank"][row_idx, safe_sl]
    local_legacy_raw = scores["legacy_raw"][row_idx, safe_sl]
    local_legacy_csls = scores["legacy_csls"][row_idx, safe_sl]

    # ── A. Raw ranks within shortlist (1-based) ──────────────────────
    def _local_rank(vals: np.ndarray) -> np.ndarray:
        masked = np.where(mask, vals, -np.inf)
        return _rank_within_shortlist(masked).astype(np.float32)

    rank_compact_csls = _local_rank(local_compact_csls)
    rank_legacy_csls = _local_rank(local_legacy_csls)
    rank_rerank = _local_rank(local_rerank)

    # ── B. Normalized ranks (rank / union_size per query) ────────────
    sizes_2d = sizes[:, None].astype(np.float32)
    rank_compact_csls_norm = rank_compact_csls / sizes_2d
    rank_legacy_csls_norm = rank_legacy_csls / sizes_2d
    rank_rerank_norm = rank_rerank / sizes_2d

    # ── C. Reciprocal ranks: 1/(rank) ───────────────────────────────
    rr_compact_csls = np.where(mask, 1.0 / np.maximum(rank_compact_csls, 1.0), 0.0)
    rr_legacy_csls = np.where(mask, 1.0 / np.maximum(rank_legacy_csls, 1.0), 0.0)
    rr_rerank = np.where(mask, 1.0 / np.maximum(rank_rerank, 1.0), 0.0)

    # ── D. Score calibration: z-scores within shortlist ──────────────
    zscore_compact_csls = _shortlist_zscore(local_compact_csls, mask)
    zscore_legacy_csls = _shortlist_zscore(local_legacy_csls, mask)
    zscore_rerank = _shortlist_zscore(local_rerank, mask)

    # ── E. Margin to top-1 per expert ────────────────────────────────
    margin_compact_csls = _shortlist_margin_to_top1(local_compact_csls, mask)
    margin_legacy_csls = _shortlist_margin_to_top1(local_legacy_csls, mask)
    margin_rerank = _shortlist_margin_to_top1(local_rerank, mask)

    # ── F. Cross-expert score differences ────────────────────────────
    diff_compact_legacy = local_compact_csls - local_legacy_csls
    diff_compact_rerank = local_compact_csls - local_rerank
    diff_rerank_legacy = local_rerank - local_legacy_csls

    # ── G. Cross-expert rank differences (normalized) ────────────────
    rank_gap_compact_legacy = (rank_compact_csls - rank_legacy_csls) / sizes_2d
    rank_gap_compact_rerank = (rank_compact_csls - rank_rerank) / sizes_2d

    # ── H. Source membership flags ───────────────────────────────────
    from_compact = sources[:, :, 0].astype(np.float32)
    from_legacy = sources[:, :, 1].astype(np.float32)
    from_both = (sources[:, :, 0] & sources[:, :, 1]).astype(np.float32)
    compact_only = (sources[:, :, 0] & ~sources[:, :, 1]).astype(np.float32)
    legacy_only = (~sources[:, :, 0] & sources[:, :, 1]).astype(np.float32)

    # ── I. Top-1 agreement/disagreement (FIX: use -inf masking) ──────
    # BUG FIX: old code used `scores * mask` which fails when scores
    # are negative (CSLS). Padded zeros beat negative valid scores.
    compact_top1_pos = _safe_argmax(local_compact_csls, mask)
    legacy_top1_pos = _safe_argmax(local_legacy_csls, mask)
    rerank_top1_pos = _safe_argmax(local_rerank, mask)

    # Per-query agreement (broadcast to all candidates)
    agree_cl = (compact_top1_pos == legacy_top1_pos).astype(np.float32)
    agree_cr = (compact_top1_pos == rerank_top1_pos).astype(np.float32)
    agree_cl_bc = np.broadcast_to(agree_cl[:, None], (n, max_size)).copy()
    agree_cr_bc = np.broadcast_to(agree_cr[:, None], (n, max_size)).copy()

    # Per-candidate top-1 indicators
    pos_range = np.arange(max_size)[None, :]
    is_compact_top1 = (pos_range == compact_top1_pos[:, None]).astype(np.float32)
    is_legacy_top1 = (pos_range == legacy_top1_pos[:, None]).astype(np.float32)
    is_rerank_top1 = (pos_range == rerank_top1_pos[:, None]).astype(np.float32)
    # Candidate is top-1 for BOTH compact and legacy
    is_both_top1 = (is_compact_top1 * is_legacy_top1).astype(np.float32)

    # ── J. Top-K indicators ─────────────────────────────────────────
    is_compact_top5 = (rank_compact_csls <= 5).astype(np.float32)
    is_legacy_top5 = (rank_legacy_csls <= 5).astype(np.float32)
    is_compact_top10 = (rank_compact_csls <= 10).astype(np.float32)
    is_legacy_top10 = (rank_legacy_csls <= 10).astype(np.float32)
    in_both_top5 = (is_compact_top5 * is_legacy_top5).astype(np.float32)
    in_both_top10 = (is_compact_top10 * is_legacy_top10).astype(np.float32)

    # ── K. Kappa feature (per-query, broadcast) ─────────────────────
    if kappas is not None:
        kappa_q = np.broadcast_to(
            kappas.astype(np.float32).reshape(n, -1).mean(axis=1, keepdims=True),
            (n, max_size),
        ).copy()
    else:
        kappa_q = np.zeros((n, max_size), dtype=np.float32)

    # ── Stack all features: (N, max_size, F) ─────────────────────────
    feature_list = [
        # Raw scores (5)
        local_compact_raw,          # 0
        local_compact_csls,         # 1
        local_rerank,               # 2
        local_legacy_raw,           # 3
        local_legacy_csls,          # 4
        # Normalized ranks (3)
        rank_compact_csls_norm,     # 5
        rank_legacy_csls_norm,      # 6
        rank_rerank_norm,           # 7
        # Reciprocal ranks (3)
        rr_compact_csls,            # 8
        rr_legacy_csls,             # 9
        rr_rerank,                  # 10
        # Z-scored shortlist scores (3)
        zscore_compact_csls,        # 11
        zscore_legacy_csls,         # 12
        zscore_rerank,              # 13
        # Margin to top-1 (3)
        margin_compact_csls,        # 14
        margin_legacy_csls,         # 15
        margin_rerank,              # 16
        # Cross-expert score diffs (3)
        diff_compact_legacy,        # 17
        diff_compact_rerank,        # 18
        diff_rerank_legacy,         # 19
        # Cross-expert rank gaps (2)
        rank_gap_compact_legacy,    # 20
        rank_gap_compact_rerank,    # 21
        # Source flags (5)
        from_compact,               # 22
        from_legacy,                # 23
        from_both,                  # 24
        compact_only,               # 25
        legacy_only,                # 26
        # Agreement structure (6)
        agree_cl_bc,                # 27  per-query: compact & legacy agree on top1?
        agree_cr_bc,                # 28  per-query: compact & rerank agree on top1?
        is_compact_top1,            # 29
        is_legacy_top1,             # 30
        is_rerank_top1,             # 31
        is_both_top1,               # 32
        # Top-K indicators (6)
        is_compact_top5,            # 33
        is_legacy_top5,             # 34
        is_compact_top10,           # 35
        is_legacy_top10,            # 36
        in_both_top5,               # 37
        in_both_top10,              # 38
        # Query-level (1)
        kappa_q,                    # 39
    ]
    features = np.stack(feature_list, axis=-1).astype(np.float32)

    # Zero out features for padded positions
    features[~mask] = 0.0

    # ── Sanity checks ────────────────────────────────────────────────
    # Top-1 flags should sum to exactly 1 per expert per query (among valid)
    for name, indicator in [("compact", is_compact_top1), ("legacy", is_legacy_top1)]:
        sums = (indicator * mask).sum(axis=1)
        bad = (sums != 1.0).sum()
        if bad > 0:
            logger.warning("SANITY: %s_top1 flag sums != 1 for %d/%d queries", name, bad, n)

    # Agreement should NOT be all-ones if oracle audit shows disagreement
    agree_rate = agree_cl.mean()
    logger.info("  Top-1 compact-legacy agreement: %.1f%% (expect ~45%% for val/shared1000)",
                agree_rate * 100)

    return features


FEATURE_NAMES = [
    # Raw scores (5)
    "compact_raw_score", "compact_csls_score", "rerank_score",
    "legacy_raw_score", "legacy_csls_score",
    # Normalized ranks (3)
    "compact_csls_rank_norm", "legacy_csls_rank_norm", "rerank_rank_norm",
    # Reciprocal ranks (3)
    "compact_csls_rr", "legacy_csls_rr", "rerank_rr",
    # Z-scored shortlist scores (3)
    "compact_csls_zscore", "legacy_csls_zscore", "rerank_zscore",
    # Margin to top-1 (3)
    "margin_compact_csls", "margin_legacy_csls", "margin_rerank",
    # Cross-expert score diffs (3)
    "diff_compact_legacy", "diff_compact_rerank", "diff_rerank_legacy",
    # Cross-expert rank gaps (2)
    "rank_gap_compact_legacy", "rank_gap_compact_rerank",
    # Source flags (5)
    "from_compact", "from_legacy", "from_both", "compact_only", "legacy_only",
    # Agreement structure (6)
    "agree_compact_legacy_top1", "agree_compact_rerank_top1",
    "is_compact_top1", "is_legacy_top1", "is_rerank_top1", "is_both_top1",
    # Top-K indicators (6)
    "is_compact_top5", "is_legacy_top5",
    "is_compact_top10", "is_legacy_top10",
    "in_both_top5", "in_both_top10",
    # Query-level (1)
    "kappa_query",
]


def _build_labels(shortlists: np.ndarray, gt_gallery_indices: np.ndarray) -> np.ndarray:
    """Build binary labels: 1 if candidate == query's GT gallery index, else 0.

    Args:
        shortlists: (N, max_size) gallery indices per candidate
        gt_gallery_indices: (N,) the GT gallery index for each query
            (for position-aligned splits this is just arange(N))
    """
    labels = (shortlists == gt_gallery_indices[:, None]).astype(np.int8)
    return labels


def _build_cache_for_split(
    split_arrays: dict[str, np.ndarray],
    split_name: str,
    shortlist_k: int,
) -> dict[str, np.ndarray]:
    """Build complete cache for one split."""
    n = split_arrays["compact_preds"].shape[0]
    logger.info("Building cache for %s: %d queries, K=%d", split_name, n, shortlist_k)

    scores = _build_split_scores(split_arrays)

    # Expert orderings (CSLS for shortlist construction)
    compact_order, compact_gt_rank = _gt_rank_from_scores(scores["compact_csls"])
    legacy_order, legacy_gt_rank = _gt_rank_from_scores(scores["legacy_csls"])

    # Build union shortlists
    shortlists, sizes, sources = _build_union_shortlist(compact_order, legacy_order, shortlist_k)
    logger.info("  Union sizes: mean=%.1f, median=%.1f, max=%d",
                np.mean(sizes), np.median(sizes), np.max(sizes))

    # Labels — GT for query i is gallery index i (position-aligned splits)
    gt_gallery_indices = np.arange(n, dtype=np.int32)
    labels = _build_labels(shortlists, gt_gallery_indices)
    gt_in_union = float(np.any(labels == 1, axis=1).mean())
    logger.info("  GT in union: %.1f%%", gt_in_union * 100)

    # Features
    kappas = split_arrays.get("compact_kappas")
    features = _extract_candidate_features(scores, shortlists, sizes, sources, kappas)
    logger.info("  Feature matrix: %s (%.1f MB)", features.shape, features.nbytes / 1e6)

    # Baselines
    compact_csls_r1 = float(np.mean(compact_gt_rank <= 1))
    legacy_csls_r1 = float(np.mean(legacy_gt_rank <= 1))

    nsd_ids = split_arrays.get("nsd_ids")

    cache = {
        "features": features,           # (N, max_size, F)
        "labels": labels,               # (N, max_size) int8
        "shortlists": shortlists,       # (N, max_size) int32, gallery indices
        "sizes": sizes,                 # (N,) int32
        "sources": sources,             # (N, max_size, 2) bool
        "gt_gallery_indices": gt_gallery_indices,  # (N,) int32, GT gallery idx per query
    }
    if nsd_ids is not None:
        cache["nsd_ids"] = nsd_ids.astype(np.int32)

    # Save metadata as JSON string
    metadata = {
        "split": split_name,
        "n_queries": int(n),
        "shortlist_k": shortlist_k,
        "max_union_size": int(shortlists.shape[1]),
        "mean_union_size": float(np.mean(sizes)),
        "gt_in_union_rate": gt_in_union,
        "feature_dim": int(features.shape[-1]),
        "feature_names": FEATURE_NAMES,
        "compact_csls_r1": compact_csls_r1,
        "legacy_csls_r1": legacy_csls_r1,
    }
    cache["metadata_json"] = np.array([json.dumps(metadata)], dtype=object)

    return cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build union-shortlist cache for V39 residual reranker"
    )
    parser.add_argument(
        "tri_results_dir", type=str,
        help="Triple-head results dir, e.g. experimental_results/V35_legacy_teacher_distill/subj01",
    )
    parser.add_argument(
        "legacy_results_dir", type=str,
        help="Legacy N1v28a results dir, e.g. experimental_results/N1v28a_dual_head/subj01",
    )
    parser.add_argument(
        "--shortlist-k", type=int, default=100,
        help="Top-K per expert for union shortlist (default: 100)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["val", "shared1000"],
        choices=["train", "val", "shared1000"],
        help="Which splits to build caches for (default: val shared1000)",
    )
    args = parser.parse_args()

    tri_results_dir = Path(args.tri_results_dir)
    legacy_results_dir = Path(args.legacy_results_dir)

    if not tri_results_dir.exists():
        nearby = _nearby_experiment_dirs(tri_results_dir)
        raise FileNotFoundError(f"Not found: {tri_results_dir}. Nearby: {nearby}")
    if not legacy_results_dir.exists():
        nearby = _nearby_experiment_dirs(legacy_results_dir)
        raise FileNotFoundError(f"Not found: {legacy_results_dir}. Nearby: {nearby}")

    tri_metrics = tri_results_dir / "metrics"
    legacy_metrics = legacy_results_dir / "metrics"

    cache_dir = tri_results_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load and align each split
    val_tri = None
    for split_name in args.splits:
        logger.info("=== Processing %s split ===", split_name.upper())

        try:
            tri_split = _load_tri_split(tri_metrics, split_name)
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", split_name, e)
            continue

        ref = val_tri if val_tri is not None else None
        try:
            legacy_split = _load_legacy_split(
                legacy_metrics, legacy_results_dir, split_name, reference_split=ref
            )
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", split_name, e)
            continue

        if split_name == "val":
            val_tri = tri_split

        aligned = _align_common_ids(tri_split, legacy_split, split_name)
        cache = _build_cache_for_split(aligned, split_name, args.shortlist_k)

        out_path = cache_dir / f"union_shortlist_{split_name}_k{args.shortlist_k}.npz"
        np.savez_compressed(out_path, **{k: v for k, v in cache.items() if k != "metadata_json"})

        # Save metadata separately as JSON for easy inspection
        meta = json.loads(cache["metadata_json"][0])
        meta_path = cache_dir / f"union_shortlist_{split_name}_k{args.shortlist_k}_meta.json"
        _save_json(meta_path, meta)

        logger.info("Saved cache to %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
        logger.info("Saved metadata to %s", meta_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
