#!/usr/bin/env python3
"""Inspect a built union-shortlist cache and print detailed diagnostics.

This utility is designed to verify cache correctness and diagnose whether
the candidate features carry meaningful signal for a reranker.

Usage:
    python scripts/evaluation/inspect_union_shortlist_cache.py \
        experimental_results/V35_legacy_teacher_distill/subj01/cache/union_shortlist_val_k100.npz

    # Inspect all caches in a directory:
    python scripts/evaluation/inspect_union_shortlist_cache.py \
        experimental_results/V35_legacy_teacher_distill/subj01/cache/ \
        --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _inspect_cache(path: Path, show_examples: int = 3) -> dict:
    """Inspect a single cache file and print diagnostics."""
    data = dict(np.load(path, allow_pickle=True))

    features = data["features"]       # (N, max_size, F)
    labels = data["labels"]           # (N, max_size)
    shortlists = data["shortlists"]   # (N, max_size)
    sizes = data["sizes"]             # (N,)
    sources = data.get("sources")     # (N, max_size, 2) or None

    n, max_size, f_dim = features.shape
    mask = shortlists >= 0

    # Load feature names from metadata if available
    meta_path = path.with_name(path.stem + "_meta.json")
    feature_names = None
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        feature_names = meta.get("feature_names")
    if feature_names is None or len(feature_names) != f_dim:
        feature_names = [f"f{i}" for i in range(f_dim)]

    print(f"\n{'='*70}")
    print(f"CACHE: {path.name}")
    print(f"{'='*70}")

    # ── Shape info ───────────────────────────────────────────────────
    print(f"\n  N queries:       {n}")
    print(f"  Max union size:  {max_size}")
    print(f"  Feature dim:     {f_dim}")

    # ── Union size stats ─────────────────────────────────────────────
    print(f"\n  Union sizes:  mean={sizes.mean():.1f}  median={np.median(sizes):.1f}  "
          f"min={sizes.min()}  max={sizes.max()}  std={sizes.std():.1f}")

    # ── GT-in-union ──────────────────────────────────────────────────
    gt_per_query = labels.sum(axis=1)
    gt_in_union = (gt_per_query > 0).mean()
    print(f"  GT in union:     {gt_in_union:.1%} ({int(gt_per_query.sum())}/{n})")
    multi_gt = (gt_per_query > 1).sum()
    if multi_gt > 0:
        print(f"  WARNING: {multi_gt} queries have >1 GT candidate in shortlist")

    # ── GT gallery indices ───────────────────────────────────────────
    if "gt_gallery_indices" in data:
        gti = data["gt_gallery_indices"]
        is_arange = np.array_equal(gti, np.arange(n))
        print(f"  GT gallery idx:  range=[{gti.min()}, {gti.max()}], is_arange={is_arange}")
    else:
        print(f"  GT gallery idx:  NOT IN CACHE (assumes arange)")

    # ── Source membership stats ──────────────────────────────────────
    if sources is not None:
        from_c = sources[:, :, 0] & mask
        from_l = sources[:, :, 1] & mask
        from_both = from_c & from_l
        c_only = from_c & ~from_l
        l_only = from_l & ~from_c

        total_valid = mask.sum()
        print(f"\n  Source membership (across all valid candidates):")
        print(f"    compact-only:  {c_only.sum():>8d}  ({c_only.sum()/total_valid:.1%})")
        print(f"    legacy-only:   {l_only.sum():>8d}  ({l_only.sum()/total_valid:.1%})")
        print(f"    both:          {from_both.sum():>8d}  ({from_both.sum()/total_valid:.1%})")

        # Per-query stats
        c_only_per_q = c_only.sum(axis=1)
        l_only_per_q = l_only.sum(axis=1)
        both_per_q = from_both.sum(axis=1)
        print(f"\n  Per-query source counts:")
        print(f"    compact-only:  mean={c_only_per_q.mean():.1f}  "
              f"median={np.median(c_only_per_q):.0f}  "
              f"pct_nonzero={float((c_only_per_q>0).mean()):.1%}")
        print(f"    legacy-only:   mean={l_only_per_q.mean():.1f}  "
              f"median={np.median(l_only_per_q):.0f}  "
              f"pct_nonzero={float((l_only_per_q>0).mean()):.1%}")
        print(f"    both:          mean={both_per_q.mean():.1f}  "
              f"median={np.median(both_per_q):.0f}")

        # GT source: where is GT found?
        gt_mask = labels.astype(bool)
        gt_in_compact = (gt_mask & from_c).any(axis=1)
        gt_in_legacy = (gt_mask & from_l).any(axis=1)
        gt_in_both = (gt_mask & from_both).any(axis=1)
        gt_c_only = (gt_in_compact & ~gt_in_legacy)
        gt_l_only = (gt_in_legacy & ~gt_in_compact)
        has_gt = gt_per_query > 0
        n_with_gt = has_gt.sum()
        if n_with_gt > 0:
            print(f"\n  GT source (among {n_with_gt} queries with GT):")
            print(f"    GT in compact-only: {gt_c_only.sum():>4d}  ({gt_c_only.sum()/n_with_gt:.1%})")
            print(f"    GT in legacy-only:  {gt_l_only.sum():>4d}  ({gt_l_only.sum()/n_with_gt:.1%})")
            print(f"    GT in both:         {gt_in_both.sum():>4d}  ({gt_in_both.sum()/n_with_gt:.1%})")

    # ── Feature statistics ───────────────────────────────────────────
    print(f"\n  {'Feature':<30s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s} {'const%':>7s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")

    suspicious_features = []
    for fi in range(f_dim):
        vals = features[:, :, fi][mask]
        f_mean = vals.mean()
        f_std = vals.std()
        f_min = vals.min()
        f_max = vals.max()
        # Constant rate: fraction of valid entries equal to the mode
        if len(vals) > 0:
            # Check if effectively constant
            unique_vals = np.unique(np.round(vals, 6))
            const_rate = (np.round(vals, 6) == unique_vals[0]).mean() if len(unique_vals) == 1 else 0.0
            if len(unique_vals) <= 2 and const_rate > 0.95:
                const_rate = (np.round(vals, 6) == unique_vals[0]).mean()
            else:
                const_rate = 0.0
        else:
            const_rate = 1.0

        flag = ""
        if const_rate > 0.99:
            flag = " *** CONSTANT"
            suspicious_features.append(feature_names[fi])
        elif f_std < 1e-6:
            flag = " *** ZERO-VAR"
            suspicious_features.append(feature_names[fi])

        print(f"  {feature_names[fi]:<30s} {f_mean:>8.4f} {f_std:>8.4f} {f_min:>8.4f} {f_max:>8.4f} "
              f"{const_rate:>6.1%}{flag}")

    if suspicious_features:
        print(f"\n  SUSPICIOUS: {len(suspicious_features)} feature(s) are constant/zero-var: "
              f"{suspicious_features}")
    else:
        print(f"\n  All features have meaningful variance.")

    # ── Top-1 agreement/disagreement ─────────────────────────────────
    # Find agreement features by name
    agree_idx = None
    for i, name in enumerate(feature_names):
        if name == "agree_compact_legacy_top1":
            agree_idx = i
            break

    if agree_idx is not None:
        # Agreement is broadcast per-query, so take one value per query
        agree_vals = features[:, 0, agree_idx]  # first candidate's value
        agree_rate = agree_vals.mean()
        disagree_rate = 1.0 - agree_rate
        print(f"\n  Top-1 expert agreement (compact vs legacy):")
        print(f"    agree:    {agree_rate:.1%}")
        print(f"    disagree: {disagree_rate:.1%}")
    else:
        print(f"\n  (No agree_compact_legacy_top1 feature found)")

    # ── Top-1 indicator sanity ───────────────────────────────────────
    for prefix in ["is_compact_top1", "is_legacy_top1", "is_rerank_top1"]:
        idx = None
        for i, name in enumerate(feature_names):
            if name == prefix:
                idx = i
                break
        if idx is not None:
            indicator = features[:, :, idx]
            sums = (indicator * mask).sum(axis=1)
            all_one = (sums == 1.0).all()
            print(f"  {prefix} sum-per-query: mean={sums.mean():.2f}, all_exactly_1={all_one}")

    # ── Per-query examples ───────────────────────────────────────────
    if show_examples > 0:
        rng = np.random.RandomState(42)
        example_ids = rng.choice(n, min(show_examples, n), replace=False)
        print(f"\n  === Example queries ===")
        for qi in example_ids:
            s = sizes[qi]
            sl = shortlists[qi, :s]
            lab = labels[qi, :s]
            gt_pos = np.where(lab > 0)[0]
            gt_cand = sl[gt_pos[0]] if len(gt_pos) > 0 else -1

            print(f"\n  Query {qi}: size={s}, GT_gallery={data.get('gt_gallery_indices', np.arange(n))[qi]}, "
                  f"GT_in_union={'YES' if len(gt_pos)>0 else 'NO'}")

            if sources is not None:
                c_only_q = (sources[qi, :s, 0] & ~sources[qi, :s, 1]).sum()
                l_only_q = (~sources[qi, :s, 0] & sources[qi, :s, 1]).sum()
                both_q = (sources[qi, :s, 0] & sources[qi, :s, 1]).sum()
                print(f"    Sources: compact_only={c_only_q}, legacy_only={l_only_q}, both={both_q}")

            # Show top-3 candidates by each expert score
            for score_name, fi in [("compact_csls", 1), ("legacy_csls", 4)]:
                if fi < f_dim:
                    local_scores = features[qi, :s, fi]
                    top3 = np.argsort(-local_scores)[:3]
                    items = [f"cand={sl[j]}(score={local_scores[j]:.4f})" for j in top3]
                    print(f"    {score_name} top-3: {', '.join(items)}")

            if len(gt_pos) > 0:
                print(f"    GT candidate {gt_cand} is at shortlist position {gt_pos[0]}")

    print()
    return {
        "n": n, "max_size": max_size, "f_dim": f_dim,
        "gt_in_union": float(gt_in_union),
        "suspicious_features": suspicious_features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect union-shortlist cache")
    parser.add_argument("path", type=str, help="Path to .npz cache file or directory")
    parser.add_argument("--all", action="store_true", help="Inspect all .npz caches in directory")
    parser.add_argument("--examples", type=int, default=3, help="Number of example queries to show")
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_dir() or args.all:
        d = path if path.is_dir() else path.parent
        npz_files = sorted(d.glob("union_shortlist_*.npz"))
        if not npz_files:
            print(f"No union_shortlist_*.npz files found in {d}")
            return
        for f in npz_files:
            _inspect_cache(f, show_examples=args.examples)
    elif path.suffix == ".npz":
        _inspect_cache(path, show_examples=args.examples)
    else:
        print(f"Expected .npz file or directory, got: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
