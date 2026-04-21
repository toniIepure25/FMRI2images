#!/usr/bin/env python3
"""
Sweep fusion weights with PPR scoring for V55 experiments.

Given saved predictions/kappas from compact and legacy experts, this script
sweeps over:
  - alpha/gamma weights for compact+legacy fusion
  - scoring methods: raw cosine, CSLS, PPR, PPR+CSLS
  - shortlist sizes: K = 50, 100, 150, 200

The output is a CSV + JSON with all fusion/scoring combinations ranked by R@1.

Usage:
    python scripts/evaluation/sweep_fusion_ppr.py \
        --compact-dir experimental_results/V55b_subj01_finetune/subj01 \
        --legacy-dir experimental_results/N1v28a_dual_head/subj01 \
        --output-dir experimental_results/V55b_subj01_finetune/subj01/fusion_sweep
"""

import argparse
import csv
import json
import logging
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sweep_fusion_ppr")


def _load_predictions(
    experiment_dir: Path,
    split: str = "shared1000",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load saved predictions, ground truth, and kappas.

    Returns:
        (predictions, ground_truth, kappas)
    """
    metrics_dir = experiment_dir / "metrics"
    if not metrics_dir.exists():
        metrics_dir = experiment_dir

    pred_file = metrics_dir / f"{split}_predictions.npy"
    gt_file = metrics_dir / f"{split}_ground_truth.npy"
    kappa_file = metrics_dir / f"{split}_kappas.npy"

    if not pred_file.exists():
        pred_file = metrics_dir / "val_predictions.npy"
        gt_file = metrics_dir / "val_ground_truth.npy"
        kappa_file = metrics_dir / "val_kappas.npy"

    if not pred_file.exists():
        raise FileNotFoundError(f"No predictions found in {metrics_dir}")

    preds = np.load(pred_file)
    gts = np.load(gt_file)
    kappas = np.load(kappa_file) if kappa_file.exists() else None

    norms = np.linalg.norm(preds, axis=-1, keepdims=True)
    preds = preds / np.maximum(norms, 1e-8)
    norms_gt = np.linalg.norm(gts, axis=-1, keepdims=True)
    gts = gts / np.maximum(norms_gt, 1e-8)

    logger.info(
        "Loaded %s: preds=%s, gts=%s, kappas=%s",
        split, preds.shape, gts.shape,
        kappas.shape if kappas is not None else "None",
    )
    return preds, gts, kappas


def _csls(sim: np.ndarray, k: int = 10) -> np.ndarray:
    """Apply CSLS correction to a similarity matrix."""
    k_eff = min(k, sim.shape[1] - 1, sim.shape[0] - 1)
    if k_eff < 1:
        return sim
    r_x = np.sort(sim, axis=1)[:, -k_eff:].mean(axis=1)
    r_y = np.sort(sim, axis=0)[-k_eff:, :].mean(axis=0)
    return 2.0 * sim - r_x[:, None] - r_y[None, :]


def _r_at_k(scores: np.ndarray, k: int = 1) -> float:
    """Compute R@k from a (N, N) score matrix (diagonal = correct)."""
    N = scores.shape[0]
    ranks = np.argsort(-scores, axis=1)
    correct = np.zeros(N, dtype=np.int32)
    for i in range(N):
        correct[i] = np.where(ranks[i] == i)[0][0]
    return float((correct < k).mean())


def _mrr(scores: np.ndarray) -> float:
    """Compute MRR from a (N, N) score matrix."""
    N = scores.shape[0]
    ranks = np.argsort(-scores, axis=1)
    correct = np.zeros(N, dtype=np.int32)
    for i in range(N):
        correct[i] = np.where(ranks[i] == i)[0][0]
    return float((1.0 / (correct + 1.0)).mean())


def _normalize_shortlist(scores: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize shortlist scores per-query."""
    if method == "zscore":
        mu = scores.mean(axis=1, keepdims=True)
        std = scores.std(axis=1, keepdims=True)
        return (scores - mu) / np.maximum(std, 1e-8)
    elif method == "minmax":
        lo = scores.min(axis=1, keepdims=True)
        hi = scores.max(axis=1, keepdims=True)
        return (scores - lo) / np.maximum(hi - lo, 1e-8)
    return scores


def _fused_retrieval(
    compact_scores: np.ndarray,
    legacy_scores: np.ndarray,
    alpha: float,
    gamma: float,
    shortlist_k: int,
    normalization: str = "zscore",
) -> Tuple[float, float, float]:
    """Run shortlist-based fusion and return (R@1, R@5, MRR).

    Uses compact scores to build shortlist, then fuses compact + legacy
    on that shortlist.
    """
    N = compact_scores.shape[0]
    compact_order = np.argsort(-compact_scores, axis=1)
    k_eff = min(shortlist_k, compact_scores.shape[1])
    shortlist = compact_order[:, :k_eff]
    row_idx = np.arange(N)[:, None]

    compact_sl = compact_scores[row_idx, shortlist]
    legacy_sl = legacy_scores[row_idx, shortlist]

    compact_sl_n = _normalize_shortlist(compact_sl, normalization)
    legacy_sl_n = _normalize_shortlist(legacy_sl, normalization)

    fused = alpha * compact_sl_n + gamma * legacy_sl_n

    gt_indices = np.arange(N)
    in_shortlist = np.any(shortlist == gt_indices[:, None], axis=1)

    fused_order = np.argsort(-fused, axis=1)
    fused_shortlist = shortlist[row_idx, fused_order]

    gt_rank = np.full(N, k_eff + 1, dtype=np.int32)
    hit_rows = np.where(in_shortlist)[0]
    if len(hit_rows) > 0:
        for i in hit_rows:
            pos = np.where(fused_shortlist[i] == gt_indices[i])[0]
            if len(pos) > 0:
                gt_rank[i] = pos[0] + 1

    r1 = float((gt_rank == 1).mean())
    r5 = float((gt_rank <= 5).mean())
    mrr_val = float((1.0 / gt_rank.astype(np.float64)).mean())

    return r1, r5, mrr_val


def sweep(
    compact_preds: np.ndarray,
    compact_gts: np.ndarray,
    compact_kappas: Optional[np.ndarray],
    legacy_preds: np.ndarray,
    legacy_gts: np.ndarray,
    legacy_kappas: Optional[np.ndarray],
    csls_k: int = 10,
) -> List[Dict]:
    """Run the full sweep over scoring methods, fusion weights, shortlist sizes."""
    from fmri2img.eval.ppr_scoring import (
        ppr_score_matrix,
        ppr_csls_score_matrix,
    )

    results = []

    compact_cosine = compact_preds @ compact_gts.T
    legacy_cosine = legacy_preds @ legacy_gts.T
    compact_csls_scores = _csls(compact_cosine, k=csls_k)
    legacy_csls_scores = _csls(legacy_cosine, k=csls_k)

    score_variants = {
        "cosine": (compact_cosine, legacy_cosine),
        "csls": (compact_csls_scores, legacy_csls_scores),
    }

    if compact_kappas is not None:
        d_c = compact_preds.shape[1]
        compact_ppr = ppr_score_matrix(compact_preds, compact_gts, compact_kappas, d=d_c)
        compact_ppr_csls = ppr_csls_score_matrix(
            compact_preds, compact_gts, compact_kappas, k=csls_k, d=d_c,
        )
        score_variants["ppr_compact"] = (compact_ppr, legacy_csls_scores)
        score_variants["ppr_csls_compact"] = (compact_ppr_csls, legacy_csls_scores)

    if legacy_kappas is not None:
        d_l = legacy_preds.shape[1]
        legacy_ppr = ppr_score_matrix(legacy_preds, legacy_gts, legacy_kappas, d=d_l)
        legacy_ppr_csls = ppr_csls_score_matrix(
            legacy_preds, legacy_gts, legacy_kappas, k=csls_k, d=d_l,
        )
        score_variants["ppr_legacy"] = (compact_csls_scores, legacy_ppr)
        score_variants["ppr_csls_legacy"] = (compact_csls_scores, legacy_ppr_csls)

    if compact_kappas is not None and legacy_kappas is not None:
        score_variants["ppr_both"] = (compact_ppr, legacy_ppr)
        score_variants["ppr_csls_both"] = (compact_ppr_csls, legacy_ppr_csls)

    # Individual expert metrics
    for name, (c_scores, l_scores) in score_variants.items():
        results.append({
            "scoring": name, "fusion": "compact_only", "alpha": 1.0, "gamma": 0.0,
            "shortlist_k": 0, "r@1": _r_at_k(c_scores, 1),
            "r@5": _r_at_k(c_scores, 5), "mrr": _mrr(c_scores),
        })
        results.append({
            "scoring": name, "fusion": "legacy_only", "alpha": 0.0, "gamma": 1.0,
            "shortlist_k": 0, "r@1": _r_at_k(l_scores, 1),
            "r@5": _r_at_k(l_scores, 5), "mrr": _mrr(l_scores),
        })

    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    shortlist_ks = [50, 100, 150, 200]

    for scoring_name, (c_scores, l_scores) in score_variants.items():
        for alpha, k in product(alpha_values, shortlist_ks):
            gamma = 1.0 - alpha
            r1, r5, mrr_val = _fused_retrieval(
                c_scores, l_scores, alpha, gamma, k, normalization="zscore",
            )
            results.append({
                "scoring": scoring_name, "fusion": "normalized_weighted",
                "alpha": alpha, "gamma": gamma, "shortlist_k": k,
                "r@1": r1, "r@5": r5, "mrr": mrr_val,
            })

    results.sort(key=lambda x: -x["r@1"])
    return results


def main():
    parser = argparse.ArgumentParser(description="Sweep fusion weights with PPR scoring")
    parser.add_argument("--compact-dir", type=str, required=True)
    parser.add_argument("--legacy-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--csls-k", type=int, default=10)
    args = parser.parse_args()

    compact_dir = Path(args.compact_dir)
    legacy_dir = Path(args.legacy_dir)
    out_dir = Path(args.output_dir) if args.output_dir else compact_dir / "fusion_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    compact_preds, compact_gts, compact_kappas = _load_predictions(compact_dir, args.split)
    legacy_preds, legacy_gts, legacy_kappas = _load_predictions(legacy_dir, args.split)

    N_c, N_l = len(compact_preds), len(legacy_preds)
    if N_c != N_l:
        logger.warning(
            "Prediction counts differ: compact=%d, legacy=%d. Using min.",
            N_c, N_l,
        )
        n = min(N_c, N_l)
        compact_preds, compact_gts = compact_preds[:n], compact_gts[:n]
        legacy_preds, legacy_gts = legacy_preds[:n], legacy_gts[:n]
        if compact_kappas is not None:
            compact_kappas = compact_kappas[:n]
        if legacy_kappas is not None:
            legacy_kappas = legacy_kappas[:n]

    results = sweep(
        compact_preds, compact_gts, compact_kappas,
        legacy_preds, legacy_gts, legacy_kappas,
        csls_k=args.csls_k,
    )

    csv_path = out_dir / "fusion_ppr_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    json_path = out_dir / "fusion_ppr_sweep.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved %d sweep results to %s", len(results), out_dir)
    logger.info("")
    logger.info("=== Top 10 Configurations ===")
    for i, r in enumerate(results[:10]):
        logger.info(
            "  #%2d  R@1=%.3f  R@5=%.3f  MRR=%.4f  scoring=%-20s  "
            "fusion=%-20s  alpha=%.1f  gamma=%.1f  K=%d",
            i + 1, r["r@1"], r["r@5"], r["mrr"],
            r["scoring"], r["fusion"], r["alpha"], r["gamma"], r["shortlist_k"],
        )

    best = results[0]
    logger.info("")
    logger.info(
        "BEST: R@1=%.4f  scoring=%s  fusion=%s  alpha=%.1f  gamma=%.1f  K=%d",
        best["r@1"], best["scoring"], best["fusion"],
        best["alpha"], best["gamma"], best["shortlist_k"],
    )


if __name__ == "__main__":
    main()
