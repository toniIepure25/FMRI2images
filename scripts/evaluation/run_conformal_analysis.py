"""
Conformal Prediction for Neural Image Retrieval
=================================================

Correct formulation: prediction SETS with P(y_true in C(x)) >= 1-alpha.

For each query x:
  1. Compute sim(x, y_j) for all gallery items
  2. Calibration nonconformity: s_i = 1 - sim(pred_i, gallery[gt_i])
  3. Threshold: tau_hat = ceil((1-alpha)(1+1/n_cal))-quantile of {s_i}
  4. Prediction set: C(x) = {y_j : sim(x, y_j) >= 1 - tau_hat}

Key innovation: kappa-modulated nonconformity
  s_kappa(x, y) = (1 - sim(x, y)) / (kappa(x) / kappa_max)
  High-kappa queries get SMALLER prediction sets.
"""
import sys
sys.path.insert(0, "/home/jovyan/work/FMRI2images/src")

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO = Path("/home/jovyan/work/FMRI2images")
RESULTS_DIR = REPO / "experimental_results" / "calibrated_uncertainty"
OUTPUT_DIR = REPO / "experimental_results" / "conformal_prediction"

EXPERT_DIRS = {
    "V61a": REPO / "experimental_results/V61a_finetune_difflr/subj01/metrics",
    "V62a": REPO / "experimental_results/V62a_cls_retrieval_768d/subj01/metrics",
    "V66a": REPO / "experimental_results/V66a_roi_pretrain/subj01/metrics",
}


def load_clip_gallery(nsd_ids):
    """Load CLIP embeddings for SHARED1000 images."""
    clip_path = REPO / "outputs/clip_cache/clip.parquet"
    df = pd.read_parquet(clip_path)
    for col in ["fused", "final", "embedding", "clip_embedding"]:
        if col in df.columns:
            gallery_df = df[df["nsdId"].isin(nsd_ids)].sort_values("nsdId")
            gallery = np.stack(gallery_df[col].values).astype(np.float32)
            gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
            return gallery, gallery_df["nsdId"].values
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if emb_cols:
        gallery_df = df[df["nsdId"].isin(nsd_ids)].sort_values("nsdId")
        gallery = gallery_df[emb_cols].values.astype(np.float32)
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
        return gallery, gallery_df["nsdId"].values
    raise ValueError("No embedding column found in clip.parquet")


def calibrate_threshold(scores, alpha):
    """Compute conformal threshold: ceil((n+1)(1-alpha))/n quantile."""
    n = len(scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def conformal_retrieval_analysis(
    predictions: np.ndarray,
    gallery: np.ndarray,
    gt_indices: np.ndarray,
    kappas: np.ndarray,
    expert_correct: np.ndarray,
    alphas: List[float],
    n_splits: int = 50,
    seed: int = 42,
):
    """
    Full conformal retrieval analysis with multiple nonconformity scores.

    Returns results for each alpha and each score type.
    """
    n = len(predictions)
    m = gallery.shape[0]
    rng = np.random.RandomState(seed)

    sim = predictions @ gallery.T  # (n, m)

    # GT similarity for each query
    gt_sims = np.array([sim[i, gt_indices[i]] for i in range(n)])

    # Nonconformity scores (smaller = more conforming)
    raw_nonconf = 1.0 - gt_sims

    kappa_normalized = kappas / kappas.max()
    kappa_modulated_nonconf = raw_nonconf / np.maximum(kappa_normalized, 1e-8)

    # Agreement-based nonconformity
    agreement_frac = expert_correct.mean(axis=1)
    agreement_nonconf = raw_nonconf / np.maximum(agreement_frac, 0.01)

    # Margin (top1 - top2 similarity gap for true image)
    margins = np.empty(n, dtype=np.float64)
    for i in range(n):
        row = sim[i]
        sorted_row = np.sort(row)[::-1]
        margins[i] = sorted_row[0] - sorted_row[1]
    margin_weighted_nonconf = raw_nonconf / np.maximum(margins / margins.max(), 1e-8)

    score_types = {
        "raw_similarity": raw_nonconf,
        "kappa_modulated": kappa_modulated_nonconf,
        "agreement_modulated": agreement_nonconf,
        "margin_modulated": margin_weighted_nonconf,
    }

    results_by_score = {}

    for score_name, nonconf_scores in score_types.items():
        logger.info("\n=== Score: %s ===", score_name)
        alpha_results = []

        for alpha in alphas:
            # Multiple random cal/test splits for stability
            coverages = []
            set_sizes_list = []
            singleton_fracs = []

            for split_idx in range(n_splits):
                perm = rng.permutation(n)
                n_cal = n // 2
                cal_idx = perm[:n_cal]
                test_idx = perm[n_cal:]

                cal_nonconf = nonconf_scores[cal_idx]
                tau = calibrate_threshold(cal_nonconf, alpha)

                test_sim = sim[test_idx]
                test_gt = gt_indices[test_idx]
                test_nonconf = nonconf_scores[test_idx]

                # Coverage: is GT in the prediction set?
                test_covered = test_nonconf <= tau
                coverage = float(test_covered.mean())
                coverages.append(coverage)

                # Set sizes: for each test query, how many gallery items
                # have nonconformity <= tau?
                # For raw: set = {j: 1 - sim(x,y_j) <= tau} = {j: sim(x,y_j) >= 1 - tau}
                # For modulated: set = {j: (1-sim(x,y_j))/w(x) <= tau} = {j: sim(x,y_j) >= 1 - tau*w(x)}
                if score_name == "raw_similarity":
                    sim_threshold = 1.0 - tau
                    sizes = (test_sim >= sim_threshold).sum(axis=1)
                elif score_name == "kappa_modulated":
                    # Each query has its own effective threshold
                    test_kappa_norm = kappa_normalized[test_idx]
                    effective_tau = tau * test_kappa_norm
                    sim_thresholds = 1.0 - effective_tau
                    sizes = np.array([
                        (test_sim[i] >= sim_thresholds[i]).sum()
                        for i in range(len(test_idx))
                    ])
                elif score_name == "agreement_modulated":
                    test_agree = agreement_frac[test_idx]
                    effective_tau = tau * np.maximum(test_agree, 0.01)
                    sim_thresholds = 1.0 - effective_tau
                    sizes = np.array([
                        (test_sim[i] >= sim_thresholds[i]).sum()
                        for i in range(len(test_idx))
                    ])
                elif score_name == "margin_modulated":
                    test_margins = margins[test_idx]
                    test_margin_norm = test_margins / margins.max()
                    effective_tau = tau * np.maximum(test_margin_norm, 1e-8)
                    sim_thresholds = 1.0 - effective_tau
                    sizes = np.array([
                        (test_sim[i] >= sim_thresholds[i]).sum()
                        for i in range(len(test_idx))
                    ])

                set_sizes_list.append(sizes)
                singleton_fracs.append(float((sizes == 1).mean()))

            all_sizes = np.concatenate(set_sizes_list)
            result = {
                "alpha": alpha,
                "target_coverage": 1 - alpha,
                "mean_coverage": float(np.mean(coverages)),
                "std_coverage": float(np.std(coverages)),
                "coverage_ci_95": [
                    float(np.percentile(coverages, 2.5)),
                    float(np.percentile(coverages, 97.5)),
                ],
                "coverage_valid": float(np.mean(coverages)) >= (1 - alpha - 0.02),
                "mean_set_size": float(all_sizes.mean()),
                "median_set_size": float(np.median(all_sizes)),
                "std_set_size": float(all_sizes.std()),
                "max_set_size": int(all_sizes.max()),
                "min_set_size": int(all_sizes.min()),
                "frac_singleton": float(np.mean(singleton_fracs)),
                "frac_leq5": float((all_sizes <= 5).mean()),
                "frac_leq10": float((all_sizes <= 10).mean()),
                "frac_leq50": float((all_sizes <= 50).mean()),
                "gallery_size": m,
            }

            logger.info(
                "  α=%.2f | cov=%.3f±%.3f (target %.2f) %s | "
                "sets: mean=%.1f, med=%.0f, sing=%.1f%%, ≤5=%.1f%%, ≤10=%.1f%%",
                alpha,
                result["mean_coverage"], result["std_coverage"],
                result["target_coverage"],
                "✓" if result["coverage_valid"] else "✗",
                result["mean_set_size"], result["median_set_size"],
                result["frac_singleton"] * 100,
                result["frac_leq5"] * 100,
                result["frac_leq10"] * 100,
            )
            alpha_results.append(result)

        results_by_score[score_name] = alpha_results

    return results_by_score


def kappa_stratified_analysis(
    predictions, gallery, gt_indices, kappas, alpha=0.10, n_splits=50, seed=42
):
    """Analyze set sizes stratified by kappa quartiles."""
    n = len(predictions)
    rng = np.random.RandomState(seed)
    sim = predictions @ gallery.T
    gt_sims = np.array([sim[i, gt_indices[i]] for i in range(n)])
    raw_nonconf = 1.0 - gt_sims

    quartile_bounds = np.percentile(kappas, [25, 50, 75])
    quartile_labels = ["Q1 (low κ)", "Q2", "Q3", "Q4 (high κ)"]
    bounds = [-np.inf] + list(quartile_bounds) + [np.inf]

    results = {}
    for ql, lo, hi in zip(quartile_labels, bounds[:-1], bounds[1:]):
        mask = (kappas >= lo) & (kappas < hi)
        results[ql] = {"n": int(mask.sum()), "kappa_range": [float(lo), float(hi)]}

    for split_idx in range(n_splits):
        perm = rng.permutation(n)
        n_cal = n // 2
        cal_idx = perm[:n_cal]
        test_idx = perm[n_cal:]

        cal_nonconf = raw_nonconf[cal_idx]
        tau = calibrate_threshold(cal_nonconf, alpha)

        sim_threshold = 1.0 - tau
        test_sim = sim[test_idx]
        test_gt = gt_indices[test_idx]
        test_kappas = kappas[test_idx]
        test_nonconf = raw_nonconf[test_idx]

        sizes = (test_sim >= sim_threshold).sum(axis=1)
        covered = test_nonconf <= tau

        for ql, lo, hi in zip(quartile_labels, bounds[:-1], bounds[1:]):
            mask = (test_kappas >= lo) & (test_kappas < hi)
            if mask.sum() == 0:
                continue
            q_sizes = sizes[mask]
            q_covered = covered[mask]

            key = f"split_{split_idx}"
            if key not in results[ql]:
                results[ql]["coverages"] = []
                results[ql]["mean_sizes"] = []
                results[ql]["median_sizes"] = []
            results[ql]["coverages"].append(float(q_covered.mean()))
            results[ql]["mean_sizes"].append(float(q_sizes.mean()))
            results[ql]["median_sizes"].append(float(np.median(q_sizes)))

    for ql in quartile_labels:
        if "coverages" in results[ql]:
            results[ql]["mean_coverage"] = float(np.mean(results[ql]["coverages"]))
            results[ql]["mean_set_size"] = float(np.mean(results[ql]["mean_sizes"]))
            results[ql]["median_set_size"] = float(np.mean(results[ql]["median_sizes"]))
            del results[ql]["coverages"]
            del results[ql]["mean_sizes"]
            del results[ql]["median_sizes"]

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load canonical nsd_ids
    for edir in EXPERT_DIRS.values():
        nid_file = edir / "shared1000_nsd_ids.npy"
        if nid_file.exists():
            nsd_ids = np.load(nid_file)
            break
    n_images = len(nsd_ids)
    logger.info("SHARED1000: %d images", n_images)

    # Load CLIP gallery
    gallery, gallery_nsd_ids = load_clip_gallery(nsd_ids)
    nsd_to_gallery = {int(nid): i for i, nid in enumerate(gallery_nsd_ids)}
    gt_indices = np.array([nsd_to_gallery[int(nid)] for nid in nsd_ids])
    logger.info("Gallery: %s", gallery.shape)

    # Load expert predictions (768-D CLS)
    expert_preds = {}
    for name, edir in EXPERT_DIRS.items():
        if not edir.exists():
            continue
        preds_file = edir / "shared1000_predictions.npy"
        if name == "V61a":
            mctta = edir / "shared1000_predictions_mctta16.npy"
            if mctta.exists():
                preds_file = mctta
        preds = np.load(preds_file)
        if preds.shape[1] > 768:
            preds = preds[:, :768].copy()
        preds = preds / np.linalg.norm(preds, axis=1, keepdims=True)
        expert_preds[name] = preds
        
        r1 = float((np.argmax(preds @ gallery.T, axis=1) == gt_indices).mean())
        logger.info("  %s: loaded (%s), cosine R@1=%.3f", name, preds.shape, r1)

    # Load kappas
    kappas = {}
    for name in expert_preds:
        kf = RESULTS_DIR / f"{name}_image_kappas.npy"
        if kf.exists():
            kappas[name] = np.load(kf)

    # Expert correctness matrix
    expert_correct = np.column_stack([
        (np.argmax(expert_preds[n] @ gallery.T, axis=1) == gt_indices)
        for n in sorted(expert_preds.keys())
    ])

    # Run analysis for V62a (clean 768-D, primary expert for conformal)
    primary = "V62a"
    preds = expert_preds[primary]
    kappa = kappas.get(primary, np.ones(n_images))

    logger.info("\n" + "=" * 70)
    logger.info("CONFORMAL RETRIEVAL ANALYSIS — Primary: %s", primary)
    logger.info("=" * 70)

    alphas = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]

    sweep_results = conformal_retrieval_analysis(
        preds, gallery, gt_indices, kappa, expert_correct,
        alphas=alphas, n_splits=50, seed=42,
    )

    # Kappa-stratified set sizes
    logger.info("\n" + "=" * 70)
    logger.info("KAPPA-STRATIFIED SET SIZES (raw similarity, α=0.10)")
    logger.info("=" * 70)

    stratified = kappa_stratified_analysis(
        preds, gallery, gt_indices, kappa, alpha=0.10, n_splits=50,
    )
    for ql, stats in stratified.items():
        logger.info(
            "  %s (n=%d): coverage=%.3f, mean_set=%.1f, median_set=%.0f",
            ql, stats["n"],
            stats.get("mean_coverage", float("nan")),
            stats.get("mean_set_size", float("nan")),
            stats.get("median_set_size", float("nan")),
        )

    # Run for all experts
    logger.info("\n" + "=" * 70)
    logger.info("PER-EXPERT CONFORMAL COMPARISON (α=0.10)")
    logger.info("=" * 70)

    per_expert_results = {}
    for name, preds_e in expert_preds.items():
        kappa_e = kappas.get(name, np.ones(n_images))
        res = conformal_retrieval_analysis(
            preds_e, gallery, gt_indices, kappa_e, expert_correct,
            alphas=[0.10], n_splits=50, seed=42,
        )
        per_expert_results[name] = res

    # Compute set size reduction from kappa modulation
    logger.info("\n" + "=" * 70)
    logger.info("SET SIZE REDUCTION: kappa_modulated vs raw_similarity")
    logger.info("=" * 70)

    for score_type in sweep_results:
        alpha_010 = [r for r in sweep_results[score_type] if r["alpha"] == 0.10]
        if alpha_010:
            r = alpha_010[0]
            logger.info(
                "  %s: mean_set=%.1f, median_set=%.0f, coverage=%.3f",
                score_type, r["mean_set_size"], r["median_set_size"],
                r["mean_coverage"],
            )

    # Save all results
    final_results = {
        "metadata": {
            "dataset": "NSD SHARED1000",
            "subject": "subj01",
            "n_images": n_images,
            "gallery_size": gallery.shape[0],
            "primary_expert": primary,
            "experts": list(expert_preds.keys()),
            "n_splits": 50,
        },
        "conformal_sweep": sweep_results,
        "kappa_stratified_sets": stratified,
        "per_expert_alpha010": {
            name: {
                score: [r for r in results[score] if r["alpha"] == 0.10][0]
                for score in results
            }
            for name, results in per_expert_results.items()
        },
    }

    out_path = OUTPUT_DIR / "conformal_analysis_subj01.json"
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    logger.info("\nResults saved to %s", out_path)

    # Save nonconformity scores for cross-subject transfer
    sim = preds @ gallery.T
    gt_sims = np.array([sim[i, gt_indices[i]] for i in range(n_images)])
    raw_nonconf = 1.0 - gt_sims

    np.savez(
        OUTPUT_DIR / "nonconformity_scores_subj01.npz",
        raw_nonconf=raw_nonconf,
        kappas_V62a=kappa,
        kappas_V61a=kappas.get("V61a", np.ones(n_images)),
        kappas_V66a=kappas.get("V66a", np.ones(n_images)),
        expert_correct=expert_correct,
        gt_indices=gt_indices,
        nsd_ids=nsd_ids,
    )

    logger.info("\n===== ANALYSIS COMPLETE =====")


if __name__ == "__main__":
    main()
