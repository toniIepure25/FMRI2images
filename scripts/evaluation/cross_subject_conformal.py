"""
Cross-Subject Conformal Transfer for Neural Image Decoding
============================================================

Tests whether conformal guarantees transfer across subjects:
  1. Calibrate conformal threshold on subj01
  2. Apply to subj02/05/07 — measure coverage gap
  3. Apply weighted conformal prediction to correct for distribution shift
  4. Compare kappa-based vs margin-based scores for transferability
"""
import sys
sys.path.insert(0, "/home/jovyan/work/FMRI2images/src")

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO = Path("/home/jovyan/work/FMRI2images")
CONFORMAL_DIR = REPO / "experimental_results" / "conformal_prediction"
CAL_DIR = REPO / "experimental_results" / "calibrated_uncertainty"
OUTPUT_DIR = CONFORMAL_DIR / "cross_subject"

SUBJECTS = ["subj01", "subj02", "subj05", "subj07"]
CAL_SUBJECT = "subj01"


def calibrate_threshold(scores, alpha):
    n = len(scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def load_subject_data(subject):
    """Load predictions, kappas, and GT for a subject."""
    if subject == "subj01":
        # Load from calibrated_uncertainty + V62a experiment
        pred_dir = REPO / "experimental_results/V62a_cls_retrieval_768d/subj01/metrics"
        preds = np.load(pred_dir / "shared1000_predictions.npy")
        gt = np.load(pred_dir / "shared1000_ground_truth.npy")
        nsd_ids = np.load(pred_dir / "shared1000_nsd_ids.npy")
        kappas = np.load(CAL_DIR / "V62a_image_kappas.npy")
        
        if preds.shape[1] > 768:
            preds = preds[:, :768].copy()
            gt = gt[:, :768].copy()
    else:
        subj_dir = CONFORMAL_DIR / subject
        preds = np.load(subj_dir / "shared1000_predictions.npy")
        gt = np.load(subj_dir / "shared1000_ground_truth.npy")
        nsd_ids = np.load(subj_dir / "shared1000_nsd_ids.npy")
        kappas = np.load(subj_dir / "shared1000_kappas.npy")
    
    preds /= np.linalg.norm(preds, axis=1, keepdims=True)
    gt /= np.linalg.norm(gt, axis=1, keepdims=True)
    
    return preds, gt, nsd_ids, kappas


def compute_nonconformity(preds, gallery, gt_indices, kappas):
    """Compute multiple nonconformity scores."""
    sim = preds @ gallery.T
    n = len(preds)
    
    # Raw similarity
    gt_sims = np.array([sim[i, gt_indices[i]] for i in range(n)])
    raw = 1.0 - gt_sims
    
    # Kappa-modulated
    kappa_norm = kappas / kappas.max()
    kappa_mod = raw / np.maximum(kappa_norm, 1e-8)
    
    # Margin
    margins = np.empty(n, dtype=np.float64)
    for i in range(n):
        row = sim[i]
        sorted_row = np.sort(row)[::-1]
        margins[i] = sorted_row[0] - sorted_row[1]
    margin_mod = raw / np.maximum(margins / margins.max(), 1e-8)
    
    # Top-1 correctness
    top1 = np.argmax(sim, axis=1)
    is_correct = top1 == gt_indices
    
    return {
        "raw": raw,
        "kappa_modulated": kappa_mod,
        "margin_modulated": margin_mod,
        "is_correct": is_correct,
        "r@1": float(is_correct.mean()),
        "kappas": kappas,
    }


def cross_subject_analysis(data_by_subject, alphas=[0.05, 0.10, 0.20]):
    """Run cross-subject conformal transfer analysis."""
    cal_data = data_by_subject[CAL_SUBJECT]
    results = {}
    
    for score_type in ["raw", "kappa_modulated", "margin_modulated"]:
        logger.info("\n=== Score: %s ===", score_type)
        score_results = {}
        
        for alpha in alphas:
            # Calibrate on source subject
            cal_scores = cal_data[score_type]
            tau = calibrate_threshold(cal_scores, alpha)
            
            # Test on all subjects
            per_subject = {}
            for subj, data in data_by_subject.items():
                test_scores = data[score_type]
                covered = test_scores <= tau
                coverage = float(covered.mean())
                target = 1 - alpha
                gap = target - coverage
                
                per_subject[subj] = {
                    "coverage": coverage,
                    "target": target,
                    "gap": gap,
                    "n": len(test_scores),
                    "acceptance_rate": float(covered.mean()),
                    "r@1": data["r@1"],
                    "is_self": subj == CAL_SUBJECT,
                }
                
                logger.info(
                    "  α=%.2f %s: coverage=%.3f (target %.2f, gap=%+.3f) R@1=%.3f %s",
                    alpha, subj, coverage, target, gap, data["r@1"],
                    "✓" if coverage >= target - 0.05 else "✗"
                )
            
            score_results[f"alpha_{alpha}"] = per_subject
        
        results[score_type] = score_results
    
    return results


def weighted_conformal_analysis(data_by_subject, alpha=0.10):
    """Apply weighted conformal prediction to correct for subject shift."""
    cal_data = data_by_subject[CAL_SUBJECT]
    results = {}
    
    for score_type in ["raw", "kappa_modulated"]:
        logger.info("\n=== Weighted CP: %s ===", score_type)
        cal_scores = cal_data[score_type]
        cal_kappas = cal_data["kappas"]
        
        # Standard threshold
        tau_standard = calibrate_threshold(cal_scores, alpha)
        
        for subj, data in data_by_subject.items():
            if subj == CAL_SUBJECT:
                continue
            
            test_scores = data[score_type]
            test_kappas = data["kappas"]
            
            # Standard coverage
            std_coverage = float((test_scores <= tau_standard).mean())
            
            # Weighted CP: use kappa ratio as importance weight
            # w(x) ∝ p_test(κ) / p_cal(κ) — approximate with kernel density
            from scipy.stats import gaussian_kde
            try:
                cal_kde = gaussian_kde(cal_kappas)
                test_kde = gaussian_kde(test_kappas)
                
                weights = test_kde(cal_kappas) / np.maximum(cal_kde(cal_kappas), 1e-8)
                weights = weights / weights.sum()
                
                # Weighted threshold
                sorted_idx = np.argsort(cal_scores)
                sorted_scores = cal_scores[sorted_idx]
                sorted_weights = weights[sorted_idx]
                cumsum = np.cumsum(sorted_weights)
                idx = np.searchsorted(cumsum, 1 - alpha)
                idx = min(idx, len(sorted_scores) - 1)
                tau_weighted = float(sorted_scores[idx])
                
                weighted_coverage = float((test_scores <= tau_weighted).mean())
                
                improvement = weighted_coverage - std_coverage
                
                results[f"{score_type}_{subj}"] = {
                    "standard_coverage": std_coverage,
                    "weighted_coverage": weighted_coverage,
                    "improvement": improvement,
                    "standard_threshold": tau_standard,
                    "weighted_threshold": tau_weighted,
                    "target": 1 - alpha,
                }
                
                logger.info(
                    "  %s: standard=%.3f → weighted=%.3f (improvement=%+.3f, target=%.2f)",
                    subj, std_coverage, weighted_coverage, improvement, 1 - alpha,
                )
            except Exception as e:
                logger.warning("  Weighted CP failed for %s: %s", subj, e)
                results[f"{score_type}_{subj}"] = {
                    "standard_coverage": std_coverage,
                    "error": str(e),
                }
    
    return results


def per_subject_calibration(data_by_subject, alphas=[0.05, 0.10, 0.20]):
    """Compare cross-subject transfer vs per-subject calibration."""
    results = {}
    
    for score_type in ["raw", "kappa_modulated"]:
        logger.info("\n=== Per-subject vs cross-subject: %s ===", score_type)
        
        for alpha in alphas:
            for subj, data in data_by_subject.items():
                scores = data[score_type]
                n = len(scores)
                
                # Per-subject calibration (random split)
                rng = np.random.RandomState(42)
                perm = rng.permutation(n)
                n_cal = n // 2
                cal_idx = perm[:n_cal]
                test_idx = perm[n_cal:]
                
                tau_self = calibrate_threshold(scores[cal_idx], alpha)
                self_coverage = float((scores[test_idx] <= tau_self).mean())
                
                # Cross-subject threshold from subj01
                cal_scores = data_by_subject[CAL_SUBJECT][score_type]
                tau_cross = calibrate_threshold(cal_scores, alpha)
                cross_coverage = float((scores <= tau_cross).mean())
                
                gap = self_coverage - cross_coverage
                
                key = f"{score_type}_alpha{alpha}_{subj}"
                results[key] = {
                    "self_calibrated_coverage": self_coverage,
                    "cross_subject_coverage": cross_coverage,
                    "gap": gap,
                    "target": 1 - alpha,
                }
                
                logger.info(
                    "  α=%.2f %s: self=%.3f, cross=%.3f (gap=%+.3f)",
                    alpha, subj, self_coverage, cross_coverage, gap,
                )
    
    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data for all subjects
    data_by_subject = {}
    available_subjects = []
    
    for subj in SUBJECTS:
        try:
            preds, gt, nsd_ids, kappas = load_subject_data(subj)
            
            # Build gallery from gt (same images for all subjects)
            nsd_to_idx = {int(nid): i for i, nid in enumerate(nsd_ids)}
            gt_indices = np.arange(len(nsd_ids))
            
            nonconf = compute_nonconformity(preds, gt, gt_indices, kappas)
            data_by_subject[subj] = nonconf
            available_subjects.append(subj)
            
            logger.info("Loaded %s: %d images, R@1=%.3f, kappa=%.2f±%.2f",
                        subj, len(preds), nonconf["r@1"],
                        kappas.mean(), kappas.std())
        except FileNotFoundError:
            logger.warning("Data not found for %s, skipping", subj)
    
    if len(available_subjects) < 2:
        logger.error("Need at least 2 subjects for cross-subject analysis (have %d)",
                     len(available_subjects))
        # Save partial results
        results = {
            "status": "insufficient_subjects",
            "available": available_subjects,
            "message": "Multi-subject training still in progress. "
                      "Re-run after training completes.",
        }
        with open(OUTPUT_DIR / "cross_subject_conformal.json", "w") as f:
            json.dump(results, f, indent=2)
        return
    
    # Run analyses
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-SUBJECT CONFORMAL TRANSFER")
    logger.info("=" * 70)
    transfer_results = cross_subject_analysis(data_by_subject)
    
    logger.info("\n" + "=" * 70)
    logger.info("WEIGHTED CONFORMAL PREDICTION")
    logger.info("=" * 70)
    weighted_results = weighted_conformal_analysis(data_by_subject)
    
    logger.info("\n" + "=" * 70)
    logger.info("PER-SUBJECT vs CROSS-SUBJECT CALIBRATION")
    logger.info("=" * 70)
    calibration_comparison = per_subject_calibration(data_by_subject)
    
    # Save results
    all_results = {
        "metadata": {
            "calibration_subject": CAL_SUBJECT,
            "test_subjects": [s for s in available_subjects if s != CAL_SUBJECT],
            "n_subjects": len(available_subjects),
        },
        "per_subject_metrics": {
            subj: {"r@1": data["r@1"], "kappa_mean": float(data["kappas"].mean()),
                    "kappa_std": float(data["kappas"].std())}
            for subj, data in data_by_subject.items()
        },
        "cross_subject_transfer": transfer_results,
        "weighted_conformal": weighted_results,
        "calibration_comparison": calibration_comparison,
    }
    
    with open(OUTPUT_DIR / "cross_subject_conformal.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info("\nResults saved to %s", OUTPUT_DIR / "cross_subject_conformal.json")
    logger.info("===== CROSS-SUBJECT ANALYSIS COMPLETE =====")


if __name__ == "__main__":
    main()
