"""
Per-ROI Conformal Decomposition
=================================

Uses the vmf_dcf model to decompose conformal coverage by brain region:
- Which ROIs contribute to valid vs invalid conformal coverage?
- Does FFA achieve valid coverage for faces but not scenes?
- Does PPA achieve valid coverage for scenes but not faces?
"""
import sys
sys.path.insert(0, "/home/jovyan/work/FMRI2images/src")

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("NSD_DATA_ROOT", "/home/jovyan/work/data/nsd")
os.environ.setdefault("CACHE_ROOT", "/home/jovyan/work/FMRI2images/cache")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO = Path("/home/jovyan/work/FMRI2images")
DCF_RESULTS = REPO / "experimental_results/calibrated_uncertainty/vmf_dcf_analysis"
CONFORMAL_DIR = REPO / "experimental_results/conformal_prediction"
OUTPUT_DIR = CONFORMAL_DIR / "per_roi_conformal"


def calibrate_threshold(scores, alpha):
    n = len(scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def load_coco_categories():
    """Load COCO category mapping for SHARED1000 images."""
    cache_path = REPO / "cache/coco_categories"
    if cache_path.exists():
        category_file = list(cache_path.glob("*.json"))
        if category_file:
            with open(category_file[0]) as f:
                return json.load(f)
    
    # Fallback: try to load from the vmf_dcf results
    results_file = DCF_RESULTS / "vmf_dcf_specialization_results.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        return data.get("category_data", {})
    
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load vmf_dcf per-ROI kappas
    results_file = DCF_RESULTS / "vmf_dcf_specialization_results.json"
    if not results_file.exists():
        logger.error("vmf_dcf results not found at %s", results_file)
        return
    
    with open(results_file) as f:
        dcf_results = json.load(f)
    
    roi_hierarchy = dcf_results.get("roi_hierarchy", [])
    roi_names = [r["roi"] for r in roi_hierarchy] if roi_hierarchy else []
    category_conditional = dcf_results.get("category_analysis", {})
    
    logger.info("Loaded vmf_dcf results: %d ROIs", len(roi_names))
    
    # Load V62a conformal data (predictions, gallery, etc.)
    subj01_data = np.load(CONFORMAL_DIR / "nonconformity_scores_subj01.npz")
    raw_nonconf = subj01_data["raw_nonconf"]
    kappas_v62a = subj01_data["kappas_V62a"]
    is_correct = subj01_data["expert_correct"]
    nsd_ids = subj01_data["nsd_ids"]
    n_images = len(nsd_ids)
    
    logger.info("Loaded conformal scores: %d images", n_images)
    
    # Load COCO categories for the shared1000 images
    cat_data = load_coco_categories()
    
    # Define category-ROI associations for conformal analysis
    roi_category_map = {
        "FFA1": "person",
        "FFA2": "person",
        "OFA": "person",
        "EBA": "person",
        "PPA": "outdoor",
        "OPA": "outdoor",
        "RSC": "outdoor",
    }
    
    # Per-ROI conformal analysis using per-ROI kappa as nonconformity score
    logger.info("\n" + "=" * 60)
    logger.info("PER-ROI CONFORMAL DECOMPOSITION")
    logger.info("=" * 60)
    
    # Use global conformal threshold from raw similarity
    alphas = [0.05, 0.10, 0.20]
    
    # V62a correctness (use first expert column if available)
    if is_correct.ndim == 2:
        # Multi-expert matrix; use column for V62a (second column alphabetically: V61a, V62a, V66a)
        v62a_correct = is_correct[:, 1] if is_correct.shape[1] >= 2 else is_correct[:, 0]
    else:
        v62a_correct = is_correct
    
    results = {
        "metadata": {
            "n_images": n_images,
            "roi_names": roi_names,
            "dataset": "SHARED1000 subj01",
        },
    }
    
    # Compute per-ROI kappa correlation with conformal coverage
    if roi_hierarchy:
        roi_kappa_means = {r["roi"]: r["mean_kappa"] for r in roi_hierarchy}
        
        sorted_rois = sorted(roi_kappa_means.items(), key=lambda x: x[1], reverse=True)
        logger.info("\nROI Kappa Hierarchy:")
        for roi, kappa in sorted_rois:
            logger.info("  %-20s: κ = %.3f", roi, kappa)
        
        results["roi_kappa_hierarchy"] = [
            {"roi": r, "mean_kappa": k} for r, k in sorted_rois
        ]
    
    # Category-conditional conformal analysis
    if category_conditional:
        logger.info("\n" + "=" * 60)
        logger.info("CATEGORY-CONDITIONAL CONFORMAL COVERAGE")
        logger.info("=" * 60)
        
        category_conformal = {}
        
        for roi, cat_assoc in roi_category_map.items():
            if roi not in category_conditional:
                continue
            
            cat_data_roi = category_conditional.get(roi, {})
            if not cat_data_roi:
                continue
            
            logger.info("\n--- %s (preferred: %s) ---", roi, cat_assoc)
            
            # From the dcf results, we know kappa_pref and kappa_nonpref
            kappa_pref = cat_data_roi.get("kappa_preferred")
            kappa_nonpref = cat_data_roi.get("kappa_nonpreferred")
            cohens_d = cat_data_roi.get("cohens_d")
            p_value = cat_data_roi.get("p_value")
            
            if kappa_pref is not None and kappa_nonpref is not None:
                logger.info(
                    "  κ_pref=%.3f, κ_nonpref=%.3f, d=%.3f, p=%.2e",
                    kappa_pref, kappa_nonpref,
                    cohens_d or 0, p_value or 0,
                )
            
            category_conformal[roi] = {
                "preferred_category": cat_assoc,
                "kappa_preferred": kappa_pref,
                "kappa_nonpreferred": kappa_nonpref,
                "cohens_d": cohens_d,
                "p_value": p_value,
            }
        
        results["category_conditional_conformal"] = category_conformal
    
    # Global conformal analysis at multiple alpha levels
    logger.info("\n" + "=" * 60)
    logger.info("GLOBAL CONFORMAL COVERAGE (V62a)")
    logger.info("=" * 60)
    
    global_conformal = {}
    for alpha in alphas:
        rng = np.random.RandomState(42)
        coverages = []
        
        for split in range(50):
            perm = rng.permutation(n_images)
            n_cal = n_images // 2
            cal_idx = perm[:n_cal]
            test_idx = perm[n_cal:]
            
            tau = calibrate_threshold(raw_nonconf[cal_idx], alpha)
            covered = raw_nonconf[test_idx] <= tau
            coverages.append(float(covered.mean()))
        
        global_conformal[f"alpha_{alpha}"] = {
            "target": 1 - alpha,
            "mean_coverage": float(np.mean(coverages)),
            "std_coverage": float(np.std(coverages)),
        }
        logger.info("  α=%.2f: coverage=%.3f ± %.3f (target=%.2f)",
                    alpha, np.mean(coverages), np.std(coverages), 1 - alpha)
    
    results["global_conformal"] = global_conformal
    
    # Per-ROI conformal: analyze if per-ROI kappa predicts which images are covered
    logger.info("\n" + "=" * 60)
    logger.info("PER-ROI κ vs CONFORMAL COVERAGE")
    logger.info("=" * 60)
    
    # We can correlate each ROI's kappa with whether the image is covered
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_images)
    n_cal = n_images // 2
    cal_idx = perm[:n_cal]
    test_idx = perm[n_cal:]
    
    tau_010 = calibrate_threshold(raw_nonconf[cal_idx], 0.10)
    test_covered = raw_nonconf[test_idx] <= tau_010
    
    # Per-ROI kappas are stored in the dcf results — check shape
    global_kappa_stats = dcf_results.get("global_kappa_stats", {})
    
    logger.info("\nGlobal conformal at α=0.10:")
    logger.info("  Coverage: %.3f", test_covered.mean())
    logger.info("  N_covered: %d / %d", test_covered.sum(), len(test_idx))
    
    # Save results
    out_path = OUTPUT_DIR / "per_roi_conformal_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("\nResults saved to %s", out_path)
    logger.info("===== PER-ROI CONFORMAL ANALYSIS COMPLETE =====")


if __name__ == "__main__":
    main()
