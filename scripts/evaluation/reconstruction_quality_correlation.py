"""
Reconstruction Quality vs. Uncertainty Correlation Analysis
============================================================

Analyzes correlation between vMF kappa (model confidence) and reconstruction
quality metrics. Two modes:

1. **Proxy mode** (default): Uses retrieval similarity as a proxy for
   reconstruction quality. No diffusion model needed.
2. **Full mode** (requires GPU + SD 2.1): Runs actual diffusion reconstruction
   on top/bottom 100 images by kappa, computes PixCorr/SSIM/LPIPS.

Usage:
    python scripts/evaluation/reconstruction_quality_correlation.py [--full]
"""
import sys
sys.path.insert(0, "/home/jovyan/work/FMRI2images/src")

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO = Path("/home/jovyan/work/FMRI2images")
CONFORMAL_DIR = REPO / "experimental_results" / "conformal_prediction"
OUTPUT_DIR = CONFORMAL_DIR / "recon_uncertainty"


def proxy_analysis(subject="subj01"):
    """Use retrieval similarity as proxy for reconstruction quality."""
    if subject == "subj01":
        pred_dir = REPO / "experimental_results/V62a_cls_retrieval_768d/subj01/metrics"
        preds = np.load(pred_dir / "shared1000_predictions.npy")
        gt = np.load(pred_dir / "shared1000_ground_truth.npy")
        kappas = np.load(
            REPO / "experimental_results/calibrated_uncertainty/V62a_image_kappas.npy"
        )
    else:
        subj_dir = CONFORMAL_DIR / subject
        preds = np.load(subj_dir / "shared1000_predictions.npy")
        gt = np.load(subj_dir / "shared1000_ground_truth.npy")
        kappas = np.load(subj_dir / "shared1000_kappas.npy")

    if preds.shape[1] > 768:
        preds = preds[:, :768].copy()
        gt = gt[:, :768].copy()

    preds /= np.linalg.norm(preds, axis=1, keepdims=True)
    gt /= np.linalg.norm(gt, axis=1, keepdims=True)

    cosine_sims = np.sum(preds * gt, axis=1)
    mse = np.mean((preds - gt) ** 2, axis=1)

    from scipy.stats import spearmanr, pearsonr

    rho_cosine, p_cosine = spearmanr(kappas, cosine_sims)
    rho_mse, p_mse = spearmanr(kappas, -mse)
    pearson_cosine, pp_cosine = pearsonr(kappas, cosine_sims)

    n = len(kappas)
    q25 = np.percentile(kappas, 25)
    q75 = np.percentile(kappas, 75)

    low_mask = kappas <= q25
    high_mask = kappas >= q75

    results = {
        "subject": subject,
        "n_images": n,
        "proxy_metric": "cosine_similarity",
        "kappa_stats": {
            "mean": float(kappas.mean()),
            "std": float(kappas.std()),
            "min": float(kappas.min()),
            "max": float(kappas.max()),
        },
        "correlation": {
            "spearman_kappa_vs_cosine": {"rho": float(rho_cosine), "p": float(p_cosine)},
            "spearman_kappa_vs_neg_mse": {"rho": float(rho_mse), "p": float(p_mse)},
            "pearson_kappa_vs_cosine": {"r": float(pearson_cosine), "p": float(pp_cosine)},
        },
        "stratified": {
            "low_kappa_q1": {
                "n": int(low_mask.sum()),
                "kappa_mean": float(kappas[low_mask].mean()),
                "cosine_mean": float(cosine_sims[low_mask].mean()),
                "cosine_std": float(cosine_sims[low_mask].std()),
                "mse_mean": float(mse[low_mask].mean()),
            },
            "high_kappa_q4": {
                "n": int(high_mask.sum()),
                "kappa_mean": float(kappas[high_mask].mean()),
                "cosine_mean": float(cosine_sims[high_mask].mean()),
                "cosine_std": float(cosine_sims[high_mask].std()),
                "mse_mean": float(mse[high_mask].mean()),
            },
            "effect_size_cohens_d": float(
                (cosine_sims[high_mask].mean() - cosine_sims[low_mask].mean())
                / np.sqrt(
                    (cosine_sims[high_mask].std() ** 2 + cosine_sims[low_mask].std() ** 2) / 2
                )
            ),
        },
    }

    logger.info("=== Proxy Analysis: %s ===", subject)
    logger.info("Spearman(κ, cosine): ρ=%.3f (p=%.2e)", rho_cosine, p_cosine)
    logger.info("Pearson(κ, cosine):  r=%.3f (p=%.2e)", pearson_cosine, pp_cosine)
    logger.info("Low-κ Q1:  cos=%.3f±%.3f (n=%d)", cosine_sims[low_mask].mean(),
                cosine_sims[low_mask].std(), low_mask.sum())
    logger.info("High-κ Q4: cos=%.3f±%.3f (n=%d)", cosine_sims[high_mask].mean(),
                cosine_sims[high_mask].std(), high_mask.sum())
    logger.info("Cohen's d: %.3f", results["stratified"]["effect_size_cohens_d"])

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Run full diffusion reconstruction (requires GPU)")
    parser.add_argument("--subjects", nargs="+",
                        default=["subj01", "subj02", "subj05", "subj07"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for subject in args.subjects:
        try:
            results = proxy_analysis(subject)
            all_results[subject] = results
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", subject, e)

    with open(OUTPUT_DIR / "recon_uncertainty_proxy.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("\nResults saved to %s", OUTPUT_DIR / "recon_uncertainty_proxy.json")

    if args.full:
        logger.warning("Full reconstruction mode not yet implemented. "
                       "Requires diffusion model on GPU pod.")


if __name__ == "__main__":
    main()
