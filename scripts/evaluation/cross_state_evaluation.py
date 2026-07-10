"""
Cross-State Evaluation: Perception vs Mental Imagery
=====================================================

Runs trained decoder models on NSD-Imagery data and compares performance
between perception and imagery conditions.

Core hypothesis: Uncertainty-aware (vMF) models will show appropriately
LOWER kappa on imagery trials, producing graceful degradation rather than
confident hallucinations (as point-estimate models do).

Evaluation protocol:
1. Load trained ROI-DCF model (from perception training)
2. Run inference on paired perception + imagery trials (same nsd_ids)
3. Compare: kappa distributions, retrieval accuracy, reconstruction quality
4. Compute Appropriate Uncertainty Ratio (AUR)

Novel metric - AUR:
    AUR = correlation(model_confidence, actual_reconstruction_fidelity)
    A well-calibrated model has AUR >> 0 (high confidence -> good output)
    An overconfident model has AUR ≈ 0 (confidence is uninformative)

References:
    - Kneeland et al. (2025) NSD-Imagery, CVPR
    - Plan Paper 2, Contribution 3: "Cross-State Generalization"
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


def compute_appropriate_uncertainty_ratio(
    confidences: np.ndarray,
    fidelities: np.ndarray,
) -> Dict[str, float]:
    """
    Compute the Appropriate Uncertainty Ratio (AUR).

    AUR measures whether model confidence correlates with actual output quality.
    A model that "knows what it doesn't know" has high AUR.

    Parameters
    ----------
    confidences : (N,) model confidence values (e.g., kappa)
    fidelities : (N,) actual quality metric (e.g., cosine similarity to GT)

    Returns
    -------
    Dict with AUR statistics
    """
    if len(confidences) < 3:
        return {"aur": 0.0, "p_value": 1.0, "interpretation": "insufficient data"}

    pearson_r, pearson_p = sp_stats.pearsonr(confidences, fidelities)
    spearman_rho, spearman_p = sp_stats.spearmanr(confidences, fidelities)

    return {
        "aur_pearson": float(pearson_r),
        "aur_spearman": float(spearman_rho),
        "p_value_pearson": float(pearson_p),
        "p_value_spearman": float(spearman_p),
        "well_calibrated": pearson_r > 0.3 and pearson_p < 0.01,
        "interpretation": (
            "Excellent calibration: confidence reliably predicts quality"
            if pearson_r > 0.5 else
            "Good calibration: confidence is informative"
            if pearson_r > 0.3 else
            "Poor calibration: confidence is uninformative"
            if pearson_r > 0.1 else
            "Overconfident: confidence does not track quality"
        ),
    }


def run_cross_state_evaluation(
    perception_mus: np.ndarray,
    perception_kappas: np.ndarray,
    imagery_mus: np.ndarray,
    imagery_kappas: np.ndarray,
    ground_truth_embeddings: np.ndarray,
    per_roi_kappas_perception: Optional[np.ndarray] = None,
    per_roi_kappas_imagery: Optional[np.ndarray] = None,
    roi_names: Optional[List[str]] = None,
) -> Dict:
    """
    Full cross-state evaluation comparing perception and imagery decoding.

    Parameters
    ----------
    perception_mus : (N, D) predicted embeddings from perception data
    perception_kappas : (N,) consensus kappa from perception
    imagery_mus : (N, D) predicted embeddings from imagery data
    imagery_kappas : (N,) consensus kappa from imagery
    ground_truth_embeddings : (N, D) CLIP embeddings of the stimuli
    per_roi_kappas_perception : (N, R) optional per-ROI for perception
    per_roi_kappas_imagery : (N, R) optional per-ROI for imagery
    roi_names : ROI names

    Returns
    -------
    Comprehensive evaluation dict
    """
    n_samples = len(perception_mus)

    # Normalize
    def normalize(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    perc_norm = normalize(perception_mus)
    img_norm = normalize(imagery_mus)
    gt_norm = normalize(ground_truth_embeddings)

    # Per-sample cosine similarities
    perc_cosine = np.sum(perc_norm * gt_norm, axis=-1)
    img_cosine = np.sum(img_norm * gt_norm, axis=-1)

    # ---- 1. Kappa distribution comparison ----
    kappa_comparison = {
        "perception": {
            "mean": float(perception_kappas.mean()),
            "std": float(perception_kappas.std()),
            "median": float(np.median(perception_kappas)),
        },
        "imagery": {
            "mean": float(imagery_kappas.mean()),
            "std": float(imagery_kappas.std()),
            "median": float(np.median(imagery_kappas)),
        },
        "kappa_drop": float(
            (perception_kappas.mean() - imagery_kappas.mean())
            / perception_kappas.mean()
        ),
        "paired_ttest": {},
    }

    t_stat, p_val = sp_stats.ttest_rel(perception_kappas, imagery_kappas)
    effect_size = (
        (perception_kappas.mean() - imagery_kappas.mean())
        / np.sqrt((perception_kappas.std() ** 2 + imagery_kappas.std() ** 2) / 2)
    )
    kappa_comparison["paired_ttest"] = {
        "t": float(t_stat), "p": float(p_val), "cohens_d": float(effect_size),
    }

    # ---- 2. Retrieval performance comparison ----
    def compute_retrieval(preds, gts):
        sim = normalize(preds) @ normalize(gts).T
        ranks = np.zeros(len(preds))
        for i in range(len(preds)):
            ranks[i] = (sim[i] > sim[i, i]).sum() + 1
        r_at_1 = (ranks == 1).mean()
        r_at_5 = (ranks <= 5).mean()
        mrr = (1.0 / ranks).mean()
        return {"r@1": float(r_at_1), "r@5": float(r_at_5), "mrr": float(mrr),
                "median_rank": float(np.median(ranks))}

    retrieval_perception = compute_retrieval(perception_mus, ground_truth_embeddings)
    retrieval_imagery = compute_retrieval(imagery_mus, ground_truth_embeddings)

    # ---- 3. AUR for both conditions ----
    aur_perception = compute_appropriate_uncertainty_ratio(
        perception_kappas, perc_cosine
    )
    aur_imagery = compute_appropriate_uncertainty_ratio(
        imagery_kappas, img_cosine
    )

    # ---- 4. Graceful degradation analysis ----
    # Does imagery performance drop proportionally to confidence drop?
    perf_drop = perc_cosine.mean() - img_cosine.mean()
    conf_drop = perception_kappas.mean() - imagery_kappas.mean()

    degradation = {
        "performance_drop": float(perf_drop),
        "confidence_drop": float(conf_drop),
        "graceful_ratio": float(perf_drop / (conf_drop + 1e-8)),
        "interpretation": (
            "Graceful degradation: confidence drops appropriately with performance"
            if conf_drop > 0 and perf_drop > 0 else
            "WARNING: Overconfident on imagery (confidence didn't drop)"
            if conf_drop <= 0 and perf_drop > 0 else
            "Unexpected: performance didn't drop on imagery"
        ),
    }

    # ---- 5. Per-ROI cross-state comparison (if available) ----
    per_roi_comparison = None
    if per_roi_kappas_perception is not None and per_roi_kappas_imagery is not None:
        from fmri2img.data.nsd_imagery import compute_perception_imagery_kappa_comparison
        per_roi_comparison = compute_perception_imagery_kappa_comparison(
            per_roi_kappas_perception, per_roi_kappas_imagery, roi_names
        )

    report = {
        "n_paired_samples": n_samples,
        "kappa_comparison": kappa_comparison,
        "retrieval": {
            "perception": retrieval_perception,
            "imagery": retrieval_imagery,
        },
        "aur": {
            "perception": aur_perception,
            "imagery": aur_imagery,
        },
        "degradation": degradation,
        "cosine_similarity": {
            "perception_mean": float(perc_cosine.mean()),
            "imagery_mean": float(img_cosine.mean()),
        },
        "per_roi_comparison": per_roi_comparison,
        "key_findings": [],
    }

    # Summarize key findings
    if kappa_comparison["kappa_drop"] > 0.1:
        report["key_findings"].append(
            f"Kappa drops {kappa_comparison['kappa_drop']*100:.1f}% "
            f"from perception to imagery (p={kappa_comparison['paired_ttest']['p']:.2e})"
        )
    if aur_perception["well_calibrated"]:
        report["key_findings"].append(
            f"Model is well-calibrated on perception (AUR={aur_perception['aur_pearson']:.3f})"
        )
    if aur_imagery["aur_pearson"] > aur_perception["aur_pearson"] * 0.5:
        report["key_findings"].append(
            "Calibration transfers to imagery (AUR maintained)"
        )

    logger.info(
        "Cross-state evaluation: %d pairs. "
        "Perception R@1=%.1f%%, Imagery R@1=%.1f%%. "
        "Kappa drop=%.1f%%. AUR_perc=%.3f, AUR_img=%.3f",
        n_samples,
        retrieval_perception["r@1"] * 100,
        retrieval_imagery["r@1"] * 100,
        kappa_comparison["kappa_drop"] * 100,
        aur_perception["aur_pearson"],
        aur_imagery["aur_pearson"],
    )

    return report


def main():
    """CLI entry point for cross-state evaluation."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Cross-state (perception vs imagery) evaluation")
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to trained ROI-DCF model checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cross_state"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Cross-state evaluation for %s", args.subject)
    logger.info("NOTE: Requires NSD-Imagery data to be preprocessed. "
                "Run: python -c 'from fmri2img.data.nsd_imagery import preprocess_imagery_betas; "
                "preprocess_imagery_betas(\"%s\")'", args.subject)

    # Placeholder for actual model loading and inference
    logger.info(
        "To run full evaluation:\n"
        "1. Preprocess imagery data: preprocess_imagery_betas('%s')\n"
        "2. Load model from %s\n"
        "3. Run inference on both perception and imagery features\n"
        "4. Call run_cross_state_evaluation() with results",
        args.subject, args.checkpoint,
    )


if __name__ == "__main__":
    main()
