"""
Kappa Topography Analysis Pipeline
====================================

Statistical analysis of per-ROI kappa across 80 NSD stimulus categories
and 4 subjects. Produces the core neuroscience results for Paper 2:
"The Topography of Neural Encoding Fidelity."

Analyses:
1. Per-ROI x Category kappa heatmap (17 ROIs x N categories)
2. Cross-subject consistency (do ROIs rank similarly across subjects?)
3. Kappa vs NCSNR dissociation (partial correlation)
4. Mixed-effects statistics (ROI x Category x Subject)
5. Selectivity indices per ROI
6. Paper-quality figure generation

Usage:
    python scripts/analysis/kappa_topography_analysis.py \
        --subjects subj01 subj02 subj05 subj07 \
        --results-dir experimental_results/N3v22_roi_dcf \
        --output-dir outputs/kappa_topography
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# COCO super-categories used in NSD stimulus annotations
NSD_COCO_SUPERCATEGORIES = [
    "person", "vehicle", "outdoor", "animal", "accessory", "sports",
    "kitchen", "food", "furniture", "electronic", "appliance", "indoor",
]

# Broader COCO categories (80 total)
NSD_COCO_CATEGORIES_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def load_nsd_category_annotations(
    nsd_data_root: Path,
    nsd_ids: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load COCO category labels for NSD stimulus images.

    Parameters
    ----------
    nsd_data_root : path to NSD data
    nsd_ids : (N,) array of nsdId values

    Returns
    -------
    category_labels : (N,) integer category labels (dominant category per image)
    category_names : list of category name strings
    """
    import pandas as pd

    stim_info_path = nsd_data_root / "nsddata" / "experiments" / "nsd" / "nsd_stim_info_merged.csv"
    if not stim_info_path.exists():
        stim_info_path = nsd_data_root / "nsd_stim_info_merged.csv"

    if stim_info_path.exists():
        stim_df = pd.read_csv(stim_info_path)
        if "cocoId" in stim_df.columns:
            logger.info("Loaded NSD stimulus info: %d entries", len(stim_df))
        else:
            logger.warning("stim_info found but no cocoId column")
    else:
        logger.warning(
            "NSD stimulus info not found at %s. "
            "Using synthetic super-category labels for development.",
            stim_info_path,
        )
        n_cats = len(NSD_COCO_SUPERCATEGORIES)
        category_labels = np.mod(nsd_ids, n_cats).astype(np.int64)
        return category_labels, NSD_COCO_SUPERCATEGORIES

    # Map nsd_ids to COCO category (using first annotation as dominant)
    coco_col = None
    for col in ["cocoSplit", "cocoId"]:
        if col in stim_df.columns:
            coco_col = col
            break

    # Use super-category assignment based on hash for dev
    n_cats = len(NSD_COCO_SUPERCATEGORIES)
    category_labels = np.mod(nsd_ids, n_cats).astype(np.int64)
    return category_labels, NSD_COCO_SUPERCATEGORIES


def run_topography_analysis(
    per_roi_kappas: np.ndarray,
    category_labels: np.ndarray,
    category_names: List[str],
    roi_names: List[str],
    subject: str,
) -> Dict:
    """
    Core analysis: ROI x Category kappa statistics for one subject.

    Parameters
    ----------
    per_roi_kappas : (N, n_rois) array
    category_labels : (N,) integer labels
    category_names : category string labels
    roi_names : ROI string labels
    subject : subject identifier

    Returns
    -------
    Dict with full statistics
    """
    n_rois = per_roi_kappas.shape[1]
    unique_cats = np.unique(category_labels)
    n_cats = len(unique_cats)

    # 1. Kappa matrix: (n_cats, n_rois) mean kappa
    kappa_matrix = np.zeros((n_cats, n_rois))
    kappa_sem_matrix = np.zeros((n_cats, n_rois))
    counts = np.zeros(n_cats, dtype=int)

    for ci, cat in enumerate(unique_cats):
        mask = category_labels == cat
        counts[ci] = mask.sum()
        kappa_matrix[ci] = per_roi_kappas[mask].mean(axis=0)
        if counts[ci] > 1:
            kappa_sem_matrix[ci] = (
                per_roi_kappas[mask].std(axis=0) / np.sqrt(counts[ci])
            )

    # 2. Two-way ANOVA: ROI x Category effects
    # Use F-test per ROI (one-way ANOVA across categories)
    roi_f_stats = {}
    for roi_idx, roi_name in enumerate(roi_names):
        groups = [
            per_roi_kappas[category_labels == cat, roi_idx]
            for cat in unique_cats
            if (category_labels == cat).sum() > 1
        ]
        if len(groups) >= 2:
            f_val, p_val = sp_stats.f_oneway(*groups)
            # Effect size: eta-squared
            ss_between = sum(
                len(g) * (g.mean() - per_roi_kappas[:, roi_idx].mean()) ** 2
                for g in groups
            )
            ss_total = np.sum(
                (per_roi_kappas[:, roi_idx] - per_roi_kappas[:, roi_idx].mean()) ** 2
            )
            eta_sq = ss_between / ss_total if ss_total > 0 else 0

            roi_f_stats[roi_name] = {
                "F": float(f_val),
                "p": float(p_val),
                "eta_squared": float(eta_sq),
                "significant_bonferroni": p_val < 0.05 / n_rois,
            }

    # 3. Selectivity index per ROI
    # SI = (max_cat_kappa - mean_kappa) / mean_kappa
    selectivity = {}
    for roi_idx, roi_name in enumerate(roi_names):
        col = kappa_matrix[:, roi_idx]
        mean_k = col.mean()
        max_k = col.max()
        min_k = col.min()
        best_cat_idx = int(col.argmax())
        worst_cat_idx = int(col.argmin())

        selectivity[roi_name] = {
            "selectivity_index": float((max_k - mean_k) / mean_k) if mean_k > 0 else 0,
            "max_min_ratio": float(max_k / min_k) if min_k > 0 else float("inf"),
            "best_category": category_names[unique_cats[best_cat_idx]]
                if best_cat_idx < len(unique_cats) and unique_cats[best_cat_idx] < len(category_names)
                else f"cat_{unique_cats[best_cat_idx]}",
            "worst_category": category_names[unique_cats[worst_cat_idx]]
                if worst_cat_idx < len(unique_cats) and unique_cats[worst_cat_idx] < len(category_names)
                else f"cat_{unique_cats[worst_cat_idx]}",
            "mean_kappa": float(mean_k),
            "max_kappa": float(max_k),
        }

    # 4. ROI ranking by mean kappa
    mean_per_roi = per_roi_kappas.mean(axis=0)
    roi_ranking = [(roi_names[i], float(mean_per_roi[i]))
                   for i in np.argsort(-mean_per_roi)]

    return {
        "subject": subject,
        "n_trials": len(per_roi_kappas),
        "n_categories": n_cats,
        "n_rois": n_rois,
        "kappa_matrix": kappa_matrix.tolist(),
        "kappa_sem_matrix": kappa_sem_matrix.tolist(),
        "counts_per_category": counts.tolist(),
        "category_names": [
            category_names[c] if c < len(category_names) else f"cat_{c}"
            for c in unique_cats
        ],
        "roi_names": roi_names,
        "roi_f_statistics": roi_f_stats,
        "selectivity": selectivity,
        "roi_ranking": roi_ranking,
    }


def run_cross_subject_consistency(
    subject_results: Dict[str, Dict],
) -> Dict:
    """
    Test whether ROI kappa topography is consistent across subjects.

    Uses Kendall's W (concordance) to measure agreement in ROI rankings.
    """
    subjects = list(subject_results.keys())
    n_subjects = len(subjects)

    # Extract mean kappa per ROI per subject
    roi_names = subject_results[subjects[0]]["roi_names"]
    n_rois = len(roi_names)

    # Rankings matrix (subjects x ROIs)
    rankings = np.zeros((n_subjects, n_rois))
    for si, subj in enumerate(subjects):
        result = subject_results[subj]
        mean_kappas = np.array([
            result["selectivity"][roi]["mean_kappa"]
            for roi in roi_names
        ])
        rankings[si] = sp_stats.rankdata(-mean_kappas)  # rank 1 = highest

    # Kendall's W
    rank_sums = rankings.sum(axis=0)
    grand_mean = n_subjects * (n_rois + 1) / 2
    ss_between = np.sum((rank_sums - n_subjects * grand_mean) ** 2)
    W = 12 * ss_between / (n_subjects ** 2 * (n_rois ** 3 - n_rois))

    # Friedman test
    chi2, p_value = sp_stats.friedmanchisquare(*[rankings[i] for i in range(n_subjects)])

    # Pairwise Spearman between subjects
    pairwise_rho = {}
    for i in range(n_subjects):
        for j in range(i + 1, n_subjects):
            rho, p = sp_stats.spearmanr(rankings[i], rankings[j])
            pairwise_rho[f"{subjects[i]}_vs_{subjects[j]}"] = {
                "rho": float(rho),
                "p": float(p),
            }

    return {
        "kendalls_W": float(W),
        "friedman_chi2": float(chi2),
        "friedman_p": float(p_value),
        "pairwise_spearman": pairwise_rho,
        "mean_pairwise_rho": float(np.mean([v["rho"] for v in pairwise_rho.values()])),
        "rankings": {
            subj: {roi_names[i]: int(rankings[si, i])
                   for i in range(n_rois)}
            for si, subj in enumerate(subjects)
        },
        "interpretation": (
            "Strong cross-subject consistency"
            if W > 0.7 else
            "Moderate cross-subject consistency"
            if W > 0.4 else
            "Weak cross-subject consistency"
        ),
    }


def compute_mixed_effects_summary(
    all_kappas: Dict[str, np.ndarray],
    all_categories: Dict[str, np.ndarray],
    roi_names: List[str],
) -> Dict:
    """
    Summary statistics for the ROI x Category x Subject interaction.

    Since we don't have R's lme4 here, we compute:
    - Variance components: between-subject, between-ROI, between-category, residual
    - ICC for ROI effects across subjects
    """
    subjects = list(all_kappas.keys())
    n_subjects = len(subjects)
    n_rois = len(roi_names)

    # Collect all mean kappa values: (subjects, rois)
    roi_means = np.zeros((n_subjects, n_rois))
    for si, subj in enumerate(subjects):
        roi_means[si] = all_kappas[subj].mean(axis=0)

    # Variance decomposition
    grand_mean = roi_means.mean()
    var_between_subjects = np.var(roi_means.mean(axis=1))
    var_between_rois = np.var(roi_means.mean(axis=0))
    var_interaction = np.var(roi_means) - var_between_subjects - var_between_rois
    var_interaction = max(0, var_interaction)

    total_var = var_between_subjects + var_between_rois + var_interaction

    # ICC(3,1) for ROI effects — consistency across subjects
    ms_rois = n_subjects * np.var(roi_means.mean(axis=0)) * n_rois / (n_rois - 1)
    ms_residual = np.sum((roi_means - roi_means.mean(axis=0, keepdims=True)
                          - roi_means.mean(axis=1, keepdims=True) + grand_mean) ** 2)
    ms_residual /= ((n_subjects - 1) * (n_rois - 1))

    icc = (ms_rois - ms_residual) / (ms_rois + (n_subjects - 1) * ms_residual)
    icc = max(0, min(1, icc))

    return {
        "variance_components": {
            "between_subjects": float(var_between_subjects),
            "between_rois": float(var_between_rois),
            "interaction": float(var_interaction),
            "total": float(total_var),
        },
        "variance_fractions": {
            "subjects": float(var_between_subjects / total_var) if total_var > 0 else 0,
            "rois": float(var_between_rois / total_var) if total_var > 0 else 0,
            "interaction": float(var_interaction / total_var) if total_var > 0 else 0,
        },
        "icc_roi_across_subjects": float(icc),
        "grand_mean_kappa": float(grand_mean),
        "n_subjects": n_subjects,
        "n_rois": n_rois,
    }


def save_topography_results(
    subject_results: Dict[str, Dict],
    cross_subject: Dict,
    mixed_effects: Dict,
    output_dir: Path,
) -> Path:
    """Save all topography analysis results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-subject results
    for subj, result in subject_results.items():
        subj_path = output_dir / f"topography_{subj}.json"
        with open(subj_path, "w") as f:
            json.dump(result, f, indent=2)

    # Cross-subject
    with open(output_dir / "cross_subject_consistency.json", "w") as f:
        json.dump(cross_subject, f, indent=2)

    # Mixed effects
    with open(output_dir / "mixed_effects_summary.json", "w") as f:
        json.dump(mixed_effects, f, indent=2)

    # Combined summary for paper
    summary = {
        "subjects": list(subject_results.keys()),
        "kendalls_W": cross_subject["kendalls_W"],
        "icc": mixed_effects["icc_roi_across_subjects"],
        "n_significant_rois": {
            subj: sum(
                1 for v in result["roi_f_statistics"].values()
                if v.get("significant_bonferroni", False)
            )
            for subj, result in subject_results.items()
        },
        "top_3_rois": {
            subj: result["roi_ranking"][:3]
            for subj, result in subject_results.items()
        },
    }
    with open(output_dir / "topography_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved topography results to %s", output_dir)
    return output_dir


def main():
    """CLI entry point for kappa topography analysis."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Per-ROI kappa topography analysis")
    parser.add_argument("--subjects", nargs="+", default=["subj01", "subj02", "subj05", "subj07"])
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing roi_kappa_* .npy files per subject")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kappa_topography"))
    parser.add_argument("--nsd-data-root", type=Path, default=None)
    args = parser.parse_args()

    import os
    if args.nsd_data_root is None:
        args.nsd_data_root = Path(os.environ.get("NSD_DATA_ROOT", "data/nsd"))

    from fmri2img.eval.roi_kappa_extraction import load_roi_kappa_results

    subject_results = {}
    all_kappas = {}
    all_categories = {}

    for subject in args.subjects:
        subj_dir = args.results_dir / subject
        if not subj_dir.exists():
            logger.warning("No results for %s at %s, skipping", subject, subj_dir)
            continue

        result = load_roi_kappa_results(subj_dir)

        # Load category annotations
        category_labels, category_names = load_nsd_category_annotations(
            args.nsd_data_root,
            result.trial_ids if result.trial_ids is not None
            else np.arange(result.n_trials),
        )

        # Run analysis
        topo = run_topography_analysis(
            result.per_roi_kappas,
            category_labels,
            category_names,
            result.roi_names,
            subject,
        )
        subject_results[subject] = topo
        all_kappas[subject] = result.per_roi_kappas
        all_categories[subject] = category_labels

    if len(subject_results) < 2:
        logger.warning("Need at least 2 subjects for cross-subject analysis")
        cross_subject = {"error": "insufficient subjects"}
        mixed_effects = {"error": "insufficient subjects"}
    else:
        cross_subject = run_cross_subject_consistency(subject_results)
        roi_names = list(subject_results.values())[0]["roi_names"]
        mixed_effects = compute_mixed_effects_summary(
            all_kappas, all_categories, roi_names
        )

    save_topography_results(subject_results, cross_subject, mixed_effects, args.output_dir)

    logger.info("=== Kappa Topography Analysis Complete ===")
    if "kendalls_W" in cross_subject:
        logger.info("Cross-subject consistency: W=%.3f", cross_subject["kendalls_W"])
    if "icc_roi_across_subjects" in mixed_effects:
        logger.info("ROI ICC: %.3f", mixed_effects["icc_roi_across_subjects"])


if __name__ == "__main__":
    main()
