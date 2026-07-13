"""
Multi-Subject Consistency Analysis — The Definitive Paper Evidence.

Runs per-ROI kappa extraction + semantic alignment on subj02, subj05, subj07
then computes cross-subject consistency statistics:
1. Kendall's W for ROI kappa ranking
2. ICC for per-ROI kappa means
3. OPA outdoor specificity replication
4. Cross-subject kappa hierarchy correlation

This proves the findings are population-level, not single-subject artifacts.
"""
import sys
sys.path.insert(0, "/home/jovyan/work/FMRI2images/src")

import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("NSD_DATA_ROOT", "/home/jovyan/work/data/nsd")
os.environ.setdefault("CACHE_ROOT", "/home/jovyan/work/FMRI2images/cache")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_ROOT = Path(os.environ["CACHE_ROOT"])
SUBJECTS = ["subj01", "subj02", "subj05", "subj07"]
OUTPUT_DIR = Path("experimental_results/UDND_analysis/multi_subject")


_CACHED_MODEL = None
_CACHED_ROI_NAMES = None
_CACHED_SUBJECTS_LIST = None

def load_model_for_subject(subject):
    """Load V66a multi-subject model (shared checkpoint, subject-specific ROI projections)."""
    global _CACHED_MODEL, _CACHED_ROI_NAMES, _CACHED_SUBJECTS_LIST
    
    from fmri2img.models.unified_model import create_model
    
    # Single checkpoint for all subjects
    checkpoint = Path("experimental_results/V66a_roi_pretrain/subj01/checkpoint_best.pt")
    if not checkpoint.exists():
        logger.warning("No checkpoint at %s", checkpoint)
        return None, None, -1
    
    if _CACHED_MODEL is not None:
        subj_idx = _CACHED_SUBJECTS_LIST.index(subject) if subject in _CACHED_SUBJECTS_LIST else -1
        return _CACHED_MODEL, _CACHED_ROI_NAMES, subj_idx
    
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    model_config = ckpt.get("model_config", config.get("model", {}))
    
    # Use subj01 dims for model creation (multi-subject model accepts all)
    subject_roi_dims = model_config.get("encoder", {}).get("subject_roi_dims", {}).get("subj01", {})
    if not subject_roi_dims:
        subject_roi_dims = model_config.get("encoder", {}).get("roi_dims", {})
    
    roi_names = list(subject_roi_dims.keys())
    
    state_dict = ckpt["model_state_dict"]
    idx_pattern = re.compile(r"encoder\._idx_(\w+)_(\d+)")
    subject_roi_indices = {}
    for key, tensor in state_dict.items():
        m = idx_pattern.match(key)
        if m:
            subj_id = m.group(1)
            roi_idx = int(m.group(2))
            if subj_id not in subject_roi_indices:
                subject_roi_indices[subj_id] = {}
            if roi_idx < len(roi_names):
                subject_roi_indices[subj_id][roi_names[roi_idx]] = tensor.numpy()
    
    # Use subj01's indices for model creation
    roi_indices = subject_roi_indices.get("subj01", {})
    encoder_cfg = model_config.get("encoder", {})
    total_voxels = sum(subject_roi_dims.values())
    if "input_dim" not in encoder_cfg:
        encoder_cfg["input_dim"] = total_voxels
        model_config["encoder"] = encoder_cfg
    
    model = create_model(model_config, roi_indices=roi_indices)
    model.load_state_dict(state_dict, strict=False)
    for key in state_dict:
        m = idx_pattern.match(key)
        if m:
            parts = key.split(".", 1)
            if len(parts) == 2:
                model.encoder.register_buffer(parts[1], state_dict[key])
    
    model = model.to(DEVICE).eval()
    
    # Get subjects list from config
    subjects_list = config.get("data", {}).get("subjects", ["subj01", "subj02", "subj05", "subj07"])
    subj_idx = subjects_list.index(subject) if subject in subjects_list else 0
    
    _CACHED_MODEL = model
    _CACHED_ROI_NAMES = roi_names
    _CACHED_SUBJECTS_LIST = subjects_list
    
    logger.info("Model loaded (multi-subject, %d subjects)", len(subjects_list))
    return model, roi_names, subj_idx


@torch.no_grad()
def extract_per_roi_data(model, features, subject_idx, batch_size=128):
    """Extract per-ROI kappa and mu for a subject."""
    n_trials = len(features)
    all_roi_kappas = []
    all_roi_mus = []
    all_cls_kappas = []
    all_alphas = []
    
    for start in range(0, n_trials, batch_size):
        end = min(start + batch_size, n_trials)
        batch = torch.from_numpy(features[start:end]).float().to(DEVICE)
        subject_ids = torch.full((len(batch),), subject_idx, dtype=torch.long, device=DEVICE)
        
        encoder_out = model.encoder(batch, subject_ids, return_roi_tokens=True)
        B, n_rois, d = encoder_out.roi_tokens.shape
        
        cls_mu, cls_kappa = model.decoder(encoder_out.cls_out)
        roi_tokens_flat = encoder_out.roi_tokens.reshape(B * n_rois, d)
        roi_mu_flat, roi_kappa_flat = model.decoder(roi_tokens_flat)
        
        all_roi_kappas.append(roi_kappa_flat.reshape(B, n_rois).cpu().numpy())
        all_roi_mus.append(roi_mu_flat.reshape(B, n_rois, -1).cpu().numpy())
        all_cls_kappas.append(cls_kappa.cpu().numpy())
        all_alphas.append(encoder_out.cls_to_roi_alpha.cpu().numpy())
    
    return {
        "roi_kappas": np.concatenate(all_roi_kappas, axis=0),
        "roi_mus": np.concatenate(all_roi_mus, axis=0),
        "cls_kappa": np.concatenate(all_cls_kappas, axis=0).squeeze(-1),
        "alphas": np.concatenate(all_alphas, axis=0),
    }


def compute_category_centroids():
    """Compute mean CLIP embedding per category."""
    clip_path = CACHE_ROOT.parent / "outputs" / "clip_cache" / "clip.parquet"
    clip_df = pd.read_parquet(clip_path)
    
    if "clip_embedding" in clip_df.columns:
        embeddings = np.stack(clip_df["clip_embedding"].values)
    elif "embedding" in clip_df.columns:
        embeddings = np.stack(clip_df["embedding"].values)
    else:
        emb_cols = sorted([c for c in clip_df.columns if c.startswith("emb_")])
        embeddings = clip_df[emb_cols].values
    
    nsd_ids = clip_df["nsdId"].values
    cat_df = pd.read_parquet(CACHE_ROOT / "nsd_category_labels.parquet")
    nsd_to_cat = dict(zip(cat_df["nsdId"].values, cat_df["supercategory"].values))
    categories = np.array([nsd_to_cat.get(int(nid), "unknown") for nid in nsd_ids])
    
    centroids = {}
    for cat in sorted(set(categories) - {"unknown"}):
        mask = categories == cat
        centroid = embeddings[mask].mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids[cat] = centroid
    
    return centroids


def run_single_subject_analysis(subject, model, roi_names, centroids, subject_idx):
    """Complete analysis for one subject."""
    logger.info("\n" + "=" * 70)
    logger.info("SUBJECT: %s (idx=%d)", subject, subject_idx)
    logger.info("=" * 70)
    
    # Load data
    features_path = CACHE_ROOT / "preextracted" / f"subject={subject}" / "fmri_features.npy"
    if not features_path.exists():
        logger.warning("No features for %s", subject)
        return None
    
    features = np.load(features_path)
    meta_df = pd.read_parquet(CACHE_ROOT / "preextracted" / f"subject={subject}" / "trial_meta.parquet")
    
    # Extract
    t0 = time.time()
    data = extract_per_roi_data(model, features, subject_idx)
    logger.info("Extraction: %.1fs for %d trials", time.time() - t0, len(features))
    
    # Category mapping
    cat_df = pd.read_parquet(CACHE_ROOT / "nsd_category_labels.parquet")
    nsd_to_cat = dict(zip(cat_df["nsdId"].values, cat_df["supercategory"].values))
    nsd_ids = meta_df["nsdId"].values
    trial_cats = np.array([nsd_to_cat.get(int(nid), "unknown") for nid in nsd_ids])
    valid = trial_cats != "unknown"
    
    roi_kappas = data["roi_kappas"][valid]
    roi_mus = data["roi_mus"][valid]
    alphas = data["alphas"][valid]
    cls_kappa = data["cls_kappa"][valid]
    trial_cats_valid = trial_cats[valid]
    
    # === FINDING 1: Per-ROI kappa hierarchy ===
    roi_kappa_means = roi_kappas.mean(axis=0)
    roi_kappa_stds = roi_kappas.std(axis=0)
    roi_kappa_ranking = np.argsort(-roi_kappa_means)
    
    logger.info("\n--- Kappa Hierarchy ---")
    for rank, idx in enumerate(roi_kappa_ranking):
        logger.info("  %2d. %-20s kappa=%.3f +/- %.3f", rank+1, roi_names[idx], 
                   roi_kappa_means[idx], roi_kappa_stds[idx])
    
    # === FINDING 2: OPA outdoor specificity ===
    unique_cats = sorted(set(trial_cats_valid) - {"unknown"})
    centroid_matrix = np.stack([centroids[c] for c in unique_cats])
    
    # Normalize mus
    roi_mus_norm = roi_mus / (np.linalg.norm(roi_mus, axis=-1, keepdims=True) + 1e-8)
    
    specialization_results = {}
    test_pairs = [
        ("EBA", "person"), ("FFA1", "person"), ("FFA2", "person"), ("OFA", "person"),
        ("PPA", "outdoor"), ("OPA", "outdoor"),
    ]
    
    for roi, expected_cat in test_pairs:
        if roi not in roi_names or expected_cat not in unique_cats:
            continue
        j = roi_names.index(roi)
        cat_idx = unique_cats.index(expected_cat)
        
        mask = trial_cats_valid == expected_cat
        other_roi_idx = [k for k in range(len(roi_names)) if k != j]
        
        # Alignment of this ROI to expected centroid vs other ROIs
        alignment = np.dot(roi_mus_norm[:, :, :], centroid_matrix[cat_idx])  # (N, n_rois)
        
        roi_align = alignment[mask, j]
        other_align = alignment[mask][:, other_roi_idx].mean(axis=1)
        
        t_stat, p_val = stats.ttest_rel(roi_align, other_align)
        d = (roi_align.mean() - other_align.mean()) / roi_align.std() if roi_align.std() > 0 else 0
        
        specialization_results[f"{roi}_{expected_cat}"] = {
            "d": float(d), "t": float(t_stat), "p": float(p_val),
            "n": int(mask.sum()),
            "direction": "positive" if d > 0 else "negative",
        }
        logger.info("  %s -> %s: d=%.4f, p=%.2e", roi, expected_cat, d, p_val)
    
    # === FINDING 3: Kappa varies by category ===
    kappa_by_cat = {}
    for cat in unique_cats:
        mask = trial_cats_valid == cat
        kappa_by_cat[cat] = float(cls_kappa[mask].mean())
    
    f_stat, p_val = stats.f_oneway(*[cls_kappa[trial_cats_valid == c] for c in unique_cats])
    
    return {
        "subject": subject,
        "n_trials": int(valid.sum()),
        "n_rois": len(roi_names),
        "roi_names": roi_names,
        "kappa_hierarchy": {roi_names[i]: float(roi_kappa_means[i]) for i in roi_kappa_ranking},
        "kappa_ranking": [roi_names[i] for i in roi_kappa_ranking],
        "kappa_means": {roi: float(m) for roi, m in zip(roi_names, roi_kappa_means)},
        "specialization": specialization_results,
        "kappa_by_category": kappa_by_cat,
        "kappa_anova": {"f": float(f_stat), "p": float(p_val)},
        "cls_kappa_mean": float(cls_kappa.mean()),
        "cls_kappa_std": float(cls_kappa.std()),
    }


def compute_cross_subject_consistency(subject_results):
    """Compute Kendall's W and ICC for cross-subject consistency."""
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-SUBJECT CONSISTENCY ANALYSIS")
    logger.info("=" * 70)
    
    # Get common ROI set
    common_rois = None
    for result in subject_results.values():
        rois = set(result["roi_names"])
        if common_rois is None:
            common_rois = rois
        else:
            common_rois = common_rois & rois
    
    common_rois = sorted(common_rois)
    n_rois = len(common_rois)
    n_subjects = len(subject_results)
    
    logger.info("Common ROIs: %d, Subjects: %d", n_rois, n_subjects)
    
    # Build matrix: (subjects x ROIs) of mean kappas
    kappa_matrix = np.zeros((n_subjects, n_rois))
    for i, (subj, result) in enumerate(subject_results.items()):
        for j, roi in enumerate(common_rois):
            kappa_matrix[i, j] = result["kappa_means"].get(roi, 0)
    
    # === Kendall's W (concordance of rankings) ===
    # Rank each subject's ROIs
    rank_matrix = np.zeros_like(kappa_matrix)
    for i in range(n_subjects):
        rank_matrix[i] = stats.rankdata(kappa_matrix[i])
    
    # Kendall's W = 12 * S / (k^2 * (n^3 - n))
    rank_sums = rank_matrix.sum(axis=0)
    grand_mean = rank_sums.mean()
    S = np.sum((rank_sums - grand_mean) ** 2)
    W = 12 * S / (n_subjects ** 2 * (n_rois ** 3 - n_rois))
    
    # Chi-square test for W
    chi2 = n_subjects * (n_rois - 1) * W
    df = n_rois - 1
    p_w = 1 - stats.chi2.cdf(chi2, df)
    
    logger.info("\n--- Kendall's W (ROI kappa ranking) ---")
    logger.info("  W = %.4f", W)
    logger.info("  Chi2(%d) = %.2f, p = %.2e", df, chi2, p_w)
    logger.info("  Interpretation: %s", 
               "strong" if W > 0.7 else "moderate" if W > 0.5 else "weak" if W > 0.3 else "poor")
    
    # === Pairwise Spearman correlations ===
    subjects_list = list(subject_results.keys())
    pairwise_rho = []
    for i in range(n_subjects):
        for j in range(i+1, n_subjects):
            rho, p = stats.spearmanr(kappa_matrix[i], kappa_matrix[j])
            pairwise_rho.append(rho)
            logger.info("  %s vs %s: rho=%.4f (p=%.4f)", subjects_list[i], subjects_list[j], rho, p)
    
    mean_rho = np.mean(pairwise_rho)
    logger.info("  Mean pairwise Spearman rho: %.4f", mean_rho)
    
    # === ICC (Intraclass Correlation) ===
    # Two-way random, absolute agreement (ICC(2,1))
    grand_mean_kappa = kappa_matrix.mean()
    ss_between = n_subjects * np.sum((kappa_matrix.mean(axis=0) - grand_mean_kappa) ** 2)
    ss_within = np.sum((kappa_matrix - kappa_matrix.mean(axis=0, keepdims=True)) ** 2)
    ss_subjects = n_rois * np.sum((kappa_matrix.mean(axis=1) - grand_mean_kappa) ** 2)
    ss_error = ss_within - ss_subjects
    
    ms_between = ss_between / (n_rois - 1)
    ms_subjects = ss_subjects / (n_subjects - 1)
    ms_error = ss_error / ((n_rois - 1) * (n_subjects - 1))
    
    icc = (ms_between - ms_error) / (ms_between + (n_subjects - 1) * ms_error)
    
    logger.info("\n--- ICC (2,1) ---")
    logger.info("  ICC = %.4f", icc)
    logger.info("  Interpretation: %s",
               "excellent" if icc > 0.9 else "good" if icc > 0.75 else "moderate" if icc > 0.5 else "poor")
    
    # === OPA Outdoor Specificity Replication ===
    logger.info("\n--- OPA Outdoor Specificity (replication) ---")
    opa_results = {}
    for subj, result in subject_results.items():
        opa_data = result["specialization"].get("OPA_outdoor", {})
        if opa_data:
            logger.info("  %s: d=%.4f, p=%.2e (%s)", subj, opa_data["d"], opa_data["p"], opa_data["direction"])
            opa_results[subj] = opa_data
    
    # Meta-analysis: Fisher's method for combining p-values
    opa_pvals = [r["p"] for r in opa_results.values() if r["p"] > 0]
    if opa_pvals:
        fisher_stat = -2 * sum(np.log(p) for p in opa_pvals)
        fisher_df = 2 * len(opa_pvals)
        fisher_p = 1 - stats.chi2.cdf(fisher_stat, fisher_df)
        logger.info("  Fisher's combined p = %.2e (Chi2(%d) = %.2f)", fisher_p, fisher_df, fisher_stat)
        
        mean_d = np.mean([r["d"] for r in opa_results.values()])
        logger.info("  Mean Cohen's d across subjects: %.4f", mean_d)
    
    # === Consensus kappa hierarchy ===
    mean_kappas_across_subjects = kappa_matrix.mean(axis=0)
    consensus_ranking = [common_rois[i] for i in np.argsort(-mean_kappas_across_subjects)]
    
    logger.info("\n--- Consensus Kappa Hierarchy (mean across subjects) ---")
    for rank, roi in enumerate(consensus_ranking):
        idx = common_rois.index(roi)
        mean_k = mean_kappas_across_subjects[idx]
        std_k = kappa_matrix[:, idx].std()
        logger.info("  %2d. %-20s kappa=%.3f +/- %.3f", rank+1, roi, mean_k, std_k)
    
    return {
        "kendalls_w": float(W),
        "kendalls_w_chi2": float(chi2),
        "kendalls_w_p": float(p_w),
        "icc": float(icc),
        "mean_pairwise_spearman": float(mean_rho),
        "pairwise_rho": pairwise_rho,
        "consensus_ranking": consensus_ranking,
        "opa_outdoor_meta": {
            "fisher_p": float(fisher_p) if opa_pvals else None,
            "mean_d": float(mean_d) if opa_pvals else None,
            "per_subject": opa_results,
        },
        "common_rois": common_rois,
        "n_subjects": n_subjects,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    centroids = compute_category_centroids()
    
    subject_results = {}
    for subject in SUBJECTS:
        model, roi_names, subj_idx = load_model_for_subject(subject)
        if model is None:
            logger.warning("Skipping %s — no model", subject)
            continue
        if subj_idx < 0:
            logger.warning("Skipping %s — not in model's subject list", subject)
            continue
        
        result = run_single_subject_analysis(subject, model, roi_names, centroids, subj_idx)
        if result:
            subject_results[subject] = result
    
    logger.info("\n\nAnalyzed %d subjects: %s", len(subject_results), list(subject_results.keys()))
    
    if len(subject_results) >= 2:
        consistency = compute_cross_subject_consistency(subject_results)
    else:
        consistency = {"error": "Not enough subjects for consistency analysis"}
    
    # Save final results
    all_results = {
        "subjects": list(subject_results.keys()),
        "per_subject": subject_results,
        "cross_subject_consistency": consistency,
    }
    
    output_file = OUTPUT_DIR / "multi_subject_analysis.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    
    logger.info("\n\nFinal results saved to: %s", output_file)
    logger.info("Done!")


if __name__ == "__main__":
    main()
