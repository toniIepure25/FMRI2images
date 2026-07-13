"""
Extract per-ROI kappas from the trained vmf_dcf model (N3v9)
and run COCO category specialization analysis.

The vmf_dcf model has true per-ROI expert heads, unlike V66a
which used a global vMF head applied to each ROI token.
"""
import sys
sys.path.insert(0, "/home/jovyan/work/FMRI2images/src")

import json
import logging
import os
import re
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
SUBJECT = "subj01"
REPO_ROOT = Path("/home/jovyan/work/FMRI2images")
CACHE_ROOT = Path(os.environ["CACHE_ROOT"])
OUTPUT_DIR = REPO_ROOT / "experimental_results" / "calibrated_uncertainty" / "vmf_dcf_analysis"


def load_vmf_dcf_model():
    """Load the trained vmf_dcf model."""
    from fmri2img.models.unified_model import create_model
    
    ckpt_path = REPO_ROOT / "experimental_results/N3v9_roi_dcf/subj01/checkpoint_best.pt"
    if not ckpt_path.exists():
        ckpt_path = REPO_ROOT / "experimental_results/N3v9_roi_dcf/subj01/checkpoints/best.pt"
    
    logger.info("Loading vmf_dcf from %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    config = ckpt.get("config", {})
    model_config = ckpt.get("model_config", config.get("model", {}))
    state_dict = ckpt["model_state_dict"]
    
    # Get ROI information
    roi_dims = model_config.get("encoder", {}).get("subject_roi_dims", {}).get(SUBJECT, {})
    if not roi_dims:
        roi_dims = model_config.get("encoder", {}).get("roi_dims", {})
    
    roi_names = list(roi_dims.keys())
    logger.info("ROI names (%d): %s", len(roi_names), roi_names)
    
    # Build ROI indices from state dict
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
    
    roi_indices = subject_roi_indices.get(SUBJECT, {})
    
    encoder_cfg = model_config.get("encoder", {})
    if "input_dim" not in encoder_cfg:
        encoder_cfg["input_dim"] = sum(roi_dims.values())
        model_config["encoder"] = encoder_cfg
    
    model = create_model(model_config, roi_indices=roi_indices)
    model.load_state_dict(state_dict, strict=False)
    
    # Register buffers
    for key in state_dict:
        m = idx_pattern.match(key)
        if m:
            parts = key.split(".", 1)
            if len(parts) == 2:
                try:
                    model.encoder.register_buffer(parts[1], state_dict[key])
                except Exception:
                    pass
    
    model = model.to(DEVICE).eval()
    
    subjects_list = config.get("data", {}).get("subjects", [SUBJECT])
    subject_idx = subjects_list.index(SUBJECT) if SUBJECT in subjects_list else 0
    
    logger.info("Model type: %s, decoder: %s", model_config.get("type"), type(model.decoder).__name__)
    
    # Check if decoder has per-ROI output
    has_per_roi = hasattr(model.decoder, 'return_per_roi') or \
                  model_config.get("decoder", {}).get("dcf", {}).get("return_per_roi", False)
    logger.info("Per-ROI output: %s", has_per_roi)
    
    return model, roi_names, subject_idx, config


def load_shared1000_data_zscored(exp_dir):
    """Load SHARED1000 data with z-scoring."""
    index_path = REPO_ROOT / f"data/indices/nsd_index/subject={SUBJECT}/index.parquet"
    index_df = pd.read_parquet(index_path)
    
    s1000_df = index_df[index_df["shared1000"] == True].copy()
    s1000_df = s1000_df.sort_values(["nsdId", "session"]).reset_index(drop=False)
    s1000_df.rename(columns={"index": "original_index"}, inplace=True)
    
    features = np.load(CACHE_ROOT / f"preextracted/subject={SUBJECT}/fmri_features.npy", mmap_mode="r")
    trial_indices = s1000_df["original_index"].values
    s1000_features = features[trial_indices].copy()
    
    sessions = s1000_df["session"].values
    nsd_ids = s1000_df["nsdId"].values
    
    # Apply z-scoring
    zscore_dir = REPO_ROOT / "experimental_results" / exp_dir / "subj01" / "zscore_stats"
    if zscore_dir.exists():
        for prefix in ["session", f"{SUBJECT}_session", "subj01_session"]:
            test_path = zscore_dir / f"{prefix}_1_mean.npy"
            if test_path.exists():
                for sess in np.unique(sessions):
                    mean_path = zscore_dir / f"{prefix}_{sess}_mean.npy"
                    std_path = zscore_dir / f"{prefix}_{sess}_std.npy"
                    if mean_path.exists() and std_path.exists():
                        mean = np.load(mean_path)
                        std = np.load(std_path)
                        std = np.where(std < 1e-6, 1.0, std)
                        mask = sessions == sess
                        s1000_features[mask] = (s1000_features[mask] - mean) / std
                logger.info("Applied z-scoring with prefix '%s'", prefix)
                break
    
    return s1000_features, nsd_ids, np.sort(np.unique(nsd_ids))


@torch.no_grad()
def extract_per_roi_kappas(model, features, roi_names, subject_idx=0, batch_size=64):
    """Extract per-ROI kappa values from vmf_dcf model using _return_per_roi flag."""
    model.eval()
    
    # Enable per-ROI output mode
    model._return_per_roi = True
    
    n_trials = len(features)
    n_rois = len(roi_names)
    
    all_global_kappas = []
    all_per_roi_kappas = []
    all_global_preds = []
    all_deltas = []
    all_alphas = []
    
    for start in range(0, n_trials, batch_size):
        end = min(start + batch_size, n_trials)
        batch = torch.from_numpy(features[start:end]).float().to(DEVICE)
        
        kwargs = {"subject_ids": torch.full((len(batch),), subject_idx, dtype=torch.long, device=DEVICE)}
        
        out = model(batch, **kwargs)
        
        if isinstance(out, tuple):
            pred, aux = out
        else:
            pred = out
            aux = None
        
        all_global_preds.append(pred.cpu().numpy())
        
        if aux is not None:
            k = aux.squeeze(-1)
            all_global_kappas.append(k.cpu().numpy())
        
        # Access per-ROI extras stored by the vmf_dcf forward path
        if hasattr(model, '_last_dcf_extras') and model._last_dcf_extras is not None:
            extras = model._last_dcf_extras
            if "per_roi_kappas" in extras:
                prk = extras["per_roi_kappas"]  # (B, n_rois, 1)
                all_per_roi_kappas.append(prk.squeeze(-1).cpu().numpy())
            if "delta" in extras:
                all_deltas.append(extras["delta"].squeeze(-1).cpu().numpy())
    
    # Reset per-ROI mode
    model._return_per_roi = False
    
    global_kappas = np.concatenate(all_global_kappas) if all_global_kappas else None
    global_preds = np.concatenate(all_global_preds)
    per_roi_kappas = np.concatenate(all_per_roi_kappas, axis=0) if all_per_roi_kappas else None
    deltas = np.concatenate(all_deltas) if all_deltas else None
    
    logger.info("Global kappas: %s", global_kappas.shape if global_kappas is not None else "None")
    logger.info("Per-ROI kappas: %s", per_roi_kappas.shape if per_roi_kappas is not None else "None")
    logger.info("Deltas (disagreement): %s", deltas.shape if deltas is not None else "None")
    logger.info("Global preds: %s", global_preds.shape)
    
    if global_kappas is not None:
        logger.info("Global kappa stats: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
                    global_kappas.mean(), global_kappas.std(), global_kappas.min(), global_kappas.max())
    
    if deltas is not None:
        logger.info("Delta (disagreement) stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                    deltas.mean(), deltas.std(), deltas.min(), deltas.max())
    
    if per_roi_kappas is not None:
        logger.info("\nPer-ROI kappa statistics:")
        for i, name in enumerate(roi_names[:per_roi_kappas.shape[1]]):
            k = per_roi_kappas[:, i]
            logger.info("  %-20s: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
                       name, k.mean(), k.std(), k.min(), k.max())
    
    return global_kappas, per_roi_kappas, global_preds


def load_coco_categories():
    """Load COCO category mapping."""
    cat_path = CACHE_ROOT / "nsd_category_labels.parquet"
    if cat_path.exists():
        return pd.read_parquet(cat_path)
    
    # Try CSV
    csv_path = REPO_ROOT / "data" / "nsd_stim_info_merged.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "cocoSplit" in df.columns:
            return df[["nsdId", "cocoId"]].drop_duplicates()
    
    logger.warning("No COCO category data found")
    return None


def aggregate_to_image(trial_values, nsd_ids, unique_nsd_ids):
    """Aggregate trial-level values to image-level (mean)."""
    result = np.zeros((len(unique_nsd_ids),) + trial_values.shape[1:])
    for i, nid in enumerate(unique_nsd_ids):
        mask = nsd_ids == nid
        result[i] = trial_values[mask].mean(axis=0)
    return result


def run_specialization_analysis(per_roi_kappas, roi_names, nsd_ids, unique_nsd_ids):
    """Test whether per-ROI kappas show functional specialization."""
    if per_roi_kappas is None:
        logger.warning("No per-ROI kappas available")
        return None
    
    image_kappas = aggregate_to_image(per_roi_kappas, nsd_ids, unique_nsd_ids)
    n_rois = min(len(roi_names), image_kappas.shape[1])
    
    logger.info("\n=== Per-ROI Kappa Hierarchy ===")
    roi_means = [(roi_names[i], image_kappas[:, i].mean(), image_kappas[:, i].std()) 
                 for i in range(n_rois)]
    roi_means.sort(key=lambda x: -x[1])
    
    for name, mean, std in roi_means:
        logger.info("  %-20s: κ = %.3f ± %.3f", name, mean, std)
    
    # Pairwise cosine similarity of ROI kappa profiles
    logger.info("\n=== Cross-ROI Kappa Profile Similarity ===")
    kappa_profiles = image_kappas.T[:n_rois]
    from numpy.linalg import norm
    cos_sim_matrix = np.zeros((n_rois, n_rois))
    for i in range(n_rois):
        for j in range(n_rois):
            cos_sim_matrix[i, j] = np.corrcoef(kappa_profiles[i], kappa_profiles[j])[0, 1]
    
    mean_off_diag = (cos_sim_matrix.sum() - np.trace(cos_sim_matrix)) / (n_rois * (n_rois - 1))
    logger.info("Mean pairwise ROI kappa correlation: %.4f", mean_off_diag)
    
    # ANOVA: do kappas differ across ROIs?
    from scipy.stats import f_oneway, kruskal
    roi_groups = [image_kappas[:, i] for i in range(n_rois)]
    f_stat, f_p = f_oneway(*roi_groups)
    h_stat, h_p = kruskal(*roi_groups)
    logger.info("One-way ANOVA across ROIs: F=%.2f, p=%.2e", f_stat, f_p)
    logger.info("Kruskal-Wallis across ROIs: H=%.2f, p=%.2e", h_stat, h_p)
    
    # Load COCO categories for category-conditional analysis
    cat_df = load_coco_categories()
    category_analysis = None
    
    if cat_df is not None and "supercategory" in cat_df.columns:
        logger.info("\n=== Category-Conditional ROI Kappa Analysis ===")
        
        nid_to_cat = dict(zip(cat_df["nsdId"], cat_df["supercategory"]))
        image_cats = [nid_to_cat.get(nid, "unknown") for nid in unique_nsd_ids]
        
        unique_cats = sorted(set(c for c in image_cats if c != "unknown"))
        logger.info("Categories found: %s", unique_cats)
        
        category_analysis = {}
        
        # For each category-selective ROI, test if kappa differs for preferred vs non-preferred
        specialization_tests = {
            "FFA1": "person",
            "FFA2": "person",
            "PPA": "outdoor",
            "EBA": "person",
            "OPA": "outdoor",
            "RSC": "outdoor",
        }
        
        logger.info("\n--- Functional Specialization Tests ---")
        for roi_name, preferred_cat in specialization_tests.items():
            if roi_name not in roi_names:
                continue
            roi_idx = roi_names.index(roi_name)
            
            preferred_mask = np.array([c == preferred_cat for c in image_cats])
            nonpreferred_mask = np.array([c != preferred_cat and c != "unknown" for c in image_cats])
            
            if preferred_mask.sum() < 10 or nonpreferred_mask.sum() < 10:
                continue
            
            kappa_preferred = image_kappas[preferred_mask, roi_idx]
            kappa_nonpreferred = image_kappas[nonpreferred_mask, roi_idx]
            
            t_stat, t_p = stats.ttest_ind(kappa_preferred, kappa_nonpreferred)
            d = (kappa_preferred.mean() - kappa_nonpreferred.mean()) / \
                np.sqrt((kappa_preferred.var() + kappa_nonpreferred.var()) / 2)
            
            direction = "HIGHER" if kappa_preferred.mean() > kappa_nonpreferred.mean() else "LOWER"
            
            logger.info("  %s (%s=%s): pref=%.3f, non-pref=%.3f, d=%.3f, p=%.4f [%s]",
                       roi_name, "prefer", preferred_cat,
                       kappa_preferred.mean(), kappa_nonpreferred.mean(),
                       d, t_p, direction)
            
            category_analysis[roi_name] = {
                "preferred_category": preferred_cat,
                "kappa_preferred": round(float(kappa_preferred.mean()), 4),
                "kappa_nonpreferred": round(float(kappa_nonpreferred.mean()), 4),
                "cohens_d": round(float(d), 4),
                "t_stat": round(float(t_stat), 4),
                "p_value": float(t_p),
                "direction": direction,
                "n_preferred": int(preferred_mask.sum()),
                "n_nonpreferred": int(nonpreferred_mask.sum()),
            }
        
        # Kappa-only category decoding
        logger.info("\n--- Kappa-Only Category Decoding ---")
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        known_mask = np.array([c != "unknown" for c in image_cats])
        X = image_kappas[known_mask, :n_rois]
        y = np.array([c for c, m in zip(image_cats, known_mask) if m])
        
        if len(np.unique(y)) >= 2:
            clf = LogisticRegression(max_iter=1000, C=1.0)
            scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
            
            # Permutation test
            n_perm = 200
            perm_scores = []
            for _ in range(n_perm):
                y_perm = np.random.permutation(y)
                perm_score = cross_val_score(clf, X, y_perm, cv=5, scoring="accuracy").mean()
                perm_scores.append(perm_score)
            
            perm_p = (np.array(perm_scores) >= scores.mean()).mean()
            
            logger.info("  Kappa-only decoding: acc=%.4f ± %.4f, chance=%.4f, perm_p=%.4f",
                       scores.mean(), scores.std(), 
                       np.mean(perm_scores), perm_p)
            
            category_analysis["kappa_only_decoding"] = {
                "accuracy": round(float(scores.mean()), 4),
                "accuracy_std": round(float(scores.std()), 4),
                "chance_level": round(float(np.mean(perm_scores)), 4),
                "permutation_p": round(float(perm_p), 4),
                "n_categories": len(np.unique(y)),
                "n_samples": len(y),
            }
    
    return {
        "roi_hierarchy": [{"roi": name, "mean_kappa": round(float(m), 4), "std_kappa": round(float(s), 4)}
                         for name, m, s in roi_means],
        "mean_pairwise_correlation": round(float(mean_off_diag), 4),
        "anova_f": round(float(f_stat), 2),
        "anova_p": float(f_p),
        "kruskal_h": round(float(h_stat), 2),
        "kruskal_p": float(h_p),
        "category_analysis": category_analysis,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, roi_names, subject_idx, config = load_vmf_dcf_model()
    
    # Load data
    features, nsd_ids, unique_nsd_ids = load_shared1000_data_zscored("N3v9_roi_dcf")
    
    # Extract per-ROI kappas
    global_kappas, per_roi_kappas, global_preds = extract_per_roi_kappas(
        model, features, roi_names, subject_idx
    )
    
    # Save raw kappas
    if global_kappas is not None:
        np.save(OUTPUT_DIR / "vmf_dcf_global_kappas.npy", global_kappas)
    if per_roi_kappas is not None:
        np.save(OUTPUT_DIR / "vmf_dcf_per_roi_kappas.npy", per_roi_kappas)
    np.save(OUTPUT_DIR / "vmf_dcf_nsd_ids.npy", nsd_ids)
    
    # Run specialization analysis
    results = run_specialization_analysis(per_roi_kappas, roi_names, nsd_ids, unique_nsd_ids)
    
    # Also analyze global kappa
    if global_kappas is not None:
        image_global_kappas = aggregate_to_image(
            global_kappas.reshape(-1, 1), nsd_ids, unique_nsd_ids
        ).squeeze()
        
        logger.info("\n=== vmf_dcf Global Kappa Stats ===")
        logger.info("  mean=%.3f, std=%.3f, range=[%.3f, %.3f]",
                    image_global_kappas.mean(), image_global_kappas.std(),
                    image_global_kappas.min(), image_global_kappas.max())
        
        if results:
            results["global_kappa_stats"] = {
                "mean": round(float(image_global_kappas.mean()), 3),
                "std": round(float(image_global_kappas.std()), 3),
                "min": round(float(image_global_kappas.min()), 3),
                "max": round(float(image_global_kappas.max()), 3),
            }
    
    # Save results
    with open(OUTPUT_DIR / "vmf_dcf_specialization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("\nResults saved to %s", OUTPUT_DIR)
    logger.info("Done!")
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
