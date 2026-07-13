"""
Semantic Alignment Analysis: Do per-ROI mu vectors recover functional specialization?

Key insight: While kappa (confidence) doesn't encode category, the per-ROI mu vectors
(embedding directions) should show category-selective patterns if the model learned
meaningful ROI specialization.

Tests:
1. Per-ROI mu alignment with category-mean CLIP embeddings
2. Category-selective projection: FFA mu aligns with "person" more than PPA mu
3. Selectivity Index for each ROI
4. Kappa-independent decoding: category prediction from per-ROI mu similarity profiles
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
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("NSD_DATA_ROOT", "/home/jovyan/work/data/nsd")
os.environ.setdefault("CACHE_ROOT", "/home/jovyan/work/FMRI2images/cache")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUBJECT = os.environ.get("SUBJECT", "subj01")
CHECKPOINT = Path(f"experimental_results/V66a_roi_pretrain/{SUBJECT}/checkpoint_best.pt")
OUTPUT_DIR = Path(f"experimental_results/UDND_analysis/{SUBJECT}/semantic_alignment")
CACHE_ROOT = Path(os.environ["CACHE_ROOT"])


def load_model_and_extract():
    """Load model and extract per-ROI mu for all trials."""
    from fmri2img.models.unified_model import create_model
    
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model_config = ckpt.get("model_config", config.get("model", {}))
    
    subject_roi_dims = model_config.get("encoder", {}).get("subject_roi_dims", {}).get(SUBJECT, {})
    if not subject_roi_dims:
        subject_roi_dims = model_config.get("encoder", {}).get("roi_dims", {})
    
    roi_names = list(subject_roi_dims.keys())
    total_voxels = sum(subject_roi_dims.values())
    
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
    
    roi_indices = subject_roi_indices.get(SUBJECT, {})
    encoder_cfg = model_config.get("encoder", {})
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
    logger.info("Model loaded: %s", type(model.encoder).__name__)
    
    # Load features
    features = np.load(CACHE_ROOT / "preextracted" / f"subject={SUBJECT}" / "fmri_features.npy")
    logger.info("Features: %s", features.shape)
    
    # Extract per-ROI mu vectors
    batch_size = 64  # smaller because we store 16 x 768 per trial
    n_trials = len(features)
    all_roi_mus = []
    all_cls_mus = []
    all_alphas = []
    
    logger.info("Extracting per-ROI mu vectors...")
    t0 = time.time()
    
    with torch.no_grad():
        for start in range(0, n_trials, batch_size):
            end = min(start + batch_size, n_trials)
            batch = torch.from_numpy(features[start:end]).float().to(DEVICE)
            subject_ids = torch.zeros(len(batch), dtype=torch.long, device=DEVICE)
            
            encoder_out = model.encoder(batch, subject_ids, return_roi_tokens=True)
            B, n_rois, d = encoder_out.roi_tokens.shape
            
            # Decode CLS
            cls_mu, _ = model.decoder(encoder_out.cls_out)
            
            # Decode each ROI token
            roi_tokens_flat = encoder_out.roi_tokens.reshape(B * n_rois, d)
            roi_mu_flat, _ = model.decoder(roi_tokens_flat)
            roi_mu = roi_mu_flat.reshape(B, n_rois, -1)
            
            all_roi_mus.append(roi_mu.cpu().numpy())
            all_cls_mus.append(cls_mu.cpu().numpy())
            all_alphas.append(encoder_out.cls_to_roi_alpha.cpu().numpy())
            
            if (start // batch_size) % 100 == 0:
                logger.info("  %d/%d (%.1f%%) %.1fs", end, n_trials, 100*end/n_trials, time.time()-t0)
    
    roi_mus = np.concatenate(all_roi_mus, axis=0)    # (N, 16, 768)
    cls_mus = np.concatenate(all_cls_mus, axis=0)    # (N, 768)
    alphas = np.concatenate(all_alphas, axis=0)      # (N, 16)
    
    logger.info("Extraction done in %.1fs. roi_mus: %s", time.time()-t0, roi_mus.shape)
    
    return model, roi_names, roi_mus, cls_mus, alphas


def compute_category_centroids():
    """Compute mean CLIP embedding per COCO supercategory."""
    clip_path = CACHE_ROOT.parent / "outputs" / "clip_cache" / "clip.parquet"
    clip_df = pd.read_parquet(clip_path)
    
    # Identify embedding column
    if "clip_embedding" in clip_df.columns:
        embeddings = np.stack(clip_df["clip_embedding"].values)
    elif "embedding" in clip_df.columns:
        embeddings = np.stack(clip_df["embedding"].values)
    else:
        emb_cols = sorted([c for c in clip_df.columns if c.startswith("emb_")])
        embeddings = clip_df[emb_cols].values
    
    nsd_ids = clip_df["nsdId"].values
    logger.info("CLIP cache: %d embeddings, dim=%d", len(embeddings), embeddings.shape[1])
    
    # Load categories
    cat_df = pd.read_parquet(CACHE_ROOT / "nsd_category_labels.parquet")
    nsd_to_cat = dict(zip(cat_df["nsdId"].values, cat_df["supercategory"].values))
    
    # Compute centroids
    categories = np.array([nsd_to_cat.get(int(nid), "unknown") for nid in nsd_ids])
    unique_cats = sorted(set(categories) - {"unknown"})
    
    centroids = {}
    for cat in unique_cats:
        mask = categories == cat
        if mask.sum() > 0:
            centroid = embeddings[mask].mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            centroids[cat] = centroid
    
    logger.info("Computed centroids for %d categories", len(centroids))
    return centroids, unique_cats


def run_semantic_alignment_analysis(roi_names, roi_mus, cls_mus, centroids, unique_cats, meta_df, cat_df):
    """
    Core analysis: measure cosine similarity between each ROI's mu and each category centroid.
    
    For each trial:
    - Compute cos(mu_roi, centroid_cat) for all (ROI, category) pairs
    - Average across trials of each category
    - This gives a (n_cats x n_rois) "alignment matrix"
    """
    from scipy import stats
    
    logger.info("\n" + "=" * 70)
    logger.info("SEMANTIC ALIGNMENT ANALYSIS")
    logger.info("=" * 70)
    
    nsd_ids = meta_df["nsdId"].values
    nsd_to_cat = dict(zip(cat_df["nsdId"].values, cat_df["supercategory"].values))
    trial_cats = np.array([nsd_to_cat.get(int(nid), "unknown") for nid in nsd_ids])
    valid = trial_cats != "unknown"
    
    roi_mus_valid = roi_mus[valid]
    trial_cats_valid = trial_cats[valid]
    n_trials = valid.sum()
    n_rois = len(roi_names)
    n_cats = len(unique_cats)
    
    # Normalize per-ROI mu vectors
    roi_mus_norm = roi_mus_valid / (np.linalg.norm(roi_mus_valid, axis=-1, keepdims=True) + 1e-8)
    
    # Build centroid matrix (n_cats, 768)
    centroid_matrix = np.stack([centroids[cat] for cat in unique_cats])
    
    # Compute per-ROI alignment to each category centroid: (n_trials, n_rois, n_cats)
    # roi_mus_norm: (N, n_rois, 768), centroid_matrix: (n_cats, 768)
    # alignment: (N, n_rois, n_cats) = roi_mus_norm @ centroid_matrix.T
    alignment = np.einsum('nrd,cd->nrc', roi_mus_norm, centroid_matrix)
    
    # Average alignment by trial category: for trials of cat_c, what's the mean alignment?
    # Result: (n_cats_trial x n_rois x n_cats_centroid)
    # Simpler: for each trial category, compute mean alignment matrix (n_rois x n_cats_centroid)
    alignment_by_cat = np.zeros((n_cats, n_rois, n_cats))
    for i, cat in enumerate(unique_cats):
        mask = trial_cats_valid == cat
        if mask.sum() > 0:
            alignment_by_cat[i] = alignment[mask].mean(axis=0)
    
    # KEY METRIC: ROI Selectivity — for each ROI, does its mu preferentially 
    # align with the "correct" category centroid when viewing that category?
    # Selectivity = alignment_to_own_category - mean_alignment_to_other_categories
    logger.info("\n--- ROI Category Selectivity ---")
    selectivity = {}
    for j, roi in enumerate(roi_names):
        # For each category: alignment when viewing that category vs others
        own_alignments = []
        other_alignments = []
        for i, cat in enumerate(unique_cats):
            own_alignments.append(alignment_by_cat[i, j, i])
            others = [alignment_by_cat[i, j, k] for k in range(n_cats) if k != i]
            other_alignments.append(np.mean(others))
        
        mean_selectivity = np.mean(np.array(own_alignments) - np.array(other_alignments))
        selectivity[roi] = float(mean_selectivity)
    
    # Sort by selectivity
    sorted_selectivity = sorted(selectivity.items(), key=lambda x: x[1], reverse=True)
    for roi, sel in sorted_selectivity:
        logger.info("  %-20s selectivity=%.6f", roi, sel)
    
    # KEY TEST: Category-specific alignment for neuroscience-relevant ROIs
    logger.info("\n--- Hypothesis Tests: Known Functional Specialization ---")
    hypothesis_tests = {}
    
    tests = [
        ("FFA1", "person", "Does FFA1 mu align more with person centroid?"),
        ("FFA2", "person", "Does FFA2 mu align more with person centroid?"),
        ("EBA", "person", "Does EBA mu align more with person centroid?"),
        ("OFA", "person", "Does OFA mu align more with person centroid?"),
        ("PPA", "outdoor", "Does PPA mu align more with outdoor centroid?"),
        ("OPA", "outdoor", "Does OPA mu align more with outdoor centroid?"),
    ]
    
    for roi, expected_cat, description in tests:
        if roi not in roi_names or expected_cat not in unique_cats:
            continue
        
        j = roi_names.index(roi)
        cat_idx = unique_cats.index(expected_cat)
        
        # For trials of the expected category:
        # Compare THIS roi's alignment to expected centroid vs OTHER rois' alignment
        mask = trial_cats_valid == expected_cat
        roi_alignment = alignment[mask, j, cat_idx]  # this ROI -> expected centroid
        
        # Mean of all other ROIs -> expected centroid (for same trials)
        other_roi_idx = [k for k in range(n_rois) if k != j]
        other_alignment = alignment[mask][:, other_roi_idx, cat_idx].mean(axis=1)
        
        # Paired t-test
        t_stat, p_val = stats.ttest_rel(roi_alignment, other_alignment)
        d = (roi_alignment.mean() - other_alignment.mean()) / roi_alignment.std() if roi_alignment.std() > 0 else 0
        
        result = {
            "roi": roi,
            "expected_category": expected_cat,
            "description": description,
            "roi_mean_alignment": float(roi_alignment.mean()),
            "other_rois_mean_alignment": float(other_alignment.mean()),
            "difference": float(roi_alignment.mean() - other_alignment.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(d),
            "n_trials": int(mask.sum()),
            "significant_bonferroni": p_val < 0.05 / len(tests),
        }
        hypothesis_tests[f"{roi}_to_{expected_cat}"] = result
        
        sig = "***" if result["significant_bonferroni"] else "n.s."
        logger.info("  %s -> %s: d=%.4f, t=%.2f, p=%.2e [%s]",
                   roi, expected_cat, d, t_stat, p_val, sig)
    
    # BROADER TEST: Does each ROI preferentially align with its OWN category?
    logger.info("\n--- Per-ROI Preferred Category (by alignment) ---")
    roi_preferences = {}
    for j, roi in enumerate(roi_names):
        # Which category centroid does this ROI most align with when viewing THAT category?
        cat_specific_alignment = np.zeros(n_cats)
        for i, cat in enumerate(unique_cats):
            mask = trial_cats_valid == cat
            cat_specific_alignment[i] = alignment[mask, j, i].mean()
        
        best_cat_idx = cat_specific_alignment.argmax()
        best_cat = unique_cats[best_cat_idx]
        
        # Compute effect: best vs mean of rest
        best_val = cat_specific_alignment[best_cat_idx]
        rest_mean = np.delete(cat_specific_alignment, best_cat_idx).mean()
        
        roi_preferences[roi] = {
            "preferred_category": best_cat,
            "alignment_preferred": float(best_val),
            "alignment_others": float(rest_mean),
            "advantage": float(best_val - rest_mean),
        }
        logger.info("  %-20s prefers %-12s (advantage=%.5f)", roi, best_cat, best_val - rest_mean)
    
    # DECODING TEST: Can we decode category from the per-ROI alignment profile?
    logger.info("\n--- Category Decoding from ROI Alignment Profiles ---")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    
    # Feature: per-ROI alignment to all centroids (n_trials, n_rois * n_cats = 16*12 = 192)
    features_alignment = alignment[..., :].reshape(n_trials, -1)  # (N, 192)
    
    # Also try: just the per-ROI mu cosine-similarity profile (n_rois, n_cats)
    # And: per-ROI kappa (for comparison)
    
    feature_sets = {
        "roi_alignment_profile_192d": features_alignment,
        "roi_mu_norms_16d": np.linalg.norm(roi_mus_valid, axis=-1),
    }
    
    for feat_name, features in feature_sets.items():
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_accs = []
        fold_bal_accs = []
        
        for train_idx, val_idx in skf.split(features, trial_cats_valid):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(features[train_idx])
            X_val = scaler.transform(features[val_idx])
            
            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
            clf.fit(X_train, trial_cats_valid[train_idx])
            
            y_pred = clf.predict(X_val)
            fold_accs.append(accuracy_score(trial_cats_valid[val_idx], y_pred))
            fold_bal_accs.append(balanced_accuracy_score(trial_cats_valid[val_idx], y_pred))
        
        logger.info("  %s: acc=%.4f, balanced_acc=%.4f (chance=%.4f)",
                   feat_name, np.mean(fold_accs), np.mean(fold_bal_accs), 1/n_cats)
    
    return {
        "selectivity": selectivity,
        "hypothesis_tests": hypothesis_tests,
        "roi_preferences": roi_preferences,
        "alignment_matrix_shape": list(alignment_by_cat.shape),
        "category_names": unique_cats,
        "roi_names": roi_names,
    }


def run_cross_roi_similarity_analysis(roi_names, roi_mus, meta_df, cat_df):
    """
    Analyze how similar per-ROI mu vectors are to each other.
    Higher similarity between functionally related ROIs (e.g., FFA1/FFA2, V1v/V1d).
    """
    from scipy import stats
    
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-ROI SIMILARITY STRUCTURE")
    logger.info("=" * 70)
    
    nsd_ids = meta_df["nsdId"].values
    nsd_to_cat = dict(zip(cat_df["nsdId"].values, cat_df["supercategory"].values))
    trial_cats = np.array([nsd_to_cat.get(int(nid), "unknown") for nid in nsd_ids])
    valid = trial_cats != "unknown"
    
    roi_mus_valid = roi_mus[valid]
    roi_mus_norm = roi_mus_valid / (np.linalg.norm(roi_mus_valid, axis=-1, keepdims=True) + 1e-8)
    
    # Mean per-ROI mu similarity (across all trials)
    n_rois = len(roi_names)
    mean_roi_mus = roi_mus_norm.mean(axis=0)  # (n_rois, 768)
    mean_roi_mus = mean_roi_mus / (np.linalg.norm(mean_roi_mus, axis=-1, keepdims=True) + 1e-8)
    
    # Pairwise cosine similarity between mean ROI mus
    roi_sim = mean_roi_mus @ mean_roi_mus.T  # (n_rois, n_rois)
    
    logger.info("Mean pairwise ROI mu similarity: %.4f +/- %.4f", 
               roi_sim[np.triu_indices(n_rois, k=1)].mean(),
               roi_sim[np.triu_indices(n_rois, k=1)].std())
    
    # Most similar pairs
    logger.info("\n--- Most Similar ROI Pairs ---")
    pairs = []
    for i in range(n_rois):
        for j in range(i+1, n_rois):
            pairs.append((roi_names[i], roi_names[j], roi_sim[i, j]))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    for r1, r2, sim in pairs[:10]:
        logger.info("  %-10s - %-10s: %.4f", r1, r2, sim)
    
    # Category-conditional change: how does pairwise ROI similarity change by category?
    logger.info("\n--- Category-Conditional ROI Coupling ---")
    trial_cats_valid = trial_cats[valid]
    unique_cats = sorted(set(trial_cats_valid) - {"unknown"})
    
    coupling_by_cat = {}
    for cat in unique_cats:
        mask = trial_cats_valid == cat
        cat_mus = roi_mus_norm[mask]  # (n_cat_trials, n_rois, 768)
        cat_mean = cat_mus.mean(axis=0)  # (n_rois, 768)
        cat_mean = cat_mean / (np.linalg.norm(cat_mean, axis=-1, keepdims=True) + 1e-8)
        cat_sim = cat_mean @ cat_mean.T
        coupling_by_cat[cat] = cat_sim
    
    # Find ROI pairs whose coupling changes most across categories
    coupling_variance = np.zeros((n_rois, n_rois))
    for i in range(n_rois):
        for j in range(i+1, n_rois):
            sims = [coupling_by_cat[cat][i, j] for cat in unique_cats]
            coupling_variance[i, j] = np.std(sims)
    
    logger.info("  Pairs with highest category-conditional coupling variance:")
    var_pairs = []
    for i in range(n_rois):
        for j in range(i+1, n_rois):
            var_pairs.append((roi_names[i], roi_names[j], coupling_variance[i, j]))
    var_pairs.sort(key=lambda x: x[2], reverse=True)
    for r1, r2, var in var_pairs[:10]:
        logger.info("    %-10s - %-10s: variance=%.6f", r1, r2, var)
    
    return {
        "mean_pairwise_similarity": float(roi_sim[np.triu_indices(n_rois, k=1)].mean()),
        "roi_similarity_matrix": roi_sim.tolist(),
        "top_similar_pairs": [(r1, r2, float(s)) for r1, r2, s in pairs[:10]],
        "category_coupling_variance": [(r1, r2, float(v)) for r1, r2, v in var_pairs[:10]],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model and extract per-ROI mu vectors
    model, roi_names, roi_mus, cls_mus, alphas = load_model_and_extract()
    
    # Load metadata
    meta_df = pd.read_parquet(CACHE_ROOT / "preextracted" / f"subject={SUBJECT}" / "trial_meta.parquet")
    cat_df = pd.read_parquet(CACHE_ROOT / "nsd_category_labels.parquet")
    
    # Compute category centroids from CLIP cache
    centroids, unique_cats = compute_category_centroids()
    
    # Run semantic alignment analysis
    alignment_results = run_semantic_alignment_analysis(
        roi_names, roi_mus, cls_mus, centroids, unique_cats, meta_df, cat_df
    )
    
    # Run cross-ROI similarity analysis
    similarity_results = run_cross_roi_similarity_analysis(
        roi_names, roi_mus, meta_df, cat_df
    )
    
    # Save all results
    all_results = {
        "subject": SUBJECT,
        "n_trials": len(roi_mus),
        "n_rois": len(roi_names),
        "roi_names": roi_names,
        "semantic_alignment": alignment_results,
        "cross_roi_similarity": similarity_results,
    }
    
    output_file = OUTPUT_DIR / "semantic_alignment_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    
    logger.info("\nAll results saved to: %s", output_file)
    logger.info("Done!")


if __name__ == "__main__":
    main()
