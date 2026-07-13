"""
Train Encoding Model: CLIP embeddings -> fMRI per ROI.

Trains Ridge regression models predicting fMRI activity from CLIP embeddings,
then computes bidirectional consistency (correlation between decoder kappa
and encoder R-squared across ROIs).
"""
import sys
sys.path.insert(0, "/home/jovyan/work/FMRI2images/src")

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("NSD_DATA_ROOT", "/home/jovyan/work/data/nsd")
os.environ.setdefault("CACHE_ROOT", "/home/jovyan/work/FMRI2images/cache")

SUBJECT = "subj01"
KAPPA_DIR = Path("experimental_results/UDND_analysis/subj01/roi_kappa_results")
OUTPUT_DIR = Path("experimental_results/UDND_analysis/subj01/encoding_model")


def load_data():
    """Load fMRI features, CLIP embeddings, and ROI kappa data."""
    # fMRI features
    features_path = Path(os.environ["CACHE_ROOT"]) / "preextracted" / f"subject={SUBJECT}" / "fmri_features.npy"
    fmri = np.load(features_path)
    logger.info("fMRI features: %s", fmri.shape)
    
    # Trial metadata
    meta_path = Path(os.environ["CACHE_ROOT"]) / "preextracted" / f"subject={SUBJECT}" / "trial_meta.parquet"
    meta_df = pd.read_parquet(meta_path)
    
    # CLIP embeddings
    clip_path = Path("outputs/clip_cache/clip.parquet")
    clip_df = pd.read_parquet(clip_path)
    
    # Build nsdId -> embedding mapping
    nsd_ids_clip = clip_df["nsdId"].values
    clip_embeds = np.stack(clip_df["clip_embedding"].values).astype(np.float32)
    id_to_embed = {int(nid): clip_embeds[i] for i, nid in enumerate(nsd_ids_clip)}
    
    # Map trials to embeddings - filter to only valid trials
    trial_nsd_ids = meta_df["nsdId"].values
    valid_mask = np.array([int(nid) in id_to_embed for nid in trial_nsd_ids])
    logger.info("Valid trials (have CLIP): %d/%d", valid_mask.sum(), len(valid_mask))
    
    # Only use valid trials
    valid_indices = np.where(valid_mask)[0]
    trial_embeds = np.array([id_to_embed[int(trial_nsd_ids[i])] for i in valid_indices])
    fmri = fmri[valid_indices]
    
    # ROI kappa data
    meta_json = json.loads((KAPPA_DIR / "roi_kappa_meta.json").read_text())
    roi_names = meta_json["roi_names"]
    per_roi_kappas = np.load(KAPPA_DIR / "roi_kappa_per_roi_kappas.npy")
    per_roi_kappas = per_roi_kappas[valid_indices]
    
    return fmri, trial_embeds, valid_mask, roi_names, per_roi_kappas, meta_json


def build_roi_voxel_mapping(roi_names, subject):
    """Build mapping from ROI names to voxel indices in the flat feature array."""
    # Load from checkpoint metadata
    ckpt_path = Path("experimental_results/V66a_roi_pretrain/subj01/checkpoint_best.pt")
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_config = ckpt.get("model_config", ckpt["config"].get("model", {}))
    subject_roi_dims = model_config.get("encoder", {}).get("subject_roi_dims", {}).get(subject, {})
    
    roi_voxel_map = {}
    offset = 0
    for roi_name, n_voxels in subject_roi_dims.items():
        roi_voxel_map[roi_name] = slice(offset, offset + n_voxels)
        offset += n_voxels
    
    logger.info("ROI voxel mapping: %d ROIs, total %d voxels",
                len(roi_voxel_map), offset)
    return roi_voxel_map


def train_encoding_models(clip_embeds, fmri, roi_voxel_map, train_frac=0.85):
    """
    Train Ridge regression: CLIP (768-D) -> fMRI voxels per ROI.
    Returns R-squared per ROI.
    """
    n_train = int(len(clip_embeds) * train_frac)
    X_train = clip_embeds[:n_train]
    X_val = clip_embeds[n_train:]
    
    results = {}
    
    for roi_name, voxel_slice in roi_voxel_map.items():
        Y_train = fmri[:n_train, voxel_slice]
        Y_val = fmri[n_train:, voxel_slice]
        
        n_voxels = Y_train.shape[1]
        
        # Ridge regression
        ridge = Ridge(alpha=1000.0, fit_intercept=True)
        ridge.fit(X_train, Y_train)
        
        Y_pred = ridge.predict(X_val)
        
        # Per-voxel R2 (sample if too many voxels for speed)
        max_eval_voxels = min(n_voxels, 500)
        eval_indices = np.random.choice(n_voxels, max_eval_voxels, replace=False) if n_voxels > 500 else np.arange(n_voxels)
        
        r2_per_voxel = np.array([
            r2_score(Y_val[:, v], Y_pred[:, v]) 
            for v in eval_indices if Y_val[:, v].std() > 1e-6
        ])
        
        # Fraction of voxels with positive R2
        frac_positive = (r2_per_voxel > 0).mean()
        
        mean_r2 = r2_per_voxel.mean() if len(r2_per_voxel) > 0 else 0
        median_r2 = np.median(r2_per_voxel) if len(r2_per_voxel) > 0 else 0
        max_r2 = r2_per_voxel.max() if len(r2_per_voxel) > 0 else 0
        
        results[roi_name] = {
            "mean_r2": float(mean_r2),
            "median_r2": float(median_r2),
            "max_r2": float(max_r2),
            "frac_positive_r2": float(frac_positive),
            "n_voxels": n_voxels,
            "n_voxels_valid": len(r2_per_voxel),
        }
        
        logger.info("  %-20s: mean_R2=%.4f, median=%.4f, max=%.4f (%d voxels, %.1f%% positive)",
                   roi_name, mean_r2, median_r2, max_r2, n_voxels, frac_positive * 100)
    
    return results


def compute_bidirectional_consistency(encoding_results, per_roi_kappas, roi_names):
    """
    Correlate encoder R2 with decoder kappa across ROIs.
    Strong correlation = bidirectional neural code theory.
    """
    # Mean decoder kappa per ROI
    decoder_kappa_per_roi = per_roi_kappas.mean(axis=0)  # (n_rois,)
    
    # Encoder R2 per ROI
    encoder_r2_per_roi = np.array([
        encoding_results.get(roi, {}).get("mean_r2", 0) for roi in roi_names
    ])
    
    # Correlation
    valid = ~np.isnan(encoder_r2_per_roi) & ~np.isnan(decoder_kappa_per_roi)
    if valid.sum() < 3:
        logger.warning("Not enough valid ROIs for correlation")
        return {"correlation": 0, "p_value": 1.0}
    
    r, p = stats.pearsonr(encoder_r2_per_roi[valid], decoder_kappa_per_roi[valid])
    rho, p_spearman = stats.spearmanr(encoder_r2_per_roi[valid], decoder_kappa_per_roi[valid])
    
    logger.info("\n=== Bidirectional Consistency ===")
    logger.info("  Pearson r = %.4f (p = %.6f)", r, p)
    logger.info("  Spearman rho = %.4f (p = %.6f)", rho, p_spearman)
    
    return {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "spearman_rho": float(rho),
        "spearman_p": float(p_spearman),
        "encoder_r2_per_roi": {roi: float(v) for roi, v in zip(roi_names, encoder_r2_per_roi)},
        "decoder_kappa_per_roi": {roi: float(v) for roi, v in zip(roi_names, decoder_kappa_per_roi)},
        "n_rois_valid": int(valid.sum()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fmri, trial_embeds, valid_mask, roi_names, per_roi_kappas, meta_json = load_data()
    
    # Build ROI voxel mapping
    roi_voxel_map = build_roi_voxel_mapping(roi_names, SUBJECT)
    
    # Train encoding models (CLIP -> fMRI per ROI)
    logger.info("\n=== Training Encoding Models ===")
    encoding_results = train_encoding_models(trial_embeds, fmri, roi_voxel_map)
    
    # Compute bidirectional consistency
    consistency = compute_bidirectional_consistency(encoding_results, per_roi_kappas, roi_names)
    
    # Save results
    all_results = {
        "encoding_per_roi": encoding_results,
        "bidirectional_consistency": consistency,
        "subject": SUBJECT,
        "n_trials": len(fmri),
        "n_rois": len(roi_names),
        "roi_names": roi_names,
    }
    
    with open(OUTPUT_DIR / "encoding_model_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    logger.info("\n=== ENCODING MODEL RANKING (by mean R2) ===")
    ranked = sorted(encoding_results.items(), key=lambda x: x[1]["mean_r2"], reverse=True)
    for rank, (roi, res) in enumerate(ranked, 1):
        logger.info("  %2d. %-20s mean_R2=%.4f, max_R2=%.4f", rank, roi, res["mean_r2"], res["max_r2"])
    
    logger.info("\nDone! Results saved to: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
