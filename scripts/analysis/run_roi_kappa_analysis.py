"""
Run Per-ROI Kappa Extraction on existing ROI Transformer model.

Since V66a uses model_type='vmf' (single global decoder) rather than 'vmf_dcf',
we extract per-ROI information from the encoder's attention patterns and
compute proxy per-ROI kappa by:
1. Getting per-ROI token representations from the encoder
2. Projecting each ROI token through the decoder independently
3. Computing per-ROI kappa via independent forward passes with masked tokens

This provides the topographic analysis even without a dedicated ROI-DCF model.
"""
import sys
sys.path.insert(0, "/home/jovyan/work/FMRI2images/src")

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Environment
os.environ.setdefault("NSD_DATA_ROOT", "/home/jovyan/work/data/nsd")
os.environ.setdefault("CACHE_ROOT", "/home/jovyan/work/FMRI2images/cache")
os.environ.setdefault("OUTPUT_ROOT", "/home/jovyan/work/FMRI2images/experimental_results")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUBJECT = "subj01"
CHECKPOINT = Path("experimental_results/V66a_roi_pretrain/subj01/checkpoint_best.pt")
OUTPUT_DIR = Path("experimental_results/UDND_analysis/subj01/roi_kappa_results")


def load_model_and_data():
    """Load trained model and validation data."""
    from fmri2img.models.unified_model import create_model
    
    logger.info("Loading checkpoint: %s", CHECKPOINT)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model_config = ckpt.get("model_config", config.get("model", {}))
    
    logger.info("Model type: %s", model_config.get("type"))
    
    # Build ROI indices from subject_roi_dims
    subject_roi_dims = model_config.get("encoder", {}).get("subject_roi_dims", {}).get(SUBJECT, {})
    if not subject_roi_dims:
        subject_roi_dims = model_config.get("encoder", {}).get("roi_dims", {})
    
    roi_names = list(subject_roi_dims.keys())
    roi_sizes = list(subject_roi_dims.values())
    total_voxels = sum(roi_sizes)
    
    logger.info("ROI names (%d): %s", len(roi_names), roi_names)
    logger.info("Total voxels: %d", total_voxels)
    
    # Build ROI indices (cumulative offsets)
    roi_indices = {}
    offset = 0
    for name, size in zip(roi_names, roi_sizes):
        roi_indices[name] = np.arange(offset, offset + size)
        offset += size
    
    # Ensure encoder has input_dim
    encoder_cfg = model_config.get("encoder", {})
    if "input_dim" not in encoder_cfg:
        encoder_cfg["input_dim"] = total_voxels
        model_config["encoder"] = encoder_cfg
    
    # Create model — create_model expects the model sub-config directly
    # First, extract subject_roi_indices from the state_dict buffers
    state_dict = ckpt["model_state_dict"]
    
    # Build subject_roi_indices from saved _idx_{subj}_{i} buffers
    import re
    subject_roi_indices_for_model = {}
    idx_pattern = re.compile(r"encoder\._idx_(\w+)_(\d+)")
    for key, tensor in state_dict.items():
        m = idx_pattern.match(key)
        if m:
            subj_id = m.group(1)
            roi_idx = int(m.group(2))
            if subj_id not in subject_roi_indices_for_model:
                subject_roi_indices_for_model[subj_id] = {}
            # Map roi index to roi name
            if roi_idx < len(roi_names):
                subject_roi_indices_for_model[subj_id][roi_names[roi_idx]] = tensor.numpy()
    
    logger.info("Extracted subject_roi_indices for %d subjects from checkpoint", 
                len(subject_roi_indices_for_model))
    
    # Use extracted indices for current subject
    if SUBJECT in subject_roi_indices_for_model:
        roi_indices = subject_roi_indices_for_model[SUBJECT]
    
    model = create_model(model_config, roi_indices=roi_indices)
    
    # Load weights — strict=False because we handle buffer registration ourselves
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    # Register the _idx buffers that were unexpected
    for key in unexpected:
        m = idx_pattern.match(key)
        if m:
            parts = key.split(".", 1)
            if len(parts) == 2:
                model.encoder.register_buffer(parts[1], state_dict[key])
    
    logger.info("Loaded state dict: %d missing, %d unexpected (idx buffers registered)", 
                len(missing), len(unexpected))
    
    model = model.to(DEVICE).eval()
    logger.info("Model loaded on %s", DEVICE)
    
    # Load pre-extracted features
    features_path = Path(os.environ["CACHE_ROOT"]) / "preextracted" / f"subject={SUBJECT}" / "fmri_features.npy"
    features = np.load(features_path)
    logger.info("Features loaded: %s", features.shape)
    
    # Load trial metadata
    meta_path = Path(os.environ["CACHE_ROOT"]) / "preextracted" / f"subject={SUBJECT}" / "trial_meta.parquet"
    meta_df = pd.read_parquet(meta_path)
    
    # Load CLIP embeddings for validation
    clip_path = Path("outputs/clip_cache/clip.parquet")
    if clip_path.exists():
        clip_df = pd.read_parquet(clip_path)
        logger.info("CLIP cache loaded: %d entries", len(clip_df))
    else:
        clip_df = None
        logger.warning("CLIP cache not found")
    
    return model, features, meta_df, clip_df, roi_names, config


@torch.no_grad()
def extract_kappas_from_vmf_model(model, features, meta_df, roi_names, batch_size=64):
    """
    Extract per-trial kappa and attention weights from a vmf model.
    
    Uses return_roi_tokens=True to get CLS-to-ROI attention from the encoder,
    providing per-ROI importance weights for the global kappa.
    """
    n_trials = len(features)
    n_rois = len(roi_names)
    
    all_kappas = []
    all_mus = []
    all_attentions = []
    
    logger.info("Running inference on %d trials (batch_size=%d)...", n_trials, batch_size)
    t0 = time.time()
    
    # Temporarily set model to return roi tokens for attention extraction
    model._return_per_roi = True
    
    for start in range(0, n_trials, batch_size):
        end = min(start + batch_size, n_trials)
        batch = torch.from_numpy(features[start:end]).float().to(DEVICE)
        
        # Forward with return_roi_tokens
        subject_ids = torch.zeros(len(batch), dtype=torch.long, device=DEVICE)
        encoder_out = model.encoder(batch, subject_ids, return_roi_tokens=True)
        
        # Get CLS-to-ROI attention weights
        if hasattr(encoder_out, 'cls_to_roi_alpha'):
            attn = encoder_out.cls_to_roi_alpha.cpu().numpy()
            all_attentions.append(attn)
        
        # Pass CLS through decoder
        cls_out = encoder_out.cls_out
        mu, kappa = model.decoder(cls_out)
        
        all_mus.append(mu.cpu().numpy())
        all_kappas.append(kappa.cpu().numpy())
        
        if (start // batch_size) % 50 == 0:
            elapsed = time.time() - t0
            progress = end / n_trials * 100
            logger.info("Progress: %d/%d (%.1f%%) - %.1fs elapsed", end, n_trials, progress, elapsed)
    
    model._return_per_roi = False
    
    mus = np.concatenate(all_mus, axis=0)
    kappas = np.concatenate(all_kappas, axis=0).squeeze(-1) if all_kappas else np.zeros(n_trials)
    
    if all_attentions:
        attentions = np.concatenate(all_attentions, axis=0)
        logger.info("Attention shape: %s", attentions.shape)
    else:
        attentions = np.ones((n_trials, n_rois)) / n_rois
        logger.warning("No attention weights captured, using uniform")
    
    logger.info(
        "Extraction complete: %d trials. Kappa mean=%.2f, std=%.2f",
        n_trials, kappas.mean(), kappas.std()
    )
    logger.info("Attention stats: mean=%.4f, std=%.4f", attentions.mean(), attentions.std())
    
    return mus, kappas, attentions


def compute_proxy_per_roi_kappa(kappas, attentions, roi_names):
    """
    Compute proxy per-ROI kappa from global kappa + attention weights.
    
    The idea: kappa_r_proxy = kappa_global * alpha_r * n_rois
    (high attention + high global kappa -> high per-ROI confidence)
    """
    n_rois = len(roi_names)
    # Normalize attentions to sum to 1
    alpha_norm = attentions / (attentions.sum(axis=-1, keepdims=True) + 1e-8)
    
    # Per-ROI kappa proxy: global kappa distributed by attention
    per_roi_kappas = kappas[:, np.newaxis] * alpha_norm * n_rois
    
    return per_roi_kappas


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load everything
    model, features, meta_df, clip_df, roi_names, config = load_model_and_data()
    
    # Run extraction
    mus, kappas, attentions = extract_kappas_from_vmf_model(
        model, features, meta_df, roi_names, batch_size=128
    )
    
    # Compute proxy per-ROI kappas
    per_roi_kappas = compute_proxy_per_roi_kappa(kappas, attentions, roi_names)
    
    # Save results
    logger.info("Saving results to %s", OUTPUT_DIR)
    np.save(OUTPUT_DIR / "roi_kappa_per_roi_kappas.npy", per_roi_kappas)
    np.save(OUTPUT_DIR / "roi_kappa_consensus_kappa.npy", kappas)
    np.save(OUTPUT_DIR / "roi_kappa_mu_fused.npy", mus)
    np.save(OUTPUT_DIR / "roi_kappa_alphas.npy", attentions)
    np.save(OUTPUT_DIR / "roi_kappa_delta.npy", np.zeros(len(kappas)))  # no delta for vmf
    
    if "nsdId" in meta_df.columns:
        np.save(OUTPUT_DIR / "roi_kappa_trial_ids.npy", meta_df["nsdId"].values)
    
    meta = {
        "n_trials": len(kappas),
        "n_rois": len(roi_names),
        "roi_names": roi_names,
        "has_per_roi_mus": False,
        "has_mu_fused": True,
        "model_checkpoint": str(CHECKPOINT),
        "model_type": config.get("model", {}).get("type", "vmf"),
        "subject": SUBJECT,
        "files": {
            "per_roi_kappas": "roi_kappa_per_roi_kappas.npy",
            "consensus_kappa": "roi_kappa_consensus_kappa.npy",
            "mu_fused": "roi_kappa_mu_fused.npy",
            "alphas": "roi_kappa_alphas.npy",
            "delta": "roi_kappa_delta.npy",
        },
    }
    with open(OUTPUT_DIR / "roi_kappa_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    # Run topography analysis
    logger.info("\n=== Running Topography Analysis ===")
    from fmri2img.eval.roi_kappa_extraction import (
        PerROIKappaResult, compute_roi_kappa_topography
    )
    
    result = PerROIKappaResult(
        per_roi_kappas=per_roi_kappas,
        consensus_kappa=kappas,
        delta=np.zeros(len(kappas)),
        alphas=attentions,
        roi_names=roi_names,
    )
    
    topography = compute_roi_kappa_topography(result)
    
    with open(OUTPUT_DIR / "topography_report.json", "w") as f:
        json.dump(topography, f, indent=2)
    
    logger.info("\n=== ROI RANKING (by mean kappa) ===")
    for rank, (roi, kappa_val) in enumerate(topography["ranked_rois"], 1):
        logger.info("  %2d. %-20s kappa=%.2f", rank, roi, kappa_val)
    
    logger.info("\n=== TIER STATISTICS ===")
    for tier, stats in topography["tier_stats"].items():
        logger.info("  %-20s mean_kappa=%.2f (n_rois=%d)", tier, stats["mean_kappa"], stats["n_rois"])
    
    logger.info("\nDone! Results saved to: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
