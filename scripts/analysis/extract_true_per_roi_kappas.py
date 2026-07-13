"""
Extract TRUE per-ROI kappa by projecting each ROI token through the decoder.

The MultiSubjectROITransformer produces 16 ROI tokens + 1 CLS token.
By passing each ROI token independently through the VonMisesFisherDecoder,
we get genuine per-ROI (mu_r, kappa_r) estimates — not attention-weighted proxies.

This is the key methodological advance: each ROI token captures region-specific
information, and its kappa reflects per-region decoding confidence.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("NSD_DATA_ROOT", "/home/jovyan/work/data/nsd")
os.environ.setdefault("CACHE_ROOT", "/home/jovyan/work/FMRI2images/cache")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUBJECT = os.environ.get("SUBJECT", "subj01")
CHECKPOINT = Path(f"experimental_results/V66a_roi_pretrain/{SUBJECT}/checkpoint_best.pt")
OUTPUT_DIR = Path(f"experimental_results/UDND_analysis/{SUBJECT}/roi_kappa_results")


def load_model():
    """Load the V66a ROI Transformer model."""
    from fmri2img.models.unified_model import create_model
    
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model_config = ckpt.get("model_config", config.get("model", {}))
    
    subject_roi_dims = model_config.get("encoder", {}).get("subject_roi_dims", {}).get(SUBJECT, {})
    if not subject_roi_dims:
        subject_roi_dims = model_config.get("encoder", {}).get("roi_dims", {})
    
    roi_names = list(subject_roi_dims.keys())
    total_voxels = sum(subject_roi_dims.values())
    
    # Extract ROI indices from state_dict
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
    
    # Register index buffers
    for key in state_dict:
        m = idx_pattern.match(key)
        if m:
            parts = key.split(".", 1)
            if len(parts) == 2:
                model.encoder.register_buffer(parts[1], state_dict[key])
    
    model = model.to(DEVICE).eval()
    logger.info("Model loaded: %s encoder, %s decoder", 
               type(model.encoder).__name__, type(model.decoder).__name__)
    
    return model, roi_names


@torch.no_grad()
def extract_per_roi_kappas_from_tokens(model, features, batch_size=128):
    """
    Extract per-ROI kappa by passing each ROI token through the decoder.
    
    The encoder with return_roi_tokens=True gives:
    - cls_out: (B, d_model) — CLS token
    - roi_tokens: (B, n_rois, d_model) — per-ROI representations
    - cls_to_roi_alpha: (B, n_rois) — attention weights
    
    We pass each ROI token through the decoder to get (mu_r, kappa_r).
    """
    n_trials = len(features)
    
    all_roi_kappas = []
    all_roi_mus = []
    all_cls_kappa = []
    all_cls_mu = []
    all_alphas = []
    
    logger.info("Extracting per-ROI kappas for %d trials (batch=%d)...", n_trials, batch_size)
    t0 = time.time()
    
    for start in range(0, n_trials, batch_size):
        end = min(start + batch_size, n_trials)
        batch = torch.from_numpy(features[start:end]).float().to(DEVICE)
        subject_ids = torch.zeros(len(batch), dtype=torch.long, device=DEVICE)
        
        # Get ROI tokens from encoder
        encoder_out = model.encoder(batch, subject_ids, return_roi_tokens=True)
        
        cls_out = encoder_out.cls_out         # (B, 768)
        roi_tokens = encoder_out.roi_tokens   # (B, n_rois, 768)
        alpha = encoder_out.cls_to_roi_alpha  # (B, n_rois)
        
        # Decode CLS token (standard)
        cls_mu, cls_kappa = model.decoder(cls_out)
        all_cls_mu.append(cls_mu.cpu().numpy())
        all_cls_kappa.append(cls_kappa.cpu().numpy())
        all_alphas.append(alpha.cpu().numpy())
        
        # Decode each ROI token independently
        B, n_rois, d = roi_tokens.shape
        roi_tokens_flat = roi_tokens.reshape(B * n_rois, d)  # (B*n_rois, 768)
        
        roi_mu_flat, roi_kappa_flat = model.decoder(roi_tokens_flat)
        
        roi_mu = roi_mu_flat.reshape(B, n_rois, -1)      # (B, n_rois, 768)
        roi_kappa = roi_kappa_flat.reshape(B, n_rois)     # (B, n_rois)
        
        all_roi_kappas.append(roi_kappa.cpu().numpy())
        all_roi_mus.append(roi_mu.cpu().numpy())
        
        if (start // batch_size) % 50 == 0:
            elapsed = time.time() - t0
            logger.info("  %d/%d (%.1f%%) - %.1fs", end, n_trials, 100 * end / n_trials, elapsed)
    
    roi_kappas = np.concatenate(all_roi_kappas, axis=0)    # (N, n_rois)
    roi_mus = np.concatenate(all_roi_mus, axis=0)          # (N, n_rois, 768)
    cls_kappa = np.concatenate(all_cls_kappa, axis=0).squeeze(-1)  # (N,)
    cls_mu = np.concatenate(all_cls_mu, axis=0)            # (N, 768)
    alphas = np.concatenate(all_alphas, axis=0)            # (N, n_rois)
    
    elapsed = time.time() - t0
    logger.info("Done in %.1fs", elapsed)
    logger.info("Per-ROI kappa shape: %s", roi_kappas.shape)
    logger.info("Per-ROI kappa stats: mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
               roi_kappas.mean(), roi_kappas.std(), roi_kappas.min(), roi_kappas.max())
    logger.info("Per-ROI kappa range across ROIs: %.2f",
               roi_kappas.mean(axis=0).max() - roi_kappas.mean(axis=0).min())
    logger.info("CLS kappa: mean=%.2f, std=%.2f", cls_kappa.mean(), cls_kappa.std())
    
    return roi_kappas, roi_mus, cls_kappa, cls_mu, alphas


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model, roi_names = load_model()
    
    # Load features
    features_path = Path(os.environ["CACHE_ROOT"]) / "preextracted" / f"subject={SUBJECT}" / "fmri_features.npy"
    features = np.load(features_path)
    logger.info("Features: %s", features.shape)
    
    # Load trial metadata
    meta_path = Path(os.environ["CACHE_ROOT"]) / "preextracted" / f"subject={SUBJECT}" / "trial_meta.parquet"
    meta_df = pd.read_parquet(meta_path)
    
    # Extract per-ROI kappas
    roi_kappas, roi_mus, cls_kappa, cls_mu, alphas = extract_per_roi_kappas_from_tokens(
        model, features, batch_size=128
    )
    
    # Save results
    np.save(OUTPUT_DIR / "roi_kappa_per_roi_kappas.npy", roi_kappas)
    np.save(OUTPUT_DIR / "roi_kappa_consensus_kappa.npy", cls_kappa)
    np.save(OUTPUT_DIR / "roi_kappa_mu_fused.npy", cls_mu)
    np.save(OUTPUT_DIR / "roi_kappa_alphas.npy", alphas)
    np.save(OUTPUT_DIR / "roi_kappa_delta.npy", np.zeros(len(cls_kappa)))
    
    if "nsdId" in meta_df.columns:
        np.save(OUTPUT_DIR / "roi_kappa_trial_ids.npy", meta_df["nsdId"].values)
    
    # Per-ROI statistics
    logger.info("\n=== PER-ROI KAPPA RANKING ===")
    roi_means = roi_kappas.mean(axis=0)
    roi_stds = roi_kappas.std(axis=0)
    ranking = sorted(zip(roi_names, roi_means, roi_stds), key=lambda x: x[1], reverse=True)
    
    for rank, (roi, mean_k, std_k) in enumerate(ranking, 1):
        logger.info("  %2d. %-20s kappa=%.2f +/- %.2f", rank, roi, mean_k, std_k)
    
    # Save metadata
    meta = {
        "n_trials": len(roi_kappas),
        "n_rois": len(roi_names),
        "roi_names": roi_names,
        "has_per_roi_mus": True,
        "has_mu_fused": True,
        "model_checkpoint": str(CHECKPOINT),
        "model_type": "vmf",
        "extraction_method": "per_roi_token_decoding",
        "subject": SUBJECT,
        "roi_kappa_means": {roi: float(m) for roi, m in zip(roi_names, roi_means)},
        "roi_kappa_stds": {roi: float(s) for roi, s in zip(roi_names, roi_stds)},
    }
    with open(OUTPUT_DIR / "roi_kappa_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    logger.info("\nDone! Saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
