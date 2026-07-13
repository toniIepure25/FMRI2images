"""
Run BrainBits Bottleneck Sweep on the H100.

Evaluates how retrieval performance degrades as we compress the embedding dimension,
computing the Neural Information Ratio (NIR) curve.
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
os.environ.setdefault("OUTPUT_ROOT", "/home/jovyan/work/FMRI2images/experimental_results")

DEVICE = "cuda"
SUBJECT = "subj01"
CHECKPOINT = Path("experimental_results/V66a_roi_pretrain/subj01/checkpoint_best.pt")
OUTPUT_DIR = Path("experimental_results/UDND_analysis/subj01/bottleneck_sweep")


def load_model_and_get_embeddings():
    """Load model, run inference, get predicted and GT embeddings."""
    from fmri2img.models.unified_model import create_model
    
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model_config = ckpt.get("model_config", config.get("model", {}))
    
    subject_roi_dims = model_config.get("encoder", {}).get("subject_roi_dims", {}).get(SUBJECT, {})
    if not subject_roi_dims:
        subject_roi_dims = model_config.get("encoder", {}).get("roi_dims", {})
    
    roi_names = list(subject_roi_dims.keys())
    
    # Build ROI indices from state_dict
    state_dict = ckpt["model_state_dict"]
    roi_indices = {}
    offset = 0
    for name, size in subject_roi_dims.items():
        roi_indices[name] = np.arange(offset, offset + size)
        offset += size
    
    # Extract buffer indices from state_dict
    idx_pattern = re.compile(r"encoder\._idx_(\w+)_(\d+)")
    subject_roi_indices_for_model = {}
    for key, tensor in state_dict.items():
        m = idx_pattern.match(key)
        if m:
            subj_id = m.group(1)
            roi_idx = int(m.group(2))
            if subj_id not in subject_roi_indices_for_model:
                subject_roi_indices_for_model[subj_id] = {}
            if roi_idx < len(roi_names):
                subject_roi_indices_for_model[subj_id][roi_names[roi_idx]] = tensor.numpy()
    
    if SUBJECT in subject_roi_indices_for_model:
        roi_indices = subject_roi_indices_for_model[SUBJECT]
    
    encoder_cfg = model_config.get("encoder", {})
    if "input_dim" not in encoder_cfg:
        encoder_cfg["input_dim"] = sum(subject_roi_dims.values())
        model_config["encoder"] = encoder_cfg
    
    model = create_model(model_config, roi_indices=roi_indices)
    model.load_state_dict(state_dict, strict=False)
    
    # Register buffers
    for key in state_dict:
        m = idx_pattern.match(key)
        if m:
            parts = key.split(".", 1)
            if len(parts) == 2:
                model.encoder.register_buffer(parts[1], state_dict[key])
    
    model = model.to(DEVICE).eval()
    
    # Load features
    features_path = Path(os.environ["CACHE_ROOT"]) / "preextracted" / f"subject={SUBJECT}" / "fmri_features.npy"
    features = np.load(features_path)
    
    # Load metadata and CLIP
    meta_path = Path(os.environ["CACHE_ROOT"]) / "preextracted" / f"subject={SUBJECT}" / "trial_meta.parquet"
    meta_df = pd.read_parquet(meta_path)
    
    clip_path = Path("outputs/clip_cache/clip.parquet")
    clip_df = pd.read_parquet(clip_path)
    
    # Get embeddings for all trials
    logger.info("Running inference on %d trials...", len(features))
    
    all_mus = []
    batch_size = 256
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            batch = torch.from_numpy(features[start:end]).float().to(DEVICE)
            subject_ids = torch.zeros(len(batch), dtype=torch.long, device=DEVICE)
            mu, kappa = model(batch, subject_ids=subject_ids)
            all_mus.append(mu.cpu())
    
    pred_embeddings = torch.cat(all_mus, dim=0)  # (N, 768)
    logger.info("Predicted embeddings: %s", pred_embeddings.shape)
    
    # Get GT CLIP embeddings
    # The clip cache has 'clip_embedding' column containing arrays
    if "clip_embedding" in clip_df.columns:
        gt_matrix = np.stack(clip_df["clip_embedding"].values).astype(np.float32)
    elif "embedding" in clip_df.columns:
        gt_matrix = np.stack(clip_df["embedding"].values).astype(np.float32)
    elif "fused" in clip_df.columns:
        gt_matrix = np.stack(clip_df["fused"].values).astype(np.float32)
    else:
        emb_cols = sorted([c for c in clip_df.columns if c.startswith("emb_")])
        gt_matrix = clip_df[emb_cols].values.astype(np.float32)
    
    logger.info("GT CLIP embeddings: %s", gt_matrix.shape)
    
    # Map trials to GT embeddings
    nsd_ids = meta_df["nsdId"].values
    clip_nsd_ids = clip_df["nsdId"].values if "nsdId" in clip_df.columns else clip_df.index.values
    
    # Create mapping
    id_to_idx = {nsd_id: idx for idx, nsd_id in enumerate(clip_nsd_ids)}
    
    # Use first 9000 trials for train, rest for val (simple split)
    n_val = 3000
    val_indices = np.arange(len(features) - n_val, len(features))
    
    val_pred = pred_embeddings[val_indices]
    val_nsd = nsd_ids[val_indices]
    val_gt_indices = [id_to_idx.get(nid, 0) for nid in val_nsd]
    val_gt = torch.from_numpy(gt_matrix[val_gt_indices])
    
    return val_pred, val_gt, pred_embeddings, gt_matrix, nsd_ids, id_to_idx


def compute_retrieval_at_rank(pred, gt_gallery, k=1):
    """Compute R@k: fraction where GT is in top-k retrieved."""
    pred_norm = F.normalize(pred, dim=-1)
    gallery_norm = F.normalize(gt_gallery, dim=-1)
    
    sims = pred_norm @ gallery_norm.T  # (N_query, N_gallery)
    _, topk_indices = sims.topk(k, dim=-1)
    
    # For each query, check if its GT index is in top-k
    gt_indices = torch.arange(len(pred)).unsqueeze(1)  # each query's GT is at same index
    hits = (topk_indices == gt_indices).any(dim=-1).float()
    return hits.mean().item()


def bottleneck_sweep():
    """Run the BrainBits bottleneck sweep."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    val_pred, val_gt, all_pred, gt_matrix, nsd_ids, id_to_idx = load_model_and_get_embeddings()
    
    # Compute SVD of predicted embeddings
    logger.info("Computing SVD of predictions (%s)...", val_pred.shape)
    pred_centered = val_pred - val_pred.mean(dim=0)
    U, S, Vh = torch.linalg.svd(pred_centered, full_matrices=False)
    
    logger.info("Top singular values: %s", S[:10].tolist())
    logger.info("Effective rank (90%% var): %d", 
                (S.cumsum(0) < 0.9 * S.sum()).sum().item() + 1)
    
    # Sweep bottleneck ranks
    ranks = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 640, 768]
    ranks = [r for r in ranks if r <= val_pred.shape[1]]
    
    results = {"ranks": [], "r_at_1": [], "r_at_5": [], "cosine_sim": [], "mse": []}
    
    # Full-rank baseline
    full_r1 = compute_retrieval_at_rank(val_pred, val_gt, k=1)
    full_r5 = compute_retrieval_at_rank(val_pred, val_gt, k=5)
    full_cos = F.cosine_similarity(val_pred, val_gt, dim=-1).mean().item()
    full_mse = F.mse_loss(val_pred, val_gt).item()
    
    logger.info("Full-rank baseline: R@1=%.4f, R@5=%.4f, cos=%.4f, MSE=%.4f",
                full_r1, full_r5, full_cos, full_mse)
    
    for rank in ranks:
        # Project through bottleneck: keep only top-k singular components
        pred_bottleneck = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
        pred_bottleneck = pred_bottleneck + val_pred.mean(dim=0)
        
        r1 = compute_retrieval_at_rank(pred_bottleneck, val_gt, k=1)
        r5 = compute_retrieval_at_rank(pred_bottleneck, val_gt, k=5)
        cos = F.cosine_similarity(pred_bottleneck, val_gt, dim=-1).mean().item()
        mse = F.mse_loss(pred_bottleneck, val_gt).item()
        
        results["ranks"].append(rank)
        results["r_at_1"].append(r1)
        results["r_at_5"].append(r5)
        results["cosine_sim"].append(cos)
        results["mse"].append(mse)
        
        nir = r1 / full_r1 if full_r1 > 0 else 0
        logger.info("  rank=%3d: R@1=%.4f (NIR=%.3f), R@5=%.4f, cos=%.4f", 
                    rank, r1, nir, r5, cos)
    
    # Compute NIR curve
    nir_curve = [r / full_r1 if full_r1 > 0 else 0 for r in results["r_at_1"]]
    results["nir_curve"] = nir_curve
    results["full_rank_baseline"] = {
        "r_at_1": full_r1, "r_at_5": full_r5,
        "cosine_sim": full_cos, "mse": full_mse,
    }
    
    # Find effective information dimension (rank where NIR >= 0.9)
    eid_90 = next((r for r, nir in zip(results["ranks"], nir_curve) if nir >= 0.9), ranks[-1])
    eid_95 = next((r for r, nir in zip(results["ranks"], nir_curve) if nir >= 0.95), ranks[-1])
    results["effective_info_dim_90"] = eid_90
    results["effective_info_dim_95"] = eid_95
    
    # Compute area under NIR curve (normalized)
    auc = np.trapz(nir_curve, x=[r / 768.0 for r in results["ranks"]])
    results["nir_auc"] = auc
    
    # Singular value spectrum
    results["singular_values"] = S.tolist()[:100]
    results["explained_variance_ratio"] = (S ** 2 / (S ** 2).sum()).tolist()[:100]
    
    logger.info("\n=== SUMMARY ===")
    logger.info("Full-rank R@1: %.4f", full_r1)
    logger.info("Effective Info Dim (90%%): %d", eid_90)
    logger.info("Effective Info Dim (95%%): %d", eid_95)
    logger.info("NIR AUC: %.4f", auc)
    logger.info("Interpretation: The model uses ~%d/%d dimensions for 90%% of performance",
                eid_90, 768)
    
    # Save
    with open(OUTPUT_DIR / "bottleneck_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    
    np.save(OUTPUT_DIR / "singular_values.npy", S.numpy())
    
    logger.info("\nDone! Results saved to: %s", OUTPUT_DIR)


if __name__ == "__main__":
    bottleneck_sweep()
