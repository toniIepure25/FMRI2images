"""
Shared 1000 evaluation helpers re-exported as a library module.

This module provides the core functions needed by eval_shared1000_full.py
and other evaluation scripts. The canonical implementations live in
scripts/evaluation/eval_comprehensive.py; this module re-exports them so
they are importable from the fmri2img package.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_nsd_shared_1000(stim_info_path: str) -> pd.DataFrame:
    """Load NSD Shared 1000 stimulus metadata from nsd_stim_info_merged.csv."""
    logger.info("Loading NSD stimulus info from %s", stim_info_path)
    df = pd.read_csv(stim_info_path)
    shared = df[df["shared1000"] == True].copy()
    logger.info("Found %d shared images", len(shared))
    return shared


def get_shared_1000_trials(
    shared_df: pd.DataFrame,
    subject: str,
    average_reps: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get trial indices and nsdIds for NSD Shared 1000."""
    subj_num = int(subject.replace("subj", ""))

    if average_reps:
        rep0 = shared_df[f"subject{subj_num}_rep0"].values
        rep1 = shared_df[f"subject{subj_num}_rep1"].values
        rep2 = shared_df[f"subject{subj_num}_rep2"].values
        trials = np.stack([rep0, rep1, rep2], axis=1)
    else:
        trials = shared_df[f"subject{subj_num}_rep0"].values

    nsd_ids = shared_df["nsdId"].values
    return trials, nsd_ids


def average_fmri_reps(
    fmri_data: np.ndarray,
    trial_indices: np.ndarray,
) -> np.ndarray:
    """Average fMRI across repetitions for higher SNR."""
    n_images, n_reps = trial_indices.shape
    n_voxels = fmri_data.shape[1]
    averaged = np.zeros((n_images, n_voxels), dtype=np.float32)
    for i in range(n_images):
        averaged[i] = fmri_data[trial_indices[i]].mean(axis=0)
    return averaged


def load_encoder(encoder_type: str, checkpoint_path: str, device: str):
    """Load encoder (ridge, mlp, two_stage, or unified) from checkpoint."""
    logger.info("Loading %s encoder from %s", encoder_type, checkpoint_path)

    if encoder_type == "ridge":
        import pickle
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)

    if encoder_type == "mlp":
        from fmri2img.models.mlp import load_mlp
        enc = load_mlp(checkpoint_path, device=device)
        enc.eval()
        return enc

    if encoder_type == "two_stage":
        from fmri2img.models.encoders import load_two_stage_encoder
        enc = load_two_stage_encoder(checkpoint_path, device=device)
        enc.eval()
        return enc

    if encoder_type == "unified":
        from fmri2img.models.unified_model import create_model
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_config = ckpt.get("model_config", ckpt.get("config", {}).get("model", {}))
        model = create_model(model_config).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    raise ValueError(f"Unknown encoder type: {encoder_type}")


def predict_clip_embeddings(
    encoder,
    encoder_type: str,
    fmri_features: np.ndarray,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Predict CLIP embeddings from fMRI features."""
    if encoder_type == "ridge":
        preds = encoder.predict(fmri_features)
        return preds / np.linalg.norm(preds, axis=1, keepdims=True)

    predictions = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, len(fmri_features), batch_size):
            batch = torch.from_numpy(fmri_features[i : i + batch_size]).float().to(device)
            out = encoder(batch)
            if isinstance(out, tuple):
                out = out[0]
            predictions.append(out.cpu().numpy())
    preds = np.concatenate(predictions, axis=0)
    norms = np.linalg.norm(preds, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return preds / norms


def compute_retrieval_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    gallery_sizes: Optional[list] = None,
) -> dict:
    """Compute retrieval metrics (R@K, MRR, median rank)."""
    if gallery_sizes is None:
        gallery_sizes = [2, 10, 50, 100, 500, 1000]

    from numpy.linalg import norm
    pred_norm = predictions / norm(predictions, axis=1, keepdims=True).clip(1e-8)
    tgt_norm = targets / norm(targets, axis=1, keepdims=True).clip(1e-8)
    sim = pred_norm @ tgt_norm.T
    N = len(sim)

    ranks = np.zeros(N, dtype=int)
    for i in range(N):
        order = np.argsort(-sim[i])
        ranks[i] = int(np.where(order == i)[0][0]) + 1

    results = {
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "mrr": float((1.0 / ranks).mean()),
    }
    for k in gallery_sizes:
        results[f"R@{k}"] = float((ranks <= k).mean())

    results["per_sample_ranks"] = ranks.tolist()
    return results


def compute_perceptual_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict:
    """Compute embedding-level perceptual metrics."""
    from numpy.linalg import norm
    pred_norm = predictions / norm(predictions, axis=1, keepdims=True).clip(1e-8)
    tgt_norm = targets / norm(targets, axis=1, keepdims=True).clip(1e-8)
    cos_sims = (pred_norm * tgt_norm).sum(axis=1)
    mse_vals = ((predictions - targets) ** 2).mean(axis=1)
    return {
        "cosine_sim_mean": float(cos_sims.mean()),
        "cosine_sim_std": float(cos_sims.std()),
        "mse_mean": float(mse_vals.mean()),
        "per_sample_cosine": cos_sims.tolist(),
    }
