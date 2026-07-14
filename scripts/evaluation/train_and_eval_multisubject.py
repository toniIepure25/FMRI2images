"""
Multi-Subject V62a Training + SHARED1000 Evaluation
=====================================================

Trains V62a-style vMF model for subj02/05/07 (subj01 already done),
then runs SHARED1000 evaluation and kappa extraction.

Since V60a pretrained weights won't work for different subject input dims,
we train from scratch using N1v28a config (fast baseline, no pretrained deps).
"""
import sys
sys.path.insert(0, "/home/jovyan/work/FMRI2images/src")

import json
import logging
import os
import time
import yaml
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("NSD_DATA_ROOT", "/home/jovyan/work/data/nsd")
os.environ.setdefault("CACHE_ROOT", "/home/jovyan/work/FMRI2images/cache")
os.environ.setdefault("DATASET_ROOT", "/home/jovyan/work/data")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ROOT = Path("/home/jovyan/work/FMRI2images")
CACHE_ROOT = Path(os.environ["CACHE_ROOT"])
OUTPUT_DIR = REPO_ROOT / "experimental_results"

SUBJECTS = ["subj02", "subj05", "subj07"]

BASE_CONFIG = {
    "experiment": {
        "name": "V62a_cls_multisubject",
        "description": "V62a-equivalent for multi-subject conformal analysis",
        "tags": ["vmf", "vmf_nce", "multisubject", "conformal"],
    },
    "data": {
        "roi": "nsdgeneral",
        "train_split": 0.90,
        "val_split": 0.10,
        "test_split": 0.0,
        "split_by_image": True,
        "exclude_shared1000": True,
        "normalize_fmri": True,
        "average_repetitions": False,
        "zscore_mode": "per_session",
        "embedding_column": "fused",
        "seed": 42,
    },
    "model": {
        "type": "vmf",
        "posterior": "vmf",
        "encoder": {
            "encoder_type": "mlp",
            "input_dim": None,
            "hidden_dims": [8192, 4096, 2048],
            "activation": "gelu",
            "dropout": 0.20,
            "use_residual": True,
        },
        "decoder": {
            "input_dim": 2048,
            "output_dim": 768,
            "hidden_dims": [2048],
            "activation": "gelu",
            "dropout": 0.15,
            "kappa_mode": "softplus",
        },
        "projection_head": {"enabled": False},
    },
    "preprocessing": {"enabled": False},
    "loss": {
        "vmf_nce": {"enabled": True, "weight": 1.0, "tau": 1.0,
                     "use_queue": True, "learnable_temperature": True},
        "regression_mse": {"enabled": True, "weight": 0.5},
        "kappa_reg": {"enabled": True, "lambda_kappa": 0.01},
        "softclip": {"enabled": True, "weight": 1.0, "tau": 0.05,
                     "vmf_mode": True, "use_queue": True, "symmetric": True},
    },
    "queue": {"enabled": True, "size": 8192},
    "training": {
        "batch_size": 128,
        "gradient_accumulation_steps": 4,
        "mixed_precision": True,
        "mixed_precision_dtype": "bf16",
        "num_epochs": 100,
        "optimizer": {
            "type": "adamw",
            "lr": 3.0e-4,
            "weight_decay": 0.05,
            "betas": [0.9, 0.999],
        },
        "warmup_epochs": 10,
        "min_lr": 1.0e-7,
        "gradient_clip": 1.0,
        "early_stop_patience": 30,
        "early_stop_min_delta": 0.001,
        "train_r1_interval": 5,
        "fmri_noise_std": 0.10,
        "voxel_dropout": 0.10,
        "ema": {"enabled": True, "decay": 0.999},
    },
    "evaluation": {
        "use_csls": True,
        "csls_k": 10,
        "eval_shared1000": True,
        "checkpoint_metric": "csls_r@1",
    },
    "inference": {"use_mean_for_retrieval": True},
}


def train_subject(subject):
    """Train V62a for a single subject."""
    exp_name = f"V62a_multisubj_{subject}"
    exp_dir = OUTPUT_DIR / exp_name / subject
    
    ckpt_file = exp_dir / "checkpoint_best.pt"
    if ckpt_file.exists():
        logger.info("Checkpoint exists for %s, skipping training", subject)
        return exp_dir
    
    config = deepcopy(BASE_CONFIG)
    config["experiment"]["name"] = exp_name
    config["data"]["subject"] = subject
    
    config_path = REPO_ROOT / f"configs/experiments/{exp_name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("Training %s with config: %s", subject, config_path)
    
    import subprocess
    result = subprocess.run(
        ["python3", "scripts/training/train_unified.py",
         "--config", str(config_path)],
        cwd=str(REPO_ROOT),
        capture_output=False,
    )
    
    if result.returncode != 0:
        logger.error("Training failed for %s (exit code %d)", subject, result.returncode)
    else:
        logger.info("Training complete for %s", subject)
    
    return exp_dir


def extract_kappas_and_predictions(subject, exp_dir):
    """Extract SHARED1000 predictions and kappas for a subject."""
    from fmri2img.models.unified_model import create_model
    from fmri2img.data.nsd_dataset import PreextractedNSDDataset
    
    ckpt_path = exp_dir / "checkpoint_best.pt"
    if not ckpt_path.exists():
        logger.error("No checkpoint for %s at %s", subject, ckpt_path)
        return None
    
    logger.info("Loading checkpoint for %s from %s", subject, ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = checkpoint.get("config", BASE_CONFIG)
    
    if isinstance(config, dict) and "model" in config:
        model_cfg = config
    else:
        model_cfg = BASE_CONFIG
    
    # Load pre-extracted features
    feat_path = CACHE_ROOT / f"preextracted/subject={subject}/fmri_features.npy"
    meta_path = CACHE_ROOT / f"preextracted/subject={subject}/trial_meta.parquet"
    
    features = np.load(feat_path)
    meta = pd.read_parquet(meta_path)
    
    input_dim = features.shape[1]
    model_cfg_copy = deepcopy(model_cfg)
    if "model" in model_cfg_copy:
        model_cfg_copy["model"]["encoder"]["input_dim"] = input_dim
    
    model = create_model(model_cfg_copy)
    
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if hasattr(model, "load_state_dict"):
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.warning("Partial load: %s", e)
    model = model.to(DEVICE).eval()
    
    # Load CLIP gallery
    clip_df = pd.read_parquet(REPO_ROOT / "outputs/clip_cache/clip.parquet")
    
    # Get SHARED1000 nsd_ids
    shared1000_path = REPO_ROOT / "experimental_results/V62a_cls_retrieval_768d/subj01/metrics/shared1000_nsd_ids.npy"
    shared1000_nsd_ids = np.load(shared1000_path)
    shared1000_set = set(int(x) for x in shared1000_nsd_ids)
    
    # Filter trials for shared1000
    meta["nsdId"] = meta["nsdId"].astype(int)
    shared_mask = meta["nsdId"].isin(shared1000_set)
    shared_meta = meta[shared_mask].copy()
    shared_features = features[shared_mask.values]
    
    logger.info("  %s: %d shared1000 trials from %d unique images",
               subject, len(shared_meta), shared_meta["nsdId"].nunique())
    
    # Z-score per session
    sessions = shared_meta["session"].values if "session" in shared_meta.columns else None
    if sessions is not None:
        from scipy.stats import zscore
        for sess in np.unique(sessions):
            mask = sessions == sess
            shared_features[mask] = zscore(shared_features[mask], axis=0)
    
    # Run inference
    all_preds = []
    all_kappas = []
    
    batch_size = 64
    with torch.no_grad():
        for start in range(0, len(shared_features), batch_size):
            end = min(start + batch_size, len(shared_features))
            batch = torch.from_numpy(shared_features[start:end]).float().to(DEVICE)
            
            out = model(batch)
            if isinstance(out, tuple):
                mu, kappa = out
            else:
                mu = out
                kappa = torch.ones(mu.shape[0], device=DEVICE)
            
            mu = F.normalize(mu, dim=-1)
            all_preds.append(mu.cpu().numpy())
            if kappa.ndim > 1:
                kappa = kappa.squeeze(-1)
            all_kappas.append(kappa.cpu().numpy())
    
    trial_preds = np.concatenate(all_preds)
    trial_kappas = np.concatenate(all_kappas)
    trial_nsd_ids = shared_meta["nsdId"].values
    
    # Aggregate to image level
    unique_ids = np.sort(shared_meta["nsdId"].unique())
    n_images = len(unique_ids)
    img_preds = np.zeros((n_images, trial_preds.shape[1]), dtype=np.float32)
    img_kappas = np.zeros(n_images, dtype=np.float32)
    
    for i, nid in enumerate(unique_ids):
        mask = trial_nsd_ids == nid
        img_preds[i] = trial_preds[mask].mean(axis=0)
        img_kappas[i] = trial_kappas[mask].mean()
    
    img_preds /= np.linalg.norm(img_preds, axis=1, keepdims=True)
    
    # Load GT embeddings
    emb_col = None
    for col in ["fused", "final", "embedding"]:
        if col in clip_df.columns:
            emb_col = col
            break
    
    gt_df = clip_df[clip_df["nsdId"].isin(unique_ids)].sort_values("nsdId")
    gt_embeddings = np.stack(gt_df[emb_col].values).astype(np.float32)
    gt_embeddings /= np.linalg.norm(gt_embeddings, axis=1, keepdims=True)
    gt_nsd_ids = gt_df["nsdId"].values
    
    # Compute retrieval
    nsd_to_idx = {int(nid): i for i, nid in enumerate(gt_nsd_ids)}
    gt_indices = np.array([nsd_to_idx[int(nid)] for nid in unique_ids])
    
    sim = img_preds @ gt_embeddings.T
    top1 = np.argmax(sim, axis=1)
    r1 = float((top1 == gt_indices).mean())
    
    logger.info("  %s: R@1=%.3f, kappa: mean=%.2f, std=%.2f",
               subject, r1, img_kappas.mean(), img_kappas.std())
    
    # Save results
    out_dir = REPO_ROOT / f"experimental_results/conformal_prediction/{subject}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "shared1000_predictions.npy", img_preds)
    np.save(out_dir / "shared1000_kappas.npy", img_kappas)
    np.save(out_dir / "shared1000_nsd_ids.npy", unique_ids)
    np.save(out_dir / "shared1000_ground_truth.npy", gt_embeddings)
    
    results = {
        "subject": subject,
        "n_images": n_images,
        "n_trials": len(shared_features),
        "r@1_cosine": r1,
        "kappa_mean": float(img_kappas.mean()),
        "kappa_std": float(img_kappas.std()),
        "kappa_min": float(img_kappas.min()),
        "kappa_max": float(img_kappas.max()),
    }
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    for subject in SUBJECTS:
        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING: %s", subject)
        logger.info("=" * 60)
        
        # Check if already evaluated
        result_file = REPO_ROOT / f"experimental_results/conformal_prediction/{subject}/metrics.json"
        if result_file.exists():
            logger.info("Results already exist for %s, skipping", subject)
            continue
        
        # Step 1: Train
        exp_dir = train_subject(subject)
        
        # Step 2: Extract predictions and kappas
        try:
            results = extract_kappas_and_predictions(subject, exp_dir)
            if results:
                logger.info("  %s done: R@1=%.3f", subject, results["r@1_cosine"])
        except Exception as e:
            logger.error("Failed to extract for %s: %s", subject, e)
            import traceback
            traceback.print_exc()
    
    logger.info("\n===== ALL SUBJECTS COMPLETE =====")


if __name__ == "__main__":
    main()
