"""
Extract SHARED1000 kappas for multi-subject conformal analysis.

Loads trained V62a checkpoints, runs inference on SHARED1000 trials,
aggregates to image-level, and saves to conformal_prediction/{subject}/.
"""
import sys
sys.path.insert(0, "/home/jovyan/work/FMRI2images/src")

import json
import logging
import os
from copy import deepcopy
from pathlib import Path

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
REPO = Path("/home/jovyan/work/FMRI2images")
CACHE = Path(os.environ["CACHE_ROOT"])


def extract_for_subject(subject):
    from fmri2img.models.unified_model import create_model

    exp_dir = REPO / f"experimental_results/V62a_multisubj_{subject}/{subject}"
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = exp_dir / "checkpoint_best.pt"
    if not ckpt_path.exists():
        for p in exp_dir.rglob("*.pt"):
            if "best" in p.name:
                ckpt_path = p
                break
    if not ckpt_path.exists():
        logger.error("No checkpoint for %s", subject)
        return

    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    feat_path = CACHE / f"preextracted/subject={subject}/fmri_features.npy"
    meta_path = CACHE / f"preextracted/subject={subject}/trial_meta.parquet"
    features = np.load(feat_path)
    meta = pd.read_parquet(meta_path)

    input_dim = features.shape[1]

    model_config = ckpt.get("model_config", None)
    if model_config is None:
        config = ckpt.get("config", {})
        model_config = config.get("model", None)
    if model_config is None:
        logger.error("No model config in checkpoint")
        return

    model_config = deepcopy(model_config)
    model_config["encoder"]["input_dim"] = input_dim
    model = create_model(model_config)

    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE).eval()

    # shared1000 nsd_ids from subj01 reference
    s1000_path = REPO / "experimental_results/V62a_cls_retrieval_768d/subj01/metrics/shared1000_nsd_ids.npy"
    if not s1000_path.exists():
        logger.info("Trying alternative path for shared1000 nsd_ids...")
        for p in (REPO / "experimental_results").rglob("shared1000_nsd_ids.npy"):
            s1000_path = p
            break

    shared_nsd_ids = np.load(s1000_path)
    shared_set = set(int(x) for x in shared_nsd_ids)

    meta["nsdId"] = meta["nsdId"].astype(int)
    shared_mask = meta["nsdId"].isin(shared_set)
    shared_meta = meta[shared_mask].copy()
    shared_features = features[shared_mask.values].copy()

    logger.info("%s: %d trials, %d unique images in shared1000",
                subject, len(shared_meta), shared_meta["nsdId"].nunique())

    if "session" in shared_meta.columns:
        from scipy.stats import zscore
        for sess in np.unique(shared_meta["session"].values):
            mask = shared_meta["session"].values == sess
            shared_features[mask] = zscore(shared_features[mask], axis=0)

    all_preds, all_kappas = [], []
    with torch.no_grad():
        for start in range(0, len(shared_features), 64):
            batch = torch.from_numpy(shared_features[start:start+64]).float().to(DEVICE)
            out = model(batch)
            if isinstance(out, tuple):
                mu, kappa = out
            else:
                mu, kappa = out, torch.ones(out.shape[0], device=DEVICE)
            mu = F.normalize(mu, dim=-1)
            if kappa.ndim > 1:
                kappa = kappa.squeeze(-1)
            all_preds.append(mu.cpu().numpy())
            all_kappas.append(kappa.cpu().numpy())

    trial_preds = np.concatenate(all_preds)
    trial_kappas = np.concatenate(all_kappas)
    trial_nsd_ids = shared_meta["nsdId"].values

    unique_ids = np.sort(shared_meta["nsdId"].unique())
    n_images = len(unique_ids)
    img_preds = np.zeros((n_images, trial_preds.shape[1]), dtype=np.float32)
    img_kappas = np.zeros(n_images, dtype=np.float32)

    for i, nid in enumerate(unique_ids):
        mask = trial_nsd_ids == nid
        img_preds[i] = trial_preds[mask].mean(axis=0)
        img_kappas[i] = trial_kappas[mask].mean()
    img_preds /= np.linalg.norm(img_preds, axis=1, keepdims=True)

    clip_df = pd.read_parquet(REPO / "outputs/clip_cache/clip.parquet")
    emb_col = next((c for c in ["fused", "final", "embedding", "clip_embedding"] if c in clip_df.columns), None)
    if emb_col is None:
        raise ValueError(f"No embedding column found in clip.parquet. Columns: {clip_df.columns.tolist()}")
    gt_df = clip_df[clip_df["nsdId"].isin(unique_ids)].sort_values("nsdId")
    gt_emb = np.stack(gt_df[emb_col].values).astype(np.float32)
    gt_emb /= np.linalg.norm(gt_emb, axis=1, keepdims=True)
    gt_nsd_ids = gt_df["nsdId"].values

    nsd_to_idx = {int(nid): i for i, nid in enumerate(gt_nsd_ids)}
    gt_indices = np.array([nsd_to_idx[int(nid)] for nid in unique_ids])

    sim = img_preds @ gt_emb.T
    r1 = float((np.argmax(sim, axis=1) == gt_indices).mean())

    logger.info("%s: R@1=%.3f, kappa=%.2f +/- %.2f, n=%d images",
                subject, r1, img_kappas.mean(), img_kappas.std(), n_images)

    out_dir = REPO / f"experimental_results/conformal_prediction/{subject}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "shared1000_predictions.npy", img_preds)
    np.save(out_dir / "shared1000_kappas.npy", img_kappas)
    np.save(out_dir / "shared1000_nsd_ids.npy", unique_ids)
    np.save(out_dir / "shared1000_ground_truth.npy", gt_emb)

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

    logger.info("Saved to %s", out_dir)
    return results


if __name__ == "__main__":
    for subj in ["subj02", "subj05", "subj07"]:
        try:
            extract_for_subject(subj)
        except Exception as e:
            logger.error("Failed for %s: %s", subj, e)
            import traceback
            traceback.print_exc()
    logger.info("Done!")
