"""
Calibrated Uncertainty Analysis for Neural Image Decoding.

Phase 1 of the A* paper: "Calibrated Uncertainty for Reliable Neural Decoding"

Strategy:
  - Use EXISTING precomputed shared1000_predictions.npy for retrieval metrics
  - Re-run inference WITH proper z-scoring to extract per-trial kappa
  - Aggregate kappa to image level and correlate with retrieval rank
  - Compute selective prediction, AURC, ECE, Brier
  - Multi-expert agreement analysis
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
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("NSD_DATA_ROOT", "/home/jovyan/work/data/nsd")
os.environ.setdefault("CACHE_ROOT", "/home/jovyan/work/FMRI2images/cache")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUBJECT = "subj01"
REPO_ROOT = Path("/home/jovyan/work/FMRI2images")
CACHE_ROOT = Path(os.environ["CACHE_ROOT"])
OUTPUT_DIR = REPO_ROOT / "experimental_results" / "calibrated_uncertainty"

EXPERTS = {
    "V62a": {
        "exp_dir": "V62a_cls_retrieval_768d",
        "checkpoint": REPO_ROOT / "experimental_results/V62a_cls_retrieval_768d/subj01/checkpoint_best.pt",
        "config": REPO_ROOT / "configs/experiments/V62a_cls_retrieval_768d.yaml",
        "embedding_dim": 768,
        "is_multi_subject": False,
        "zscore_prefix": "session",
    },
    "V66a": {
        "exp_dir": "V66a_roi_pretrain",
        "checkpoint": REPO_ROOT / "experimental_results/V66a_roi_pretrain/subj01/checkpoint_best.pt",
        "config": REPO_ROOT / "configs/experiments/V66a_roi_pretrain.yaml",
        "embedding_dim": 768,
        "is_multi_subject": True,
        "zscore_prefix": "subj01_session",
    },
    "V61a": {
        "exp_dir": "V61a_finetune_difflr",
        "checkpoint": REPO_ROOT / "experimental_results/V61a_finetune_difflr/subj01/checkpoint_best.pt",
        "config": REPO_ROOT / "configs/experiments/V61a_finetune_difflr.yaml",
        "embedding_dim": 768,
        "is_multi_subject": False,
        "zscore_prefix": "session",
        "is_token_model": True,
    },
}


def load_shared1000_trial_data():
    """Load SHARED1000 trial index and raw fMRI features."""
    index_path = REPO_ROOT / f"data/indices/nsd_index/subject={SUBJECT}/index.parquet"
    index_df = pd.read_parquet(index_path)
    
    s1000_df = index_df[index_df["shared1000"] == True].copy()
    s1000_df = s1000_df.sort_values(["nsdId", "session"]).reset_index(drop=False)
    s1000_df.rename(columns={"index": "original_index"}, inplace=True)
    
    unique_nsd_ids = np.sort(s1000_df["nsdId"].unique())
    n_images = len(unique_nsd_ids)
    n_trials = len(s1000_df)
    
    logger.info("SHARED1000: %d trials -> %d unique images", n_trials, n_images)
    
    features_path = CACHE_ROOT / f"preextracted/subject={SUBJECT}/fmri_features.npy"
    features = np.load(features_path, mmap_mode="r")
    logger.info("fMRI features: %s", features.shape)
    
    trial_indices = s1000_df["original_index"].values
    s1000_features = features[trial_indices].copy()
    
    sessions = s1000_df["session"].values
    nsd_ids = s1000_df["nsdId"].values
    
    return s1000_features, sessions, nsd_ids, unique_nsd_ids, n_trials, n_images


def apply_zscore(features, sessions, zscore_dir, prefix="session"):
    """Apply per-session z-scoring."""
    result = features.copy()
    unique_sessions = np.unique(sessions)
    
    for sess in unique_sessions:
        mean_path = zscore_dir / f"{prefix}_{sess}_mean.npy"
        std_path = zscore_dir / f"{prefix}_{sess}_std.npy"
        
        if not mean_path.exists() or not std_path.exists():
            logger.warning("Z-score stats missing for session %d, skipping", sess)
            continue
        
        mean = np.load(mean_path)
        std = np.load(std_path)
        std = np.where(std < 1e-6, 1.0, std)
        
        mask = sessions == sess
        result[mask] = (result[mask] - mean) / std
    
    return result


def load_precomputed_predictions(exp_dir):
    """Load existing precomputed predictions and ground truth."""
    metrics_dir = REPO_ROOT / "experimental_results" / exp_dir / "subj01" / "metrics"
    
    preds_path = metrics_dir / "shared1000_predictions.npy"
    gt_path = metrics_dir / "shared1000_ground_truth.npy"
    nsd_ids_path = metrics_dir / "shared1000_nsd_ids.npy"
    
    preds = np.load(preds_path)
    gt = np.load(gt_path)
    nsd_ids = np.load(nsd_ids_path) if nsd_ids_path.exists() else None
    
    logger.info("  Precomputed preds: %s, gt: %s", preds.shape, gt.shape)
    return preds, gt, nsd_ids


def load_model(expert_name, expert_info):
    """Load a model from checkpoint."""
    from fmri2img.models.unified_model import create_model
    
    ckpt_path = expert_info["checkpoint"]
    logger.info("Loading %s from %s", expert_name, ckpt_path)
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    model_config = ckpt.get("model_config", config.get("model", {}))
    state_dict = ckpt["model_state_dict"]
    
    if expert_info["is_multi_subject"]:
        subject_roi_dims = model_config.get("encoder", {}).get("subject_roi_dims", {}).get(SUBJECT, {})
        if not subject_roi_dims:
            subject_roi_dims = model_config.get("encoder", {}).get("roi_dims", {})
        roi_names = list(subject_roi_dims.keys())
        total_voxels = sum(subject_roi_dims.values())
        
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
    else:
        encoder_cfg = model_config.get("encoder", {})
        if "input_dim" not in encoder_cfg:
            for key in state_dict:
                if "encoder" in key and "weight" in key:
                    if state_dict[key].dim() == 2 and state_dict[key].shape[1] > 10000:
                        encoder_cfg["input_dim"] = state_dict[key].shape[1]
                        break
            if "input_dim" not in encoder_cfg:
                encoder_cfg["input_dim"] = 15724
            model_config["encoder"] = encoder_cfg
        model = create_model(model_config)
    
    model.load_state_dict(state_dict, strict=False)
    
    if expert_info["is_multi_subject"]:
        idx_pattern = re.compile(r"encoder\._idx_(\w+)_(\d+)")
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
    
    model_type = model_config.get("type", "unknown")
    subjects_list = config.get("data", {}).get("subjects", [SUBJECT])
    subject_idx = subjects_list.index(SUBJECT) if SUBJECT in subjects_list else 0
    
    logger.info("  Loaded: type=%s, encoder=%s, subject_idx=%d",
               model_type, type(model.encoder).__name__, subject_idx)
    
    return model, model_type, subject_idx


@torch.no_grad()
def extract_kappas(model, features, is_multi_subject, subject_idx=0, 
                   mc_samples=1, batch_size=64):
    """Run inference and extract per-trial kappa values only."""
    n_trials = len(features)
    
    all_kappas_runs = []
    
    for mc_run in range(mc_samples):
        if mc_samples > 1:
            model.train()
        else:
            model.eval()
        
        run_kappas = []
        
        for start in range(0, n_trials, batch_size):
            end = min(start + batch_size, n_trials)
            batch = torch.from_numpy(features[start:end]).float().to(DEVICE)
            
            kwargs = {}
            if is_multi_subject:
                kwargs["subject_ids"] = torch.full((len(batch),), subject_idx, 
                                                   dtype=torch.long, device=DEVICE)
            
            out = model(batch, **kwargs)
            
            if isinstance(out, tuple):
                _, aux = out
            else:
                aux = None
            
            if aux is not None:
                k = aux.squeeze(-1)
                if hasattr(model, 'decoder') and hasattr(model.decoder, 'kappa_mode'):
                    if model.decoder.kappa_mode == 'log':
                        k = k.exp()
                run_kappas.append(k.cpu().numpy())
            else:
                run_kappas.append(np.ones(end - start))
        
        all_kappas_runs.append(np.concatenate(run_kappas))
    
    model.eval()
    
    if mc_samples > 1:
        kappas = np.stack(all_kappas_runs).mean(axis=0)
        kappa_variance = np.stack(all_kappas_runs).var(axis=0)
        return kappas, kappa_variance
    else:
        return np.concatenate([all_kappas_runs[0]]), None


def aggregate_kappas_to_image(trial_kappas, nsd_ids, unique_nsd_ids):
    """Aggregate trial-level kappas to image level."""
    n_images = len(unique_nsd_ids)
    image_kappas = np.zeros(n_images)
    image_kappa_stds = np.zeros(n_images)
    image_kappa_mins = np.zeros(n_images)
    image_kappa_maxs = np.zeros(n_images)
    
    for i, nid in enumerate(unique_nsd_ids):
        mask = nsd_ids == nid
        trial_k = trial_kappas[mask]
        image_kappas[i] = trial_k.mean()
        image_kappa_stds[i] = trial_k.std() if len(trial_k) > 1 else 0
        image_kappa_mins[i] = trial_k.min()
        image_kappa_maxs[i] = trial_k.max()
    
    return image_kappas, image_kappa_stds, image_kappa_mins, image_kappa_maxs


def compute_retrieval_ranks(predictions, gallery, gt_indices=None):
    """Compute per-query retrieval ranks from precomputed predictions."""
    predictions_n = predictions / (np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8)
    gallery_n = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-8)
    
    sim = predictions_n @ gallery_n.T
    
    if gt_indices is None:
        gt_indices = np.arange(len(predictions))
    
    ranks = np.zeros(len(predictions), dtype=int)
    scores = np.zeros(len(predictions))
    
    for i in range(len(predictions)):
        sorted_idx = np.argsort(-sim[i])
        rank = np.where(sorted_idx == gt_indices[i])[0]
        ranks[i] = rank[0] if len(rank) > 0 else len(gallery)
        scores[i] = sim[i, gt_indices[i]]
    
    return ranks, scores


def compute_csls_ranks(predictions, gallery, k=3, gt_indices=None):
    """CSLS retrieval ranks."""
    predictions_n = predictions / (np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8)
    gallery_n = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-8)
    
    sim = predictions_n @ gallery_n.T
    
    knn_query = np.sort(sim, axis=1)[:, -k:].mean(axis=1)
    knn_gallery = np.sort(sim, axis=0)[-k:, :].mean(axis=0)
    csls_scores = 2 * sim - knn_query[:, None] - knn_gallery[None, :]
    
    if gt_indices is None:
        gt_indices = np.arange(len(predictions))
    
    ranks = np.zeros(len(predictions), dtype=int)
    
    for i in range(len(predictions)):
        sorted_idx = np.argsort(-csls_scores[i])
        rank = np.where(sorted_idx == gt_indices[i])[0]
        ranks[i] = rank[0] if len(rank) > 0 else len(gallery)
    
    return ranks


# =========================================================================
# SELECTIVE PREDICTION & CALIBRATION
# =========================================================================

def compute_selective_prediction_curve(kappas, ranks, coverages=None):
    """Sort by kappa (descending), compute R@1 at each coverage level."""
    if coverages is None:
        coverages = np.concatenate([
            np.arange(1.0, 0.05, -0.05),
            np.array([0.05, 0.02, 0.01])
        ])
    
    sorted_idx = np.argsort(-kappas)
    sorted_ranks = ranks[sorted_idx]
    sorted_kappas = kappas[sorted_idx]
    
    results = []
    for cov in coverages:
        n_keep = max(1, int(len(kappas) * cov))
        kept_ranks = sorted_ranks[:n_keep]
        
        r1 = (kept_ranks == 0).mean()
        r5 = (kept_ranks < 5).mean()
        r10 = (kept_ranks < 10).mean()
        mrr = (1.0 / (kept_ranks + 1)).mean()
        
        results.append({
            "coverage": round(float(cov), 3),
            "n_retained": n_keep,
            "r@1": round(float(r1), 4),
            "r@5": round(float(r5), 4),
            "r@10": round(float(r10), 4),
            "mrr": round(float(mrr), 4),
            "kappa_threshold": round(float(sorted_kappas[n_keep - 1]), 3),
            "mean_kappa_retained": round(float(sorted_kappas[:n_keep].mean()), 3),
        })
    
    return results


def compute_aurc(kappas, is_correct):
    """Area Under Risk-Coverage curve. Lower = better calibration."""
    sorted_idx = np.argsort(-kappas)
    sorted_correct = is_correct[sorted_idx]
    
    n = len(kappas)
    cumsum = np.cumsum(sorted_correct)
    coverages = np.arange(1, n + 1) / n
    accuracies = cumsum / np.arange(1, n + 1)
    risks = 1 - accuracies
    aurc = np.trapz(risks, coverages)
    
    optimal = np.sort(is_correct)[::-1]
    cumsum_opt = np.cumsum(optimal)
    acc_opt = cumsum_opt / np.arange(1, n + 1)
    optimal_aurc = np.trapz(1 - acc_opt, coverages)
    
    random_aurc = 1 - is_correct.mean()
    
    return {
        "aurc": round(float(aurc), 6),
        "optimal_aurc": round(float(optimal_aurc), 6),
        "excess_aurc": round(float(aurc - optimal_aurc), 6),
        "random_aurc": round(float(random_aurc), 6),
    }


def compute_ece(kappas, is_correct, n_bins=15):
    """Expected Calibration Error using normalized kappa as confidence."""
    kappa_min, kappa_max = kappas.min(), kappas.max()
    if kappa_max - kappa_min < 1e-8:
        return {"ece": 0.0, "bins": []}
    
    confidence = (kappas - kappa_min) / (kappa_max - kappa_min)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins = []
    
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (confidence >= bin_edges[i]) & (confidence < bin_edges[i + 1])
        else:
            mask = (confidence >= bin_edges[i]) & (confidence <= bin_edges[i + 1])
        
        if mask.sum() == 0:
            continue
        
        bin_conf = confidence[mask].mean()
        bin_acc = is_correct[mask].mean()
        bin_size = mask.sum()
        ece += (bin_size / len(kappas)) * abs(bin_acc - bin_conf)
        bins.append({
            "bin": i, "confidence": round(float(bin_conf), 4),
            "accuracy": round(float(bin_acc), 4), "count": int(bin_size),
            "gap": round(float(abs(bin_acc - bin_conf)), 4),
        })
    
    return {"ece": round(float(ece), 6), "bins": bins}


def compute_brier(kappas, is_correct):
    """Brier score using normalized kappa."""
    kappa_min, kappa_max = kappas.min(), kappas.max()
    if kappa_max - kappa_min < 1e-8:
        return 0.0
    confidence = (kappas - kappa_min) / (kappa_max - kappa_min)
    return round(float(((confidence - is_correct.astype(float)) ** 2).mean()), 6)


# =========================================================================
# MAIN ANALYSIS
# =========================================================================

def analyze_expert(expert_name, expert_info, trial_features_zscored, sessions, 
                   nsd_ids, unique_nsd_ids):
    """Complete calibration analysis for one expert."""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERT: %s", expert_name)
    logger.info("=" * 70)
    
    # Step 1: Load precomputed predictions and compute retrieval ranks
    preds, gt, saved_nsd_ids = load_precomputed_predictions(expert_info["exp_dir"])
    
    if saved_nsd_ids is None:
        saved_nsd_ids = unique_nsd_ids[:len(preds)]
    
    cosine_ranks, cosine_scores = compute_retrieval_ranks(preds, gt)
    csls_ranks = compute_csls_ranks(preds, gt, k=3)
    
    r1_cos = (cosine_ranks == 0).mean()
    r1_csls = (csls_ranks == 0).mean()
    r5_cos = (cosine_ranks < 5).mean()
    r5_csls = (csls_ranks < 5).mean()
    
    logger.info("From precomputed preds: R@1=%.4f (CSLS=%.4f), R@5=%.4f (CSLS=%.4f)",
               r1_cos, r1_csls, r5_cos, r5_csls)
    
    # Step 2: Load model and extract per-trial kappa with z-scored features
    model, model_type, subject_idx = load_model(expert_name, expert_info)
    
    mc_samples = 16 if expert_name == "V61a" else 1
    kappas_result = extract_kappas(
        model, trial_features_zscored,
        is_multi_subject=expert_info["is_multi_subject"],
        subject_idx=subject_idx,
        mc_samples=mc_samples,
        batch_size=64,
    )
    
    if mc_samples > 1:
        trial_kappas, trial_kappa_var = kappas_result
    else:
        trial_kappas, trial_kappa_var = kappas_result
    
    logger.info("Trial kappa stats: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
               trial_kappas.mean(), trial_kappas.std(), trial_kappas.min(), trial_kappas.max())
    
    # Step 3: Aggregate kappa to image level
    image_kappas, image_kappa_stds, image_kappa_mins, image_kappa_maxs = \
        aggregate_kappas_to_image(trial_kappas, nsd_ids, unique_nsd_ids)
    
    # Match kappas to precomputed prediction order (saved_nsd_ids)
    nid_to_kappa = dict(zip(unique_nsd_ids, image_kappas))
    nid_to_kappa_std = dict(zip(unique_nsd_ids, image_kappa_stds))
    
    matched_kappas = np.array([nid_to_kappa.get(nid, 0) for nid in saved_nsd_ids])
    matched_kappa_stds = np.array([nid_to_kappa_std.get(nid, 0) for nid in saved_nsd_ids])
    
    logger.info("Image kappa stats: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
               matched_kappas.mean(), matched_kappas.std(), matched_kappas.min(), matched_kappas.max())
    
    # Step 4: Selective prediction curves
    is_correct_cos = (cosine_ranks == 0).astype(float)
    is_correct_csls = (csls_ranks == 0).astype(float)
    
    sel_cos = compute_selective_prediction_curve(matched_kappas, cosine_ranks)
    sel_csls = compute_selective_prediction_curve(matched_kappas, csls_ranks)
    
    logger.info("\n--- Selective Prediction (Cosine) ---")
    for p in sel_cos:
        if p["coverage"] in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1]:
            logger.info("  Coverage %.0f%%: R@1=%.4f (n=%d, kappa>%.2f)",
                       p["coverage"] * 100, p["r@1"], p["n_retained"], p["kappa_threshold"])
    
    logger.info("\n--- Selective Prediction (CSLS) ---")
    for p in sel_csls:
        if p["coverage"] in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1]:
            logger.info("  Coverage %.0f%%: CSLS R@1=%.4f (n=%d, kappa>%.2f)",
                       p["coverage"] * 100, p["r@1"], p["n_retained"], p["kappa_threshold"])
    
    # Step 5: Calibration metrics
    aurc_cos = compute_aurc(matched_kappas, is_correct_cos)
    aurc_csls = compute_aurc(matched_kappas, is_correct_csls)
    ece_cos = compute_ece(matched_kappas, is_correct_cos)
    ece_csls = compute_ece(matched_kappas, is_correct_csls)
    brier_cos = compute_brier(matched_kappas, is_correct_cos)
    brier_csls = compute_brier(matched_kappas, is_correct_csls)
    
    # Kappa-rank correlations
    rank_corr = stats.spearmanr(matched_kappas, cosine_ranks)
    score_corr = stats.spearmanr(matched_kappas, cosine_scores)
    csls_rank_corr = stats.spearmanr(matched_kappas, csls_ranks)
    
    logger.info("\n--- Calibration Metrics ---")
    logger.info("  AURC (cos): %.4f, excess: %.4f", aurc_cos["aurc"], aurc_cos["excess_aurc"])
    logger.info("  AURC (CSLS): %.4f, excess: %.4f", aurc_csls["aurc"], aurc_csls["excess_aurc"])
    logger.info("  ECE (cos): %.4f, ECE (CSLS): %.4f", ece_cos["ece"], ece_csls["ece"])
    logger.info("  Brier (cos): %.4f, Brier (CSLS): %.4f", brier_cos, brier_csls)
    logger.info("  Kappa-Rank rho (cos): %.4f (p=%.2e)", rank_corr.statistic, rank_corr.pvalue)
    logger.info("  Kappa-Rank rho (CSLS): %.4f (p=%.2e)", csls_rank_corr.statistic, csls_rank_corr.pvalue)
    logger.info("  Kappa-Score rho: %.4f (p=%.2e)", score_corr.statistic, score_corr.pvalue)
    
    # Save per-trial kappas
    np.save(OUTPUT_DIR / f"{expert_name}_trial_kappas.npy", trial_kappas)
    np.save(OUTPUT_DIR / f"{expert_name}_image_kappas.npy", matched_kappas)
    
    del model
    torch.cuda.empty_cache()
    
    return {
        "expert": expert_name,
        "model_type": model_type,
        "mc_samples": mc_samples,
        "n_images": len(preds),
        "n_trials": len(trial_kappas),
        "retrieval_cosine": {
            "r@1": round(float(r1_cos), 4),
            "r@5": round(float(r5_cos), 4),
            "mrr": round(float((1/(cosine_ranks+1)).mean()), 4),
            "median_rank": round(float(np.median(cosine_ranks)), 1),
        },
        "retrieval_csls": {
            "r@1": round(float(r1_csls), 4),
            "r@5": round(float(r5_csls), 4),
            "mrr": round(float((1/(csls_ranks+1)).mean()), 4),
            "median_rank": round(float(np.median(csls_ranks)), 1),
        },
        "kappa_stats": {
            "trial_mean": round(float(trial_kappas.mean()), 3),
            "trial_std": round(float(trial_kappas.std()), 3),
            "trial_min": round(float(trial_kappas.min()), 3),
            "trial_max": round(float(trial_kappas.max()), 3),
            "image_mean": round(float(matched_kappas.mean()), 3),
            "image_std": round(float(matched_kappas.std()), 3),
            "image_min": round(float(matched_kappas.min()), 3),
            "image_max": round(float(matched_kappas.max()), 3),
        },
        "selective_prediction_cosine": sel_cos,
        "selective_prediction_csls": sel_csls,
        "aurc_cosine": aurc_cos,
        "aurc_csls": aurc_csls,
        "ece_cosine": ece_cos,
        "ece_csls": ece_csls,
        "brier_cosine": brier_cos,
        "brier_csls": brier_csls,
        "correlations": {
            "kappa_vs_cos_rank": {"rho": round(float(rank_corr.statistic), 4), "p": float(rank_corr.pvalue)},
            "kappa_vs_cos_score": {"rho": round(float(score_corr.statistic), 4), "p": float(score_corr.pvalue)},
            "kappa_vs_csls_rank": {"rho": round(float(csls_rank_corr.statistic), 4), "p": float(csls_rank_corr.pvalue)},
        },
        "per_query_kappas": matched_kappas.tolist(),
        "per_query_kappa_stds": matched_kappa_stds.tolist(),
        "per_query_ranks_cosine": cosine_ranks.tolist(),
        "per_query_ranks_csls": csls_ranks.tolist(),
        "per_query_nsd_ids": saved_nsd_ids.tolist(),
    }


def multi_expert_agreement(expert_results):
    """Analyze multi-expert agreement and its predictive value."""
    logger.info("\n" + "=" * 70)
    logger.info("MULTI-EXPERT AGREEMENT ANALYSIS")
    logger.info("=" * 70)
    
    names = list(expert_results.keys())
    n_experts = len(names)
    
    # Align all experts to same nsd_id order
    common_nsd_ids = None
    for name in names:
        ids = set(expert_results[name]["per_query_nsd_ids"])
        if common_nsd_ids is None:
            common_nsd_ids = ids
        else:
            common_nsd_ids = common_nsd_ids & ids
    
    common_nsd_ids = sorted(common_nsd_ids)
    n_common = len(common_nsd_ids)
    logger.info("Common images across all experts: %d", n_common)
    
    # Build aligned arrays
    aligned_kappas = {}
    aligned_ranks_cos = {}
    aligned_ranks_csls = {}
    
    for name in names:
        nid_list = expert_results[name]["per_query_nsd_ids"]
        nid_to_idx = {nid: i for i, nid in enumerate(nid_list)}
        
        kappas = np.array(expert_results[name]["per_query_kappas"])
        ranks_cos = np.array(expert_results[name]["per_query_ranks_cosine"])
        ranks_csls = np.array(expert_results[name]["per_query_ranks_csls"])
        
        idx = [nid_to_idx[nid] for nid in common_nsd_ids]
        aligned_kappas[name] = kappas[idx]
        aligned_ranks_cos[name] = ranks_cos[idx]
        aligned_ranks_csls[name] = ranks_csls[idx]
    
    # Correct/incorrect per expert
    correct_cos = {name: (ranks == 0) for name, ranks in aligned_ranks_cos.items()}
    correct_csls = {name: (ranks == 0) for name, ranks in aligned_ranks_csls.items()}
    
    n_correct_cos = sum(correct_cos[n] for n in names).astype(int)
    n_correct_csls = sum(correct_csls[n] for n in names).astype(int)
    
    # Agreement-stratified analysis
    agreement_cos = {}
    agreement_csls = {}
    
    logger.info("\n--- Agreement vs Accuracy (Cosine) ---")
    for n_agree in range(n_experts + 1):
        mask = n_correct_cos == n_agree
        if mask.sum() == 0:
            continue
        mean_kappas = {name: round(float(aligned_kappas[name][mask].mean()), 3) for name in names}
        agreement_cos[str(n_agree)] = {
            "n_queries": int(mask.sum()),
            "fraction": round(float(mask.mean()), 4),
            "mean_kappa_per_expert": mean_kappas,
            "overall_mean_kappa": round(float(np.mean([aligned_kappas[n][mask].mean() for n in names])), 3),
        }
        logger.info("  %d/%d correct: %d queries (%.1f%%), mean_kappa=%.2f",
                   n_agree, n_experts, mask.sum(), mask.mean()*100,
                   np.mean([aligned_kappas[n][mask].mean() for n in names]))
    
    logger.info("\n--- Agreement vs Accuracy (CSLS) ---")
    for n_agree in range(n_experts + 1):
        mask = n_correct_csls == n_agree
        if mask.sum() == 0:
            continue
        mean_kappas = {name: round(float(aligned_kappas[name][mask].mean()), 3) for name in names}
        agreement_csls[str(n_agree)] = {
            "n_queries": int(mask.sum()),
            "fraction": round(float(mask.mean()), 4),
            "mean_kappa_per_expert": mean_kappas,
            "overall_mean_kappa": round(float(np.mean([aligned_kappas[n][mask].mean() for n in names])), 3),
        }
        logger.info("  %d/%d correct: %d queries (%.1f%%), mean_kappa=%.2f",
                   n_agree, n_experts, mask.sum(), mask.mean()*100,
                   np.mean([aligned_kappas[n][mask].mean() for n in names]))
    
    # Pairwise agreement
    pairwise = {}
    logger.info("\n--- Pairwise Expert Agreement (CSLS) ---")
    for i in range(n_experts):
        for j in range(i+1, n_experts):
            ni, nj = names[i], names[j]
            agree = correct_csls[ni] == correct_csls[nj]
            both_correct = (correct_csls[ni] & correct_csls[nj]).mean()
            both_wrong = (~correct_csls[ni] & ~correct_csls[nj]).mean()
            key = f"{ni}_vs_{nj}"
            pairwise[key] = {
                "agreement": round(float(agree.mean()), 4),
                "both_correct": round(float(both_correct), 4),
                "both_wrong": round(float(both_wrong), 4),
                "only_first": round(float((correct_csls[ni] & ~correct_csls[nj]).mean()), 4),
                "only_second": round(float((~correct_csls[ni] & correct_csls[nj]).mean()), 4),
            }
            logger.info("  %s vs %s: agree=%.1f%%, both_correct=%.1f%%, both_wrong=%.1f%%",
                       ni, nj, agree.mean()*100, both_correct*100, both_wrong*100)
    
    # Selective prediction with different confidence signals
    best_expert = max(names, key=lambda n: expert_results[n]["retrieval_csls"]["r@1"])
    best_ranks = aligned_ranks_csls[best_expert]
    
    mean_kappa = np.mean([aligned_kappas[n] for n in names], axis=0)
    max_kappa = np.max([aligned_kappas[n] for n in names], axis=0)
    agreement_signal = n_correct_csls.astype(float) / n_experts
    
    # Normalize each signal to [0,1]
    def normalize(x):
        xmin, xmax = x.min(), x.max()
        return (x - xmin) / (xmax - xmin + 1e-8)
    
    combined = normalize(mean_kappa) + normalize(agreement_signal)
    
    signals = {
        f"single_best_kappa ({best_expert})": (aligned_kappas[best_expert], best_ranks),
        "mean_kappa_all_experts": (mean_kappa, best_ranks),
        "max_kappa_all_experts": (max_kappa, best_ranks),
        "agreement_score": (agreement_signal, best_ranks),
        "kappa_plus_agreement": (combined, best_ranks),
    }
    
    logger.info("\n--- Selective Prediction: Different Confidence Signals ---")
    signal_results = {}
    for sig_name, (sig, rnks) in signals.items():
        curve = compute_selective_prediction_curve(sig, rnks)
        signal_results[sig_name] = curve
        
        logger.info("\n  %s:", sig_name)
        for p in curve:
            if p["coverage"] in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1]:
                logger.info("    Coverage %.0f%%: R@1=%.4f (n=%d)",
                           p["coverage"]*100, p["r@1"], p["n_retained"])
    
    return {
        "n_common_images": n_common,
        "expert_names": names,
        "best_expert_csls": best_expert,
        "agreement_cosine": agreement_cos,
        "agreement_csls": agreement_csls,
        "pairwise_agreement": pairwise,
        "confidence_signal_curves": signal_results,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load trial data
    s1000_features, sessions, nsd_ids, unique_nsd_ids, n_trials, n_images = \
        load_shared1000_trial_data()
    
    # Process each expert
    expert_results = {}
    for name, info in EXPERTS.items():
        if not info["checkpoint"].exists():
            logger.warning("SKIPPING %s — checkpoint not found", name)
            continue
        
        # Apply z-scoring for this expert
        zscore_dir = REPO_ROOT / "experimental_results" / info["exp_dir"] / "subj01" / "zscore_stats"
        if zscore_dir.exists():
            zscored_features = apply_zscore(s1000_features, sessions, zscore_dir, info["zscore_prefix"])
            logger.info("Applied z-scoring from %s", zscore_dir)
        else:
            logger.warning("No zscore_stats for %s, using raw features", name)
            zscored_features = s1000_features
        
        result = analyze_expert(
            name, info, zscored_features, sessions, nsd_ids, unique_nsd_ids
        )
        expert_results[name] = result
        
        # Save per-expert results
        expert_out = {k: v for k, v in result.items() 
                     if k not in ["per_query_kappas", "per_query_kappa_stds",
                                  "per_query_ranks_cosine", "per_query_ranks_csls",
                                  "per_query_nsd_ids"]}
        with open(OUTPUT_DIR / f"{name}_calibration.json", "w") as f:
            json.dump(expert_out, f, indent=2)
        logger.info("Saved %s results", name)
    
    # Multi-expert agreement
    agreement = None
    if len(expert_results) >= 2:
        agreement = multi_expert_agreement(expert_results)
        with open(OUTPUT_DIR / "multi_expert_agreement.json", "w") as f:
            json.dump(agreement, f, indent=2)
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    
    summary_table = []
    for name, r in expert_results.items():
        row = {
            "expert": name,
            "R@1_cos": r["retrieval_cosine"]["r@1"],
            "R@1_csls": r["retrieval_csls"]["r@1"],
            "kappa_mean": r["kappa_stats"]["image_mean"],
            "kappa_std": r["kappa_stats"]["image_std"],
            "AURC_csls": r["aurc_csls"]["aurc"],
            "excess_AURC": r["aurc_csls"]["excess_aurc"],
            "ECE_csls": r["ece_csls"]["ece"],
            "Brier_csls": r["brier_csls"],
            "rho_kappa_rank": r["correlations"]["kappa_vs_csls_rank"]["rho"],
        }
        summary_table.append(row)
        
        logger.info("\n%s:", name)
        logger.info("  R@1: cos=%.4f, CSLS=%.4f", row["R@1_cos"], row["R@1_csls"])
        logger.info("  Kappa: mean=%.2f, std=%.2f", row["kappa_mean"], row["kappa_std"])
        logger.info("  AURC: %.4f (excess: %.4f)", row["AURC_csls"], row["excess_AURC"])
        logger.info("  ECE: %.4f, Brier: %.4f", row["ECE_csls"], row["Brier_csls"])
        logger.info("  Kappa-Rank rho: %.4f", row["rho_kappa_rank"])
        
        for p in r["selective_prediction_csls"]:
            if p["coverage"] == 0.8:
                logger.info("  Selective 80%% cov: CSLS R@1=%.4f", p["r@1"])
            if p["coverage"] == 0.5:
                logger.info("  Selective 50%% cov: CSLS R@1=%.4f", p["r@1"])
    
    # Save combined summary
    combined = {
        "summary_table": summary_table,
        "experts": {name: {k: v for k, v in r.items() 
                          if k not in ["per_query_kappas", "per_query_kappa_stds",
                                       "per_query_ranks_cosine", "per_query_ranks_csls",
                                       "per_query_nsd_ids"]}
                   for name, r in expert_results.items()},
        "agreement": agreement,
    }
    with open(OUTPUT_DIR / "calibration_summary.json", "w") as f:
        json.dump(combined, f, indent=2)
    
    logger.info("\nAll results saved to %s", OUTPUT_DIR)
    logger.info("Done!")


if __name__ == "__main__":
    main()
