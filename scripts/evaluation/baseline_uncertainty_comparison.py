"""
Uncertainty Baseline Comparison.

Compare vMF kappa calibration to:
  1. MC-Dropout entropy (run N forward passes, measure prediction variance)
  2. Max cosine similarity (score-based confidence — not model uncertainty)
  3. Temperature scaling (post-hoc calibration of cosine scores)
  4. Ensemble variance (disagreement across the 3 experts)

All baselines are evaluated on the same SHARED1000 task using the same
selective prediction and calibration metrics as the kappa analysis.
"""
import sys
sys.path.insert(0, "/home/jovyan/work/FMRI2images/src")

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import torch
from scipy import stats, optimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("NSD_DATA_ROOT", "/home/jovyan/work/data/nsd")
os.environ.setdefault("CACHE_ROOT", "/home/jovyan/work/FMRI2images/cache")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUBJECT = "subj01"
REPO_ROOT = Path("/home/jovyan/work/FMRI2images")
CACHE_ROOT = Path(os.environ["CACHE_ROOT"])
OUTPUT_DIR = REPO_ROOT / "experimental_results" / "calibrated_uncertainty"


# =========================================================================
# METRICS (copied from main analysis for self-containedness)
# =========================================================================

def compute_selective_curve(confidence, ranks, coverages=None):
    if coverages is None:
        coverages = np.concatenate([np.arange(1.0, 0.05, -0.05), [0.05, 0.02, 0.01]])
    
    sorted_idx = np.argsort(-confidence)
    sorted_ranks = ranks[sorted_idx]
    
    results = []
    for cov in coverages:
        n = max(1, int(len(confidence) * cov))
        kept = sorted_ranks[:n]
        results.append({
            "coverage": round(float(cov), 3),
            "n_retained": n,
            "r@1": round(float((kept == 0).mean()), 4),
            "r@5": round(float((kept < 5).mean()), 4),
        })
    return results


def compute_aurc(confidence, is_correct):
    sorted_idx = np.argsort(-confidence)
    sorted_correct = is_correct[sorted_idx]
    n = len(confidence)
    cumsum = np.cumsum(sorted_correct)
    coverages = np.arange(1, n+1) / n
    risks = 1 - cumsum / np.arange(1, n+1)
    aurc = np.trapz(risks, coverages)
    
    opt = np.sort(is_correct)[::-1]
    opt_risks = 1 - np.cumsum(opt) / np.arange(1, n+1)
    opt_aurc = np.trapz(opt_risks, coverages)
    
    return {
        "aurc": round(float(aurc), 6),
        "excess_aurc": round(float(aurc - opt_aurc), 6),
    }


def compute_ece(confidence_norm, is_correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (confidence_norm >= bins[i]) & (confidence_norm < bins[i+1])
        if i == n_bins - 1:
            mask = (confidence_norm >= bins[i]) & (confidence_norm <= bins[i+1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(confidence_norm)) * abs(is_correct[mask].mean() - confidence_norm[mask].mean())
    return round(float(ece), 6)


def normalize_01(x):
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-8:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


# =========================================================================
# BASELINE 1: MC-Dropout (prediction variance as uncertainty)
# =========================================================================

def run_mc_dropout_baseline(model, features, is_multi_subject, subject_idx,
                            n_samples_list=[2, 4, 8, 16, 32]):
    """Run multiple MC-Dropout passes and compute prediction variance."""
    logger.info("Running MC-Dropout baseline...")
    n_trials = len(features)
    
    results = {}
    for n_mc in n_samples_list:
        model.train()
        all_preds = []
        
        for mc_run in range(n_mc):
            run_preds = []
            for start in range(0, n_trials, 64):
                end = min(start + 64, n_trials)
                batch = torch.from_numpy(features[start:end]).float().to(DEVICE)
                kwargs = {}
                if is_multi_subject:
                    kwargs["subject_ids"] = torch.full((len(batch),), subject_idx,
                                                      dtype=torch.long, device=DEVICE)
                
                with torch.no_grad():
                    out = model(batch, **kwargs)
                    pred = out[0] if isinstance(out, tuple) else out
                    
                    if pred.shape[1] == 197376:
                        pred = pred.reshape(pred.shape[0], 257, 768)[:, 0, :]
                    
                    pred = torch.nn.functional.normalize(pred, dim=-1)
                    run_preds.append(pred.cpu().numpy())
            
            all_preds.append(np.concatenate(run_preds))
        
        model.eval()
        stacked = np.stack(all_preds)
        mean_pred = stacked.mean(axis=0)
        pred_var = stacked.var(axis=0).mean(axis=1)
        
        entropy = -np.log(np.clip(1 / (pred_var + 1e-8), 1e-8, None))
        
        confidence = -pred_var
        
        results[n_mc] = {
            "confidence": confidence,
            "mean_pred": mean_pred,
            "pred_variance": pred_var,
        }
        
        logger.info("  MC-%d: var mean=%.6f, std=%.6f", n_mc, pred_var.mean(), pred_var.std())
    
    return results


# =========================================================================
# BASELINE 2: Max Cosine Similarity (score-based)
# =========================================================================

def max_cosine_baseline(predictions, gallery):
    """Use maximum cosine similarity as confidence (no model uncertainty)."""
    preds_n = predictions / (np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8)
    gal_n = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-8)
    
    sim = preds_n @ gal_n.T
    max_sim = sim.max(axis=1)
    margin = np.sort(sim, axis=1)[:, -1] - np.sort(sim, axis=1)[:, -2]
    
    return {
        "max_cosine": max_sim,
        "margin": margin,
    }


# =========================================================================
# BASELINE 3: Temperature Scaling
# =========================================================================

def temperature_scaling(predictions, gallery, gt_indices, val_fraction=0.3):
    """Post-hoc temperature scaling of cosine similarities."""
    preds_n = predictions / (np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8)
    gal_n = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-8)
    sim = preds_n @ gal_n.T
    
    n = len(predictions)
    n_val = int(n * val_fraction)
    val_idx = np.random.choice(n, n_val, replace=False)
    test_idx = np.array([i for i in range(n) if i not in val_idx])
    
    val_sim = sim[val_idx]
    val_gt = gt_indices[val_idx]
    
    def nll(T):
        T = max(T, 0.01)
        scaled = val_sim / T
        log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True) + 1e-8)
        nll_val = -np.mean([log_probs[i, val_gt[i]] for i in range(len(val_gt))])
        return nll_val
    
    result = optimize.minimize_scalar(nll, bounds=(0.01, 20.0), method='bounded')
    T_opt = result.x
    
    scaled_sim = sim / T_opt
    max_scaled = scaled_sim.max(axis=1)
    
    logger.info("  Temperature scaling: T_opt=%.4f", T_opt)
    
    return {
        "confidence": max_scaled,
        "temperature": float(T_opt),
        "test_idx": test_idx,
    }


# =========================================================================
# BASELINE 4: Ensemble Variance
# =========================================================================

def ensemble_variance_baseline(expert_preds_dict):
    """Use prediction variance across experts as uncertainty."""
    preds_list = []
    for name, preds in expert_preds_dict.items():
        preds_n = preds / (np.linalg.norm(preds, axis=1, keepdims=True) + 1e-8)
        preds_list.append(preds_n)
    
    stacked = np.stack(preds_list)
    pred_var = stacked.var(axis=0).mean(axis=1)
    
    mean_pairwise_cos = np.zeros(preds_list[0].shape[0])
    count = 0
    for i in range(len(preds_list)):
        for j in range(i+1, len(preds_list)):
            cos = (preds_list[i] * preds_list[j]).sum(axis=1)
            mean_pairwise_cos += cos
            count += 1
    mean_pairwise_cos /= count
    
    return {
        "variance_confidence": -pred_var,
        "agreement_confidence": mean_pairwise_cos,
    }


# =========================================================================
# MAIN
# =========================================================================

def load_model_and_data(expert_name, exp_dir, is_multi_subject, zscore_prefix):
    """Load model, apply z-scoring, return features and model."""
    import pandas as pd
    from fmri2img.models.unified_model import create_model
    
    index_path = REPO_ROOT / f"data/indices/nsd_index/subject={SUBJECT}/index.parquet"
    index_df = pd.read_parquet(index_path)
    s1000_df = index_df[index_df["shared1000"] == True].sort_values(["nsdId", "session"]).reset_index(drop=False)
    s1000_df.rename(columns={"index": "original_index"}, inplace=True)
    
    features = np.load(CACHE_ROOT / f"preextracted/subject={SUBJECT}/fmri_features.npy", mmap_mode="r")
    trial_indices = s1000_df["original_index"].values
    s1000_features = features[trial_indices].copy()
    sessions = s1000_df["session"].values
    nsd_ids = s1000_df["nsdId"].values
    unique_nsd_ids = np.sort(s1000_df["nsdId"].unique())
    
    zscore_dir = REPO_ROOT / "experimental_results" / exp_dir / "subj01" / "zscore_stats"
    if zscore_dir.exists():
        for sess in np.unique(sessions):
            mean_path = zscore_dir / f"{zscore_prefix}_{sess}_mean.npy"
            std_path = zscore_dir / f"{zscore_prefix}_{sess}_std.npy"
            if mean_path.exists() and std_path.exists():
                mean = np.load(mean_path)
                std = np.load(std_path)
                std = np.where(std < 1e-6, 1.0, std)
                mask = sessions == sess
                s1000_features[mask] = (s1000_features[mask] - mean) / std
    
    ckpt_path = REPO_ROOT / "experimental_results" / exp_dir / "subj01" / "checkpoint_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    model_config = ckpt.get("model_config", config.get("model", {}))
    state_dict = ckpt["model_state_dict"]
    
    if is_multi_subject:
        roi_dims = model_config.get("encoder", {}).get("subject_roi_dims", {}).get(SUBJECT, {})
        if not roi_dims:
            roi_dims = model_config.get("encoder", {}).get("roi_dims", {})
        roi_names = list(roi_dims.keys())
        
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
        enc = model_config.get("encoder", {})
        enc["input_dim"] = sum(roi_dims.values())
        model_config["encoder"] = enc
        model = create_model(model_config, roi_indices=roi_indices)
    else:
        enc = model_config.get("encoder", {})
        if "input_dim" not in enc:
            for key in state_dict:
                if "encoder" in key and "weight" in key and state_dict[key].dim() == 2 and state_dict[key].shape[1] > 10000:
                    enc["input_dim"] = state_dict[key].shape[1]
                    break
            if "input_dim" not in enc:
                enc["input_dim"] = 15724
            model_config["encoder"] = enc
        model = create_model(model_config)
    
    model.load_state_dict(state_dict, strict=False)
    
    if is_multi_subject:
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
    
    subjects_list = config.get("data", {}).get("subjects", [SUBJECT])
    subject_idx = subjects_list.index(SUBJECT) if SUBJECT in subjects_list else 0
    
    return model, s1000_features, nsd_ids, unique_nsd_ids, subject_idx


def aggregate_trials_to_images(trial_values, nsd_ids, unique_nsd_ids, agg="mean"):
    """Aggregate trial-level values to image level."""
    n = len(unique_nsd_ids)
    result = np.zeros(n)
    for i, nid in enumerate(unique_nsd_ids):
        mask = nsd_ids == nid
        vals = trial_values[mask]
        if agg == "mean":
            result[i] = vals.mean()
        elif agg == "min":
            result[i] = vals.min()
        elif agg == "max":
            result[i] = vals.max()
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    
    # Use V62a as the primary model (768-D, most standard architecture)
    logger.info("Loading V62a model and data...")
    model_v62, features_v62, nsd_ids, unique_nsd_ids, subj_idx = \
        load_model_and_data("V62a", "V62a_cls_retrieval_768d", False, "session")
    
    # Load precomputed predictions and gallery
    preds_v62 = np.load(REPO_ROOT / "experimental_results/V62a_cls_retrieval_768d/subj01/metrics/shared1000_predictions.npy")
    gt_v62 = np.load(REPO_ROOT / "experimental_results/V62a_cls_retrieval_768d/subj01/metrics/shared1000_ground_truth.npy")
    nsd_ids_v62 = np.load(REPO_ROOT / "experimental_results/V62a_cls_retrieval_768d/subj01/metrics/shared1000_nsd_ids.npy")
    
    # Also load precomputed kappas
    kappas_v62 = np.load(OUTPUT_DIR / "V62a_image_kappas.npy")
    kappas_v61 = np.load(OUTPUT_DIR / "V61a_image_kappas.npy")
    
    # Compute retrieval ranks (from precomputed predictions)
    preds_n = preds_v62 / (np.linalg.norm(preds_v62, axis=1, keepdims=True) + 1e-8)
    gt_n = gt_v62 / (np.linalg.norm(gt_v62, axis=1, keepdims=True) + 1e-8)
    sim = preds_n @ gt_n.T
    ranks = np.array([np.where(np.argsort(-sim[i]) == i)[0][0] for i in range(len(preds_v62))])
    
    # CSLS ranks
    knn_q = np.sort(sim, axis=1)[:, -3:].mean(axis=1)
    knn_g = np.sort(sim, axis=0)[-3:, :].mean(axis=0)
    csls = 2 * sim - knn_q[:, None] - knn_g[None, :]
    csls_ranks = np.array([np.where(np.argsort(-csls[i]) == i)[0][0] for i in range(len(preds_v62))])
    
    is_correct = (csls_ranks == 0).astype(float)
    
    logger.info("V62a baseline: CSLS R@1 = %.4f", is_correct.mean())
    
    # =====================================================================
    # BASELINE 1: MC-Dropout
    # =====================================================================
    logger.info("\n=== BASELINE 1: MC-Dropout ===")
    mc_results = run_mc_dropout_baseline(model_v62, features_v62, False, subj_idx,
                                         n_samples_list=[4, 8, 16, 32])
    
    mc_baselines = {}
    for n_mc, mc_data in mc_results.items():
        trial_conf = mc_data["confidence"]
        image_conf = aggregate_trials_to_images(trial_conf, nsd_ids, unique_nsd_ids)
        
        nid_to_idx = {nid: i for i, nid in enumerate(unique_nsd_ids)}
        aligned_conf = np.array([image_conf[nid_to_idx[nid]] for nid in nsd_ids_v62])
        
        curve = compute_selective_curve(aligned_conf, csls_ranks)
        aurc = compute_aurc(aligned_conf, is_correct)
        ece = compute_ece(normalize_01(aligned_conf), is_correct)
        rho, p = stats.spearmanr(aligned_conf, csls_ranks)
        
        mc_baselines[f"mc_dropout_{n_mc}"] = {
            "selective_curve": curve,
            "aurc": aurc,
            "ece": ece,
            "kappa_rank_spearman": {"rho": round(float(rho), 4), "p": float(p)},
        }
        
        r1_50 = next((p["r@1"] for p in curve if p["coverage"] == 0.5), None)
        logger.info("  MC-%d: AURC=%.4f, excess=%.4f, rho=%.4f (p=%.2e), R@1@50%%=%.4f",
                    n_mc, aurc["aurc"], aurc["excess_aurc"], rho, p, r1_50 or 0)
    
    del model_v62
    torch.cuda.empty_cache()
    
    # =====================================================================
    # BASELINE 2: Max Cosine Similarity & Margin
    # =====================================================================
    logger.info("\n=== BASELINE 2: Max Cosine & Margin ===")
    cos_baseline = max_cosine_baseline(preds_v62, gt_v62)
    
    score_baselines = {}
    for name, conf in [("max_cosine", cos_baseline["max_cosine"]),
                       ("margin", cos_baseline["margin"])]:
        curve = compute_selective_curve(conf, csls_ranks)
        aurc = compute_aurc(conf, is_correct)
        ece = compute_ece(normalize_01(conf), is_correct)
        rho, p = stats.spearmanr(conf, csls_ranks)
        
        score_baselines[name] = {
            "selective_curve": curve,
            "aurc": aurc,
            "ece": ece,
            "rank_spearman": {"rho": round(float(rho), 4), "p": float(p)},
        }
        
        r1_50 = next((p_["r@1"] for p_ in curve if p_["coverage"] == 0.5), None)
        logger.info("  %s: AURC=%.4f, excess=%.4f, rho=%.4f (p=%.2e), R@1@50%%=%.4f",
                    name, aurc["aurc"], aurc["excess_aurc"], rho, p, r1_50 or 0)
    
    # =====================================================================
    # BASELINE 3: Temperature Scaling
    # =====================================================================
    logger.info("\n=== BASELINE 3: Temperature Scaling ===")
    gt_indices = np.arange(len(preds_v62))
    temp_result = temperature_scaling(preds_v62, gt_v62, gt_indices)
    
    curve = compute_selective_curve(temp_result["confidence"], csls_ranks)
    aurc = compute_aurc(temp_result["confidence"], is_correct)
    ece = compute_ece(normalize_01(temp_result["confidence"]), is_correct)
    rho, p = stats.spearmanr(temp_result["confidence"], csls_ranks)
    
    temp_baseline = {
        "temperature": temp_result["temperature"],
        "selective_curve": curve,
        "aurc": aurc,
        "ece": ece,
        "rank_spearman": {"rho": round(float(rho), 4), "p": float(p)},
    }
    
    r1_50 = next((p_["r@1"] for p_ in curve if p_["coverage"] == 0.5), None)
    logger.info("  T=%.4f: AURC=%.4f, excess=%.4f, rho=%.4f (p=%.2e), R@1@50%%=%.4f",
               temp_result["temperature"], aurc["aurc"], aurc["excess_aurc"], rho, p, r1_50 or 0)
    
    # =====================================================================
    # vMF KAPPA (our method)
    # =====================================================================
    logger.info("\n=== OUR METHOD: vMF Kappa ===")
    
    kappa_curve = compute_selective_curve(kappas_v62, csls_ranks)
    kappa_aurc = compute_aurc(kappas_v62, is_correct)
    kappa_ece = compute_ece(normalize_01(kappas_v62), is_correct)
    kappa_rho, kappa_p = stats.spearmanr(kappas_v62, csls_ranks)
    
    r1_50 = next((p_["r@1"] for p_ in kappa_curve if p_["coverage"] == 0.5), None)
    logger.info("  vMF kappa (V62a): AURC=%.4f, excess=%.4f, rho=%.4f (p=%.2e), R@1@50%%=%.4f",
               kappa_aurc["aurc"], kappa_aurc["excess_aurc"], kappa_rho, kappa_p, r1_50 or 0)
    
    # =====================================================================
    # SUMMARY TABLE
    # =====================================================================
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE COMPARISON SUMMARY (V62a, CSLS, SHARED1000)")
    logger.info("=" * 80)
    logger.info("%-25s %10s %12s %10s %10s %10s", 
               "Method", "AURC↓", "Excess AURC↓", "ECE↓", "ρ(conf,rank)", "R@1@50%↑")
    logger.info("-" * 80)
    
    all_methods = {}
    
    # vMF kappa
    all_methods["vmf_kappa_V62a"] = {
        "aurc": kappa_aurc, "ece": kappa_ece,
        "selective_curve": kappa_curve,
        "rank_spearman": {"rho": round(float(kappa_rho), 4), "p": float(kappa_p)},
    }
    
    for name, data in [("vmf_kappa_V62a", all_methods["vmf_kappa_V62a"])]:
        r1 = next((p_["r@1"] for p_ in data["selective_curve"] if p_["coverage"] == 0.5), 0)
        logger.info("%-25s %10.4f %12.4f %10.4f %10.4f %10.4f",
                   name, data["aurc"]["aurc"], data["aurc"]["excess_aurc"],
                   data["ece"], data["rank_spearman"]["rho"], r1)
    
    for name, data in mc_baselines.items():
        r1 = next((p_["r@1"] for p_ in data["selective_curve"] if p_["coverage"] == 0.5), 0)
        logger.info("%-25s %10.4f %12.4f %10.4f %10.4f %10.4f",
                   name, data["aurc"]["aurc"], data["aurc"]["excess_aurc"],
                   data["ece"], data["kappa_rank_spearman"]["rho"], r1)
        all_methods[name] = data
    
    for name, data in score_baselines.items():
        r1 = next((p_["r@1"] for p_ in data["selective_curve"] if p_["coverage"] == 0.5), 0)
        logger.info("%-25s %10.4f %12.4f %10.4f %10.4f %10.4f",
                   name, data["aurc"]["aurc"], data["aurc"]["excess_aurc"],
                   data["ece"], data["rank_spearman"]["rho"], r1)
        all_methods[name] = data
    
    r1_temp = next((p_["r@1"] for p_ in temp_baseline["selective_curve"] if p_["coverage"] == 0.5), 0)
    logger.info("%-25s %10.4f %12.4f %10.4f %10.4f %10.4f",
               "temp_scaling", temp_baseline["aurc"]["aurc"], temp_baseline["aurc"]["excess_aurc"],
               temp_baseline["ece"], temp_baseline["rank_spearman"]["rho"], r1_temp)
    all_methods["temp_scaling"] = temp_baseline
    
    # Save
    with open(OUTPUT_DIR / "baseline_comparison.json", "w") as f:
        json.dump(all_methods, f, indent=2, default=str)
    
    logger.info("\nSaved to %s", OUTPUT_DIR / "baseline_comparison.json")
    logger.info("Done!")


if __name__ == "__main__":
    main()
