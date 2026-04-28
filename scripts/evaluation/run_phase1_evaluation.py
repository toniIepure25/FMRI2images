#!/usr/bin/env python3
"""
Phase 1 Evaluation: V55b + V55a on val/shared1000, fusion with N1v28a.

Self-contained: loads data the same way train_unified.py does, no fragile imports.
"""
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("phase1_eval")


# ---------------------------------------------------------------------------
# Data loading (mirrors train_unified.py _evaluate_shared1000)
# ---------------------------------------------------------------------------

def load_shared1000_data(subject: str, zscore_stats_dir: Optional[Path] = None):
    """Load shared1000 trials: features, nsdIds, CLIP embeddings."""
    cache_root = os.environ.get("CACHE_ROOT", "cache")
    feat_path = Path(cache_root) / "preextracted" / f"subject={subject}" / "fmri_features.npy"
    index_path = Path("data/indices/nsd_index") / f"subject={subject}" / "index.parquet"

    if not feat_path.exists():
        raise FileNotFoundError(f"Features not found: {feat_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    index_df = pd.read_parquet(index_path)
    features = np.load(feat_path, mmap_mode="r")

    s1000_mask = index_df["shared1000"].fillna(False).astype(bool).values
    s1000_features = np.array(features[s1000_mask], dtype=np.float32)
    s1000_df = index_df[s1000_mask].reset_index(drop=True)
    log.info("Shared1000: %d trials", len(s1000_df))

    if zscore_stats_dir is not None:
        zdir = Path(zscore_stats_dir)
        if "session" in s1000_df.columns:
            fb_mean_path = zdir / "global_fallback_mean.npy"
            fb_std_path = zdir / "global_fallback_std.npy"
            fb_mean = np.load(fb_mean_path) if fb_mean_path.exists() else None
            fb_std = np.load(fb_std_path) if fb_std_path.exists() else None
            sessions = s1000_df["session"].values
            for sess in np.unique(sessions):
                m_path = zdir / f"{subject}_session_{int(sess)}_mean.npy"
                s_path = zdir / f"{subject}_session_{int(sess)}_std.npy"
                if not m_path.exists():
                    m_path = zdir / f"session_{int(sess)}_mean.npy"
                    s_path = zdir / f"session_{int(sess)}_std.npy"
                sess_mask = sessions == sess
                if m_path.exists() and s_path.exists():
                    s_mean = np.load(m_path)
                    s_std = np.load(s_path)
                elif fb_mean is not None and fb_std is not None:
                    s_mean, s_std = fb_mean, fb_std
                else:
                    continue
                s1000_features[sess_mask] = (
                    (s1000_features[sess_mask] - s_mean) / (s_std + 1e-8)
                ).astype(np.float32)
            log.info("Applied per-session z-scoring")
        else:
            g_mean = zdir / "global_fallback_mean.npy"
            g_std = zdir / "global_fallback_std.npy"
            if g_mean.exists():
                s1000_features = (
                    (s1000_features - np.load(g_mean)) / (np.load(g_std) + 1e-8)
                ).astype(np.float32)
                log.info("Applied global z-scoring")

    clip_path = Path(os.environ.get("CLIP_CACHE", "outputs/clip_cache/clip.parquet"))
    clip_df = pd.read_parquet(clip_path)

    emb_col = None
    for col in ["fused", "embedding", "final", "clip_embedding"]:
        if col in clip_df.columns:
            emb_col = col
            break
    if emb_col is None:
        emb_cols = [c for c in clip_df.columns if c.startswith("emb_")]
        if emb_cols:
            emb_col = emb_cols[0]
    if emb_col is None:
        raise ValueError(f"No embedding column in clip cache: {clip_df.columns.tolist()}")

    nsd_to_emb = {}
    for _, row in clip_df.iterrows():
        nsd_id = int(row["nsdId"])
        emb = np.array(row[emb_col], dtype=np.float32)
        nsd_to_emb[nsd_id] = emb

    return s1000_features, s1000_df, nsd_to_emb


def load_val_data(subject: str, split_json: Path, zscore_stats_dir: Optional[Path] = None):
    """Load validation trials from a saved split file."""
    cache_root = os.environ.get("CACHE_ROOT", "cache")
    feat_path = Path(cache_root) / "preextracted" / f"subject={subject}" / "fmri_features.npy"
    index_path = Path("data/indices/nsd_index") / f"subject={subject}" / "index.parquet"

    index_df = pd.read_parquet(index_path)
    features = np.load(feat_path, mmap_mode="r")

    with open(split_json) as f:
        split_data = json.load(f)
    val_nsd_ids = set(split_data.get("val_nsd_ids", []))
    if not val_nsd_ids:
        raise ValueError(f"No val_nsd_ids in {split_json}")

    val_mask = index_df["nsdId"].isin(val_nsd_ids).values
    val_features = np.array(features[val_mask], dtype=np.float32)
    val_df = index_df[val_mask].reset_index(drop=True)
    log.info("Val: %d trials (%d unique images)", len(val_df), len(val_nsd_ids))

    if zscore_stats_dir is not None:
        zdir = Path(zscore_stats_dir)
        if "session" in val_df.columns:
            fb_mean = np.load(zdir / "global_fallback_mean.npy") if (zdir / "global_fallback_mean.npy").exists() else None
            fb_std = np.load(zdir / "global_fallback_std.npy") if (zdir / "global_fallback_std.npy").exists() else None
            sessions = val_df["session"].values
            for sess in np.unique(sessions):
                m_path = zdir / f"{subject}_session_{int(sess)}_mean.npy"
                s_path = zdir / f"{subject}_session_{int(sess)}_std.npy"
                if not m_path.exists():
                    m_path = zdir / f"session_{int(sess)}_mean.npy"
                    s_path = zdir / f"session_{int(sess)}_std.npy"
                sess_mask = sessions == sess
                if m_path.exists() and s_path.exists():
                    val_features[sess_mask] = (
                        (val_features[sess_mask] - np.load(m_path)) / (np.load(s_path) + 1e-8)
                    ).astype(np.float32)
                elif fb_mean is not None and fb_std is not None:
                    val_features[sess_mask] = (
                        (val_features[sess_mask] - fb_mean) / (fb_std + 1e-8)
                    ).astype(np.float32)

    clip_path = Path(os.environ.get("CLIP_CACHE", "outputs/clip_cache/clip.parquet"))
    clip_df = pd.read_parquet(clip_path)
    emb_col = None
    for col in ["fused", "embedding", "final", "clip_embedding"]:
        if col in clip_df.columns:
            emb_col = col
            break
    if emb_col is None:
        emb_cols = [c for c in clip_df.columns if c.startswith("emb_")]
        emb_col = emb_cols[0] if emb_cols else None
    nsd_to_emb = {}
    for _, row in clip_df.iterrows():
        nsd_to_emb[int(row["nsdId"])] = np.array(row[emb_col], dtype=np.float32)

    return val_features, val_df, nsd_to_emb


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

def load_model(experiment_dir: Path, subject: str):
    """Load vMF model from experiment directory."""
    from fmri2img.models.unified_model import create_model

    config_path = experiment_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})

    cache_root = os.environ.get("CACHE_ROOT", "cache")
    feat_path = Path(cache_root) / "preextracted" / f"subject={subject}" / "fmri_features.npy"
    if feat_path.exists():
        n_voxels = np.load(feat_path, mmap_mode="r").shape[1]
        model_cfg.setdefault("encoder", {})["input_dim"] = n_voxels

    dec_cfg = model_cfg.get("decoder", {})
    if dec_cfg.get("output_dim") is None:
        num_tokens = dec_cfg.get("num_tokens", 1)
        token_dim = dec_cfg.get("token_dim", 768)
        if dec_cfg.get("regression_head", False) and num_tokens > 1:
            dec_cfg["output_dim"] = num_tokens * token_dim
        else:
            dec_cfg["output_dim"] = token_dim

    model = create_model(model_cfg)
    ckpt_path = experiment_dir / "checkpoint_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    epoch = ckpt.get("epoch", "?")
    log.info("Loaded %s (epoch %s) on %s", ckpt_path, epoch, device)
    return model, config, device


def run_inference_on_features(model, features_np, device, batch_size=128):
    """Run model on raw numpy features, return (mu, kappa)."""
    n = len(features_np)
    all_preds, all_kappas = [], []

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            fmri = torch.from_numpy(features_np[start:end]).to(device, dtype=torch.float32)
            pred, aux = model(fmri)
            pred_np = pred.cpu().float().numpy()
            pred_np /= np.linalg.norm(pred_np, axis=-1, keepdims=True) + 1e-8
            all_preds.append(pred_np)
            if aux is not None:
                all_kappas.append(aux.squeeze(-1).cpu().float().numpy())

    predictions = np.concatenate(all_preds, axis=0)
    kappas = np.concatenate(all_kappas, axis=0) if all_kappas else None
    return predictions, kappas


def aggregate_to_images(preds, kappas, nsd_ids, gt_embeddings_map):
    """Average trial predictions per image, kappa-weighted."""
    unique_ids = np.unique(nsd_ids)
    valid_ids = [nid for nid in unique_ids if nid in gt_embeddings_map]
    n = len(valid_ids)
    d = preds.shape[1]

    img_preds = np.zeros((n, d), dtype=np.float32)
    img_gts = np.zeros((n, gt_embeddings_map[valid_ids[0]].shape[0]), dtype=np.float32)
    img_kappas = np.zeros(n, dtype=np.float32)
    img_nsd_ids = np.array(valid_ids)

    for i, nid in enumerate(valid_ids):
        mask = nsd_ids == nid
        trial_preds = preds[mask]
        if kappas is not None:
            k = kappas[mask]
            w = k / (k.sum() + 1e-8)
            img_preds[i] = (trial_preds * w[:, None]).sum(axis=0)
            img_kappas[i] = k.mean()
        else:
            img_preds[i] = trial_preds.mean(axis=0)
        img_gts[i] = gt_embeddings_map[nid]

    norms = np.linalg.norm(img_preds, axis=-1, keepdims=True) + 1e-8
    img_preds /= norms
    norms = np.linalg.norm(img_gts, axis=-1, keepdims=True) + 1e-8
    img_gts /= norms

    log.info("Aggregated: %d trials -> %d images (from %d unique nsdIds)", len(preds), n, len(unique_ids))
    return img_preds, img_gts, img_kappas, img_nsd_ids


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_retrieval(predictions, ground_truth, kappas=None, csls_k=10):
    """Compute comprehensive retrieval metrics including PPR variants."""
    sim = predictions @ ground_truth.T

    def ranks_from_scores(S):
        diag = np.diag(S)
        return (S >= diag[:, None]).sum(axis=1) - 1

    def csls(S, k):
        row_means = np.mean(np.sort(S, axis=1)[:, -k:], axis=1)
        col_means = np.mean(np.sort(S, axis=0)[-k:, :], axis=0)
        return 2 * S - row_means[:, None] - col_means[None, :]

    raw_ranks = ranks_from_scores(sim)
    csls_scores = csls(sim, csls_k)
    csls_ranks = ranks_from_scores(csls_scores)

    m = {
        "raw_r@1": float((raw_ranks == 0).mean()),
        "raw_r@5": float((raw_ranks < 5).mean()),
        "raw_r@10": float((raw_ranks < 10).mean()),
        "raw_mrr": float((1.0 / (raw_ranks + 1)).mean()),
        "csls_r@1": float((csls_ranks == 0).mean()),
        "csls_r@5": float((csls_ranks < 5).mean()),
        "csls_r@10": float((csls_ranks < 10).mean()),
        "csls_mrr": float((1.0 / (csls_ranks + 1)).mean()),
        "n_images": len(predictions),
    }

    if kappas is not None and len(kappas) == len(predictions):
        ppr_scores = kappas[:, None] * sim
        ppr_ranks = ranks_from_scores(ppr_scores)
        m["ppr_kappa_r@1"] = float((ppr_ranks == 0).mean())
        m["ppr_kappa_r@5"] = float((ppr_ranks < 5).mean())

        ppr_csls_scores = csls(ppr_scores, csls_k)
        ppr_csls_ranks = ranks_from_scores(ppr_csls_scores)
        m["ppr_csls_r@1"] = float((ppr_csls_ranks == 0).mean())
        m["ppr_csls_r@5"] = float((ppr_csls_ranks < 5).mean())

        m["kappa_mean"] = float(kappas.mean())
        m["kappa_std"] = float(kappas.std())
        m["kappa_min"] = float(kappas.min())
        m["kappa_max"] = float(kappas.max())

    return m


def compute_fusion_sweep(
    compact_preds, compact_gts, compact_kappas,
    legacy_preds, legacy_gts,
    csls_k=10,
):
    """Sweep alpha/gamma fusion weights."""
    def csls(S, k):
        row_means = np.mean(np.sort(S, axis=1)[:, -k:], axis=1)
        col_means = np.mean(np.sort(S, axis=0)[-k:, :], axis=0)
        return 2 * S - row_means[:, None] - col_means[None, :]

    def zscore(M):
        mu = M.mean(axis=1, keepdims=True)
        std = M.std(axis=1, keepdims=True) + 1e-8
        return (M - mu) / std

    def ranks_from_scores(S):
        diag = np.diag(S)
        return (S >= diag[:, None]).sum(axis=1) - 1

    compact_sim = compact_preds @ compact_gts.T
    legacy_sim = legacy_preds @ legacy_gts.T

    compact_csls = csls(compact_sim, csls_k)
    legacy_csls = csls(legacy_sim, csls_k)

    compact_z = zscore(compact_csls)
    legacy_z = zscore(legacy_csls)

    results = []
    for alpha in np.arange(0.05, 0.96, 0.05):
        gamma = 1.0 - alpha
        fused = alpha * compact_z + gamma * legacy_z
        r = ranks_from_scores(fused)
        results.append({
            "alpha": round(float(alpha), 2),
            "gamma": round(float(gamma), 2),
            "score_type": "csls_zscore",
            "r@1": float((r == 0).mean()),
            "r@5": float((r < 5).mean()),
            "r@10": float((r < 10).mean()),
        })

    if compact_kappas is not None:
        ppr_scores = compact_kappas[:, None] * compact_sim
        ppr_csls = csls(ppr_scores, csls_k)
        ppr_z = zscore(ppr_csls)

        for alpha in np.arange(0.05, 0.96, 0.05):
            gamma = 1.0 - alpha
            fused = alpha * ppr_z + gamma * legacy_z
            r = ranks_from_scores(fused)
            results.append({
                "alpha": round(float(alpha), 2),
                "gamma": round(float(gamma), 2),
                "score_type": "ppr_csls_zscore",
                "r@1": float((r == 0).mean()),
                "r@5": float((r < 5).mean()),
                "r@10": float((r < 10).mean()),
            })

    results.sort(key=lambda x: x["r@1"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Evaluate one experiment
# ---------------------------------------------------------------------------

def evaluate_experiment(experiment_dir: Path, subject: str, name: str):
    """Run val + shared1000 evaluation for one experiment."""
    model, config, device = load_model(experiment_dir, subject)
    data_cfg = config.get("data", {})
    metrics_dir = experiment_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    zscore_dir = experiment_dir / "zscore_stats"
    if not zscore_dir.exists():
        zscore_dir = None

    all_metrics = {}

    for split in ["shared1000", "val"]:
        log.info("\n=== %s: %s ===", name, split)
        try:
            if split == "shared1000":
                features, df, nsd_to_emb = load_shared1000_data(
                    subject, zscore_stats_dir=zscore_dir,
                )
            elif split == "val":
                split_json = experiment_dir / "split.json"
                if not split_json.exists():
                    log.warning("No split.json found for %s — skipping val", name)
                    continue
                features, df, nsd_to_emb = load_val_data(
                    subject, split_json, zscore_stats_dir=zscore_dir,
                )
            else:
                continue
        except Exception as e:
            log.error("Failed to load %s data for %s: %s", split, name, e)
            continue

        preds, kappas = run_inference_on_features(model, features, device)
        nsd_ids = df["nsdId"].values

        img_preds, img_gts, img_kappas, img_nsd_ids = aggregate_to_images(
            preds, kappas, nsd_ids, nsd_to_emb,
        )

        np.save(metrics_dir / f"{split}_predictions.npy", img_preds)
        np.save(metrics_dir / f"{split}_ground_truth.npy", img_gts)
        np.save(metrics_dir / f"{split}_nsd_ids.npy", img_nsd_ids)
        if kappas is not None:
            np.save(metrics_dir / f"{split}_kappas.npy", img_kappas)
        log.info("Saved predictions to %s", metrics_dir)

        m = compute_retrieval(img_preds, img_gts, img_kappas)
        all_metrics[split] = m
        log.info(
            "%s %s: raw_R@1=%.3f  csls_R@1=%.3f  ppr_csls_R@1=%.3f  (n=%d)",
            name, split, m["raw_r@1"], m["csls_r@1"], m.get("ppr_csls_r@1", 0), m["n_images"],
        )

    with open(metrics_dir / "phase1_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return all_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    subject = "subj01"
    results = {}

    log.info("=" * 70)
    log.info("PHASE 1 EVALUATION — Road to 90%%+ R@1")
    log.info("=" * 70)

    v55b_dir = Path("experimental_results/V55b_subj01_finetune/subj01")
    if (v55b_dir / "checkpoint_best.pt").exists():
        results["V55b"] = evaluate_experiment(v55b_dir, subject, "V55b")

    v55a_dir = Path("experimental_results/V55a_multi_subject_dual_head/subj01")
    if (v55a_dir / "checkpoint_best.pt").exists():
        results["V55a"] = evaluate_experiment(v55a_dir, subject, "V55a")

    # --- Fusion sweep ---
    log.info("\n" + "=" * 70)
    log.info("FUSION SWEEP: V55b + N1v28a")
    log.info("=" * 70)

    v55b_m = Path("experimental_results/V55b_subj01_finetune/subj01/metrics")
    n1v28a_m = Path("experimental_results/N1v28a_dual_head/subj01/metrics")

    for split in ["shared1000", "val"]:
        c_pred = v55b_m / f"{split}_predictions.npy"
        c_gt = v55b_m / f"{split}_ground_truth.npy"
        c_kappa = v55b_m / f"{split}_kappas.npy"

        l_pred = n1v28a_m / f"{split}_predictions_compact.npy"
        l_gt = n1v28a_m / f"{split}_ground_truth_compact.npy"

        if not l_pred.exists():
            l_pred = n1v28a_m / f"{split}_predictions.npy"
            l_gt = n1v28a_m / f"{split}_ground_truth.npy"

        if not c_pred.exists() or not l_pred.exists():
            log.warning("Missing predictions for %s fusion (compact=%s, legacy=%s)",
                        split, c_pred.exists(), l_pred.exists())
            continue

        compact_preds = np.load(c_pred)
        compact_gts = np.load(c_gt)
        compact_kappas = np.load(c_kappa) if c_kappa.exists() else None

        legacy_preds = np.load(l_pred)
        legacy_gts = np.load(l_gt)

        for arr_name, arr in [("compact_preds", compact_preds), ("compact_gts", compact_gts),
                              ("legacy_preds", legacy_preds), ("legacy_gts", legacy_gts)]:
            norms = np.linalg.norm(arr, axis=-1, keepdims=True) + 1e-8
            arr /= norms

        if compact_preds.shape[1] != legacy_preds.shape[1]:
            log.warning(
                "Dim mismatch: compact=%d, legacy=%d. Using nsd_id matching.",
                compact_preds.shape[1], legacy_preds.shape[1],
            )
            continue

        n_c, n_l = len(compact_preds), len(legacy_preds)
        if n_c != n_l:
            c_ids = np.load(v55b_m / f"{split}_nsd_ids.npy")
            l_ids = np.load(n1v28a_m / f"{split}_nsd_ids.npy")
            common = np.intersect1d(c_ids, l_ids)
            log.info("Size mismatch (%d vs %d), using %d common nsd_ids", n_c, n_l, len(common))
            c_idx = np.array([np.where(c_ids == cid)[0][0] for cid in common])
            l_idx = np.array([np.where(l_ids == cid)[0][0] for cid in common])
            compact_preds, compact_gts = compact_preds[c_idx], compact_gts[c_idx]
            if compact_kappas is not None:
                compact_kappas = compact_kappas[c_idx]
            legacy_preds, legacy_gts = legacy_preds[l_idx], legacy_gts[l_idx]

        fusion_results = compute_fusion_sweep(
            compact_preds, compact_gts, compact_kappas,
            legacy_preds, legacy_gts,
        )

        log.info("\n--- %s: Top 10 Fusion Results ---", split)
        for r in fusion_results[:10]:
            log.info(
                "  alpha=%.2f gamma=%.2f [%s] R@1=%.3f  R@5=%.3f  R@10=%.3f",
                r["alpha"], r["gamma"], r["score_type"],
                r["r@1"], r.get("r@5", 0), r.get("r@10", 0),
            )

        results[f"fusion_{split}"] = fusion_results[:30]

    # --- Save all results ---
    out_path = Path("experimental_results/phase1_evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("\nResults saved to: %s", out_path)

    # --- Summary ---
    log.info("\n" + "=" * 70)
    log.info("PHASE 1 SUMMARY")
    log.info("=" * 70)
    for exp_name, exp_data in results.items():
        if isinstance(exp_data, dict) and any(k in exp_data for k in ["val", "shared1000"]):
            for split, m in exp_data.items():
                if isinstance(m, dict):
                    log.info(
                        "  %-8s %-12s  raw=%.3f  csls=%.3f  ppr_csls=%.3f  (n=%d)",
                        exp_name, split,
                        m.get("raw_r@1", 0), m.get("csls_r@1", 0),
                        m.get("ppr_csls_r@1", 0), m.get("n_images", 0),
                    )
        elif isinstance(exp_data, list) and exp_data:
            best = exp_data[0]
            log.info(
                "  %-20s  alpha=%.2f gamma=%.2f [%s] R@1=%.3f",
                exp_name, best["alpha"], best["gamma"], best["score_type"], best["r@1"],
            )

    prev_best = 0.772
    log.info("\n  Previous best (V35+N1v28a frozen): %.1f%%", prev_best * 100)
    for k in ["fusion_shared1000", "fusion_val"]:
        if k in results and results[k]:
            best_fusion = results[k][0]["r@1"]
            delta = best_fusion - prev_best
            log.info("  New best %s: %.1f%% (%+.1fpp vs previous)", k, best_fusion * 100, delta * 100)


if __name__ == "__main__":
    main()
