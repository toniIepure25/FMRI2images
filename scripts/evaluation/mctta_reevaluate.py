#!/usr/bin/env python3
"""Re-evaluate an existing checkpoint on shared1000 with MC-TTA.

Loads a trained model checkpoint, runs MC-TTA forward passes with dropout
enabled, then computes shared1000 retrieval metrics. This gives better
predictions than single deterministic pass.

Usage:
    python scripts/evaluation/mctta_reevaluate.py \
        --experiment V61a_finetune_difflr \
        --config configs/experiments/V61a_finetune_difflr.yaml \
        --mc-samples 16
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mc-samples", type=int, default=16)
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    exp_dir = Path(f"experimental_results/{args.experiment}/{args.subject}")
    ckpt_path = exp_dir / "checkpoint_best.pt"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    # Load model
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fmri2img.models.unified_model import create_model

    cache_root = os.environ.get("CACHE_ROOT", "cache")
    features_path = Path(cache_root) / "preextracted" / f"subject={args.subject}" / "fmri_features.npy"
    meta_path = Path(cache_root) / "preextracted" / f"subject={args.subject}" / "meta.json"

    with open(meta_path) as f:
        meta = json.load(f)
    input_dim = meta["n_voxels"]

    # Load checkpoint first to infer output dimensions
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Infer output_dim from checkpoint's decoder.mu_head.weight
    mu_head_key = "decoder.mu_head.weight"
    if mu_head_key in state_dict:
        embedding_dim = state_dict[mu_head_key].shape[0]
    else:
        embedding_dim = config["model"]["decoder"].get("output_dim", 768) or 768
    print(f"Inferred embedding_dim from checkpoint: {embedding_dim}")

    model_config = config["model"]
    model_config["encoder"]["input_dim"] = input_dim
    model_config["decoder"]["output_dim"] = embedding_dim

    # Set token geometry if applicable
    token_dim = model_config["decoder"].get("token_dim", 768)
    if token_dim and embedding_dim > token_dim:
        num_tokens = embedding_dim // token_dim
        model_config["decoder"]["num_tokens"] = num_tokens
        model_config["decoder"]["token_dim"] = token_dim
        print(f"Token mode: {num_tokens} tokens x {token_dim} = {embedding_dim}-D")
    else:
        print(f"CLS mode: {embedding_dim}-D")

    # Load token cache for ground truth
    token_cache_path = config["data"].get("token_cache_path")
    token_cache = None
    if token_cache_path and os.path.exists(token_cache_path) and embedding_dim > 1000:
        from fmri2img.data.token_clip_cache import TokenCLIPCache
        token_cache = TokenCLIPCache(token_cache_path)
        print(f"Token cache loaded for GT: {token_cache.num_tokens}x{token_cache.token_dim}")

    model = create_model(model_config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")

    model = model.to(args.device).float()
    model_type = getattr(model, "model_type", "vmf")

    # Load shared1000 data
    index_path = Path(f"data/indices/nsd_index/subject={args.subject}/index.parquet")
    index_df = pd.read_parquet(index_path)
    s1000_mask = index_df["shared1000"].fillna(False).astype(bool).values
    features = np.load(features_path, mmap_mode="r")
    s1000_features = np.array(features[s1000_mask], dtype=np.float32)
    s1000_df = index_df[s1000_mask].reset_index(drop=True)

    # Z-score
    zscore_dir = exp_dir / "zscore_stats"
    if zscore_dir.exists() and "session" in s1000_df.columns:
        fb_mean_path = zscore_dir / "global_fallback_mean.npy"
        fb_std_path = zscore_dir / "global_fallback_std.npy"
        fb_mean = np.load(fb_mean_path) if fb_mean_path.exists() else None
        fb_std = np.load(fb_std_path) if fb_std_path.exists() else None
        sessions = s1000_df["session"].values
        applied = 0
        for sess in np.unique(sessions):
            m_path = zscore_dir / f"session_{int(sess)}_mean.npy"
            s_path = zscore_dir / f"session_{int(sess)}_std.npy"
            if not m_path.exists():
                m_path = zscore_dir / f"{args.subject}_session_{int(sess)}_mean.npy"
                s_path = zscore_dir / f"{args.subject}_session_{int(sess)}_std.npy"
            sess_mask = sessions == sess
            if m_path.exists():
                s_mean = np.load(m_path)
                s_std = np.load(s_path)
            elif fb_mean is not None:
                s_mean, s_std = fb_mean, fb_std
            else:
                continue
            s1000_features[sess_mask] = ((s1000_features[sess_mask] - s_mean) / s_std).astype(np.float32)
            applied += int(sess_mask.sum())
        print(f"Z-scored {applied}/{len(s1000_features)} trials (per-session)")

    nsd_ids = s1000_df["nsdId"].values
    unique_ids = np.unique(nsd_ids)
    n_images = len(unique_ids)
    print(f"Shared1000: {len(s1000_features)} trials -> {n_images} images")

    # MC-TTA forward passes
    batch_size = config["training"].get("batch_size", 64)
    ds = TensorDataset(torch.from_numpy(s1000_features))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    print(f"\nRunning MC-TTA with {args.mc_samples} samples...")
    model.train()  # enable dropout

    mu_sum = None
    kappa_sum = None
    n_trials = len(s1000_features)

    for sample_idx in range(args.mc_samples):
        batch_preds, batch_kappas = [], []
        with torch.no_grad():
            for (batch_fmri,) in loader:
                batch_fmri = batch_fmri.to(args.device, dtype=torch.float32)
                out = model(batch_fmri)
                pred, aux = (out if isinstance(out, tuple) else (out, None))
                batch_preds.append(pred.cpu().numpy())
                if aux is not None:
                    k = aux.squeeze(-1)
                    batch_kappas.append(k.cpu().numpy())

        sample_mu = np.concatenate(batch_preds)
        if mu_sum is None:
            mu_sum = sample_mu.copy()
        else:
            mu_sum += sample_mu

        if batch_kappas:
            sample_k = np.concatenate(batch_kappas)
            if kappa_sum is None:
                kappa_sum = sample_k.copy()
            else:
                kappa_sum += sample_k

        if (sample_idx + 1) % 4 == 0:
            print(f"  Sample {sample_idx + 1}/{args.mc_samples}")

    model.eval()

    # Average and normalize
    trial_preds = mu_sum / args.mc_samples
    trial_norms = np.linalg.norm(trial_preds, axis=-1, keepdims=True)
    trial_preds = trial_preds / np.maximum(trial_norms, 1e-8)
    trial_kappas = (kappa_sum / args.mc_samples) if kappa_sum is not None else None

    # Aggregate by nsdId (kappa-weighted)
    preds = np.zeros((n_images, trial_preds.shape[1]), dtype=np.float32)
    for i, uid in enumerate(unique_ids):
        mask = nsd_ids == uid
        if trial_kappas is not None:
            kw = trial_kappas[mask]
            kw = kw / (kw.sum() + 1e-8)
            preds[i] = (trial_preds[mask] * kw[:, None]).sum(axis=0)
        else:
            preds[i] = trial_preds[mask].mean(axis=0)
    norms = np.linalg.norm(preds, axis=-1, keepdims=True)
    preds = preds / np.maximum(norms, 1e-8)

    # Ground truth
    if token_cache is not None:
        gt = np.zeros((n_images, embedding_dim), dtype=np.float32)
        for i, uid in enumerate(unique_ids):
            try:
                gt[i] = token_cache.get_flat(int(uid))
            except (KeyError, Exception):
                pass
    else:
        clip_df = pd.read_parquet("outputs/clip_cache/clip.parquet")
        emb_col = config["data"].get("embedding_column", "fused")
        cols = clip_df.columns.tolist()
        if emb_col not in cols:
            for c in cols:
                if c not in ("nsdId", "nsd_id"):
                    emb_col = c
                    break
        emb_lookup = {int(row["nsdId"]): i for i, (_, row) in enumerate(clip_df.iterrows())}
        gt = np.zeros((n_images, embedding_dim), dtype=np.float32)
        for i, uid in enumerate(unique_ids):
            idx = emb_lookup.get(int(uid))
            if idx is not None:
                gt[i] = np.asarray(clip_df.iloc[idx][emb_col], dtype=np.float32)

    gt_norms = np.linalg.norm(gt, axis=-1, keepdims=True)
    gt = gt / np.maximum(gt_norms, 1e-8)

    # Compute metrics on GPU
    pred_t = torch.from_numpy(preds).to(args.device).float()
    gt_t = torch.from_numpy(gt).to(args.device).float()
    sims = (pred_t @ gt_t.T).cpu().numpy()

    def compute_metrics(sims, csls_k=10):
        diag = np.array([sims[i, i] for i in range(n_images)])
        ranks = np.array([(sims[i] > diag[i]).sum() + 1 for i in range(n_images)])
        r1 = (ranks == 1).mean()
        r5 = (ranks <= 5).mean()
        mrr = (1.0 / ranks).mean()
        topk_p = np.partition(-sims, csls_k, axis=1)[:, :csls_k]
        hub_s = -topk_p.mean(axis=1)
        topk_g = np.partition(-(sims.T), csls_k, axis=1)[:, :csls_k]
        hub_t = -topk_g.mean(axis=1)
        csls = 2 * sims - hub_s[:, None] - hub_t[None, :]
        csls_diag = np.array([csls[i, i] for i in range(n_images)])
        csls_ranks = np.array([(csls[i] > csls_diag[i]).sum() + 1 for i in range(n_images)])
        return {"r1": r1, "r5": r5, "mrr": mrr,
                "csls_r1": (csls_ranks == 1).mean(),
                "csls_r5": (csls_ranks <= 5).mean(),
                "mean_pos_sim": diag.mean()}

    print(f"\n{'='*60}")
    print(f"MC-TTA RESULTS ({args.mc_samples} samples)")
    print(f"{'='*60}")
    for k in [3, 5, 10]:
        m = compute_metrics(sims, csls_k=k)
        print(f"  k={k:2d}: R@1={m['r1']:.3f}  CSLS_R@1={m['csls_r1']:.3f}  "
              f"R@5={m['r5']:.3f}  MRR={m['mrr']:.3f}  pos_sim={m['mean_pos_sim']:.4f}")

    # Save predictions for fusion
    out_dir = exp_dir / "metrics"
    np.save(out_dir / f"shared1000_predictions_mctta{args.mc_samples}.npy", preds)
    print(f"\nSaved MC-TTA predictions to {out_dir / f'shared1000_predictions_mctta{args.mc_samples}.npy'}")

    # Also save metrics
    m10 = compute_metrics(sims, csls_k=10)
    m3 = compute_metrics(sims, csls_k=3)
    mc_metrics = {
        "mc_tta_samples": args.mc_samples,
        "csls_r@1_k10": float(m10["csls_r1"]),
        "csls_r@1_k3": float(m3["csls_r1"]),
        "r@1": float(m10["r1"]),
        "r@5": float(m10["r5"]),
        "mean_pos_sim": float(m10["mean_pos_sim"]),
    }
    with open(out_dir / f"shared1000_mctta{args.mc_samples}_metrics.json", "w") as f:
        json.dump(mc_metrics, f, indent=2)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
