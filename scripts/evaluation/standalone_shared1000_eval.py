#!/usr/bin/env python3
"""Standalone shared1000 evaluation — runs on a separate GPU stream without
interrupting ongoing training. Loads checkpoint_best.pt, evaluates on shared1000."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    exp_dir = Path(f"experimental_results/{args.experiment}/{args.subject}")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (exp_dir / "checkpoint_best.pt")
    if not ckpt_path.exists():
        print(f"ERROR: No checkpoint at {ckpt_path}")
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fmri2img.models.unified_model import create_model

    cache_root = os.environ.get("CACHE_ROOT", "cache")
    features_path = Path(cache_root) / "preextracted" / f"subject={args.subject}" / "fmri_features.npy"
    meta_path = Path(cache_root) / "preextracted" / f"subject={args.subject}" / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    input_dim = meta["n_voxels"]

    # Infer output_dim from checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    mu_key = "decoder.mu_head.weight"
    embedding_dim = state_dict[mu_key].shape[0] if mu_key in state_dict else 768
    ckpt_epoch = ckpt.get("epoch", "?")
    print(f"  Checkpoint epoch: {ckpt_epoch}, embedding_dim: {embedding_dim}")

    model_config = config["model"]
    model_config["encoder"]["input_dim"] = input_dim
    model_config["decoder"]["output_dim"] = embedding_dim
    token_dim = model_config["decoder"].get("token_dim", 768)
    if token_dim and embedding_dim > token_dim:
        model_config["decoder"]["num_tokens"] = embedding_dim // token_dim
        model_config["decoder"]["token_dim"] = token_dim

    model = create_model(model_config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded: {len(missing)} missing, {len(unexpected)} unexpected")
    model = model.to(args.device).float()
    model.eval()

    # Load shared1000 data
    index_df = pd.read_parquet(f"data/indices/nsd_index/subject={args.subject}/index.parquet")
    s1k_mask = index_df["shared1000"].fillna(False).astype(bool).values
    features = np.load(features_path, mmap_mode="r")
    s1k_features = np.array(features[s1k_mask], dtype=np.float32)
    s1k_df = index_df[s1k_mask].reset_index(drop=True)

    # Z-score normalization
    zscore_dir = exp_dir / "zscore_stats"
    if zscore_dir.exists() and "session" in s1k_df.columns:
        fb_mean_path = zscore_dir / "global_fallback_mean.npy"
        fb_std_path = zscore_dir / "global_fallback_std.npy"
        fb_mean = np.load(fb_mean_path) if fb_mean_path.exists() else None
        fb_std = np.load(fb_std_path) if fb_std_path.exists() else None
        sessions = s1k_df["session"].values
        for sess in np.unique(sessions):
            m_path = zscore_dir / f"session_{int(sess)}_mean.npy"
            s_path = zscore_dir / f"session_{int(sess)}_std.npy"
            if not m_path.exists():
                m_path = zscore_dir / f"{args.subject}_session_{int(sess)}_mean.npy"
                s_path = zscore_dir / f"{args.subject}_session_{int(sess)}_std.npy"
            sess_mask = sessions == sess
            if m_path.exists():
                s_mean, s_std = np.load(m_path), np.load(s_path)
            elif fb_mean is not None:
                s_mean, s_std = fb_mean, fb_std
            else:
                continue
            s1k_features[sess_mask] = ((s1k_features[sess_mask] - s_mean) / s_std).astype(np.float32)
        print(f"  Z-scored per-session")

    nsd_ids = s1k_df["nsdId"].values
    unique_ids = np.unique(nsd_ids)
    n = len(unique_ids)
    print(f"  Shared1000: {len(s1k_features)} trials -> {n} images")

    # --- DATA LEAKAGE VERIFICATION (inline) ---
    non_s1k = index_df[~s1k_mask]
    from sklearn.model_selection import train_test_split
    non_s1k_unique = sorted(non_s1k["nsdId"].unique())
    train_ids_set, val_ids_set = train_test_split(non_s1k_unique, test_size=0.10, random_state=42)
    train_ids_set, val_ids_set = set(train_ids_set), set(val_ids_set)
    s1k_set = set(unique_ids)
    leak = s1k_set & (train_ids_set | val_ids_set)
    print(f"  Leak check: {len(leak)} shared1000 images in train/val (should be 0)")

    # Single deterministic forward pass
    from torch.utils.data import DataLoader, TensorDataset
    batch_size = 32
    ds = TensorDataset(torch.from_numpy(s1k_features))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_mu, all_kappa = [], []
    with torch.no_grad():
        for (batch_fmri,) in loader:
            batch_fmri = batch_fmri.to(args.device, dtype=torch.float32)
            out = model(batch_fmri)
            pred, aux = (out if isinstance(out, tuple) else (out, None))
            all_mu.append(pred.cpu().numpy())
            if aux is not None:
                all_kappa.append(aux.squeeze(-1).cpu().numpy())
    trial_preds = np.concatenate(all_mu)
    trial_kappas = np.concatenate(all_kappa) if all_kappa else None

    # Normalize
    norms = np.linalg.norm(trial_preds, axis=-1, keepdims=True)
    trial_preds = trial_preds / np.maximum(norms, 1e-8)

    # Aggregate by nsdId (kappa-weighted)
    preds = np.zeros((n, embedding_dim), dtype=np.float32)
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
    token_cache_path = config["data"].get("token_cache_path")
    if token_cache_path and os.path.exists(token_cache_path) and embedding_dim > 1000:
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        from fmri2img.data.token_clip_cache import TokenCLIPCache
        token_cache = TokenCLIPCache(token_cache_path)
        gt = np.zeros((n, embedding_dim), dtype=np.float32)
        for i, uid in enumerate(unique_ids):
            try:
                gt[i] = token_cache.get_flat(int(uid))
            except Exception:
                pass
        gt_check = np.linalg.norm(gt, axis=-1)
        print(f"  GT loaded: {(gt_check > 0).sum()}/{n} non-zero")
    else:
        clip_df = pd.read_parquet("outputs/clip_cache/clip.parquet")
        emb_col = config["data"].get("embedding_column", "fused")
        gt = np.zeros((n, embedding_dim), dtype=np.float32)
        for i, uid in enumerate(unique_ids):
            row = clip_df[clip_df["nsdId"] == int(uid)]
            if len(row):
                gt[i] = np.asarray(row.iloc[0][emb_col], dtype=np.float32)

    gt_norms = np.linalg.norm(gt, axis=-1, keepdims=True)
    gt = gt / np.maximum(gt_norms, 1e-8)

    # Compute metrics on GPU
    pred_t = torch.from_numpy(preds).to(args.device).float()
    gt_t = torch.from_numpy(gt).to(args.device).float()
    sims = (pred_t @ gt_t.T).cpu().numpy()

    def compute_metrics(sims, csls_k=10):
        diag = np.array([sims[i, i] for i in range(n)])
        ranks = np.array([(sims[i] > diag[i]).sum() + 1 for i in range(n)])
        r1 = (ranks == 1).mean()
        r5 = (ranks <= 5).mean()
        r10 = (ranks <= 10).mean()
        mrr = (1.0 / ranks).mean()
        topk_p = np.partition(-sims, csls_k, axis=1)[:, :csls_k]
        hub_s = -topk_p.mean(axis=1)
        topk_g = np.partition(-(sims.T), csls_k, axis=1)[:, :csls_k]
        hub_t = -topk_g.mean(axis=1)
        csls = 2 * sims - hub_s[:, None] - hub_t[None, :]
        csls_diag = np.array([csls[i, i] for i in range(n)])
        csls_ranks = np.array([(csls[i] > csls_diag[i]).sum() + 1 for i in range(n)])
        return {"r1": r1, "r5": r5, "r10": r10, "mrr": mrr,
                "csls_r1": (csls_ranks == 1).mean(),
                "csls_r5": (csls_ranks <= 5).mean(),
                "mean_pos_sim": diag.mean(),
                "med_rank": np.median(ranks)}

    print(f"\n{'='*60}")
    print(f"SHARED1000 RESULTS (deterministic, epoch {ckpt_epoch})")
    print(f"{'='*60}")
    for k in [3, 5, 10]:
        m = compute_metrics(sims, csls_k=k)
        print(f"  CSLS k={k:2d}: R@1={m['r1']:.4f}  CSLS_R@1={m['csls_r1']:.4f}  "
              f"R@5={m['r5']:.4f}  MRR={m['mrr']:.4f}  pos_sim={m['mean_pos_sim']:.4f}")

    # Kappa stats
    if trial_kappas is not None:
        print(f"\n  Kappa: mean={trial_kappas.mean():.1f}  std={trial_kappas.std():.1f}  "
              f"min={trial_kappas.min():.1f}  max={trial_kappas.max():.1f}")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
