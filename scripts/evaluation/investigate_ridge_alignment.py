#!/usr/bin/env python3
"""Investigate MindEye2-style ridge regression alignment for cross-subject decoding.

This script evaluates whether ridge regression alignment (as used in MindEye2,
Scotti et al. 2024) can replace the learned per-subject linear adapters currently
used in the cross-subject training pipeline. Ridge alignment is analytically
fitted (no SGD), faster to train, and may provide better functional alignment.

The investigation:
1. Loads pre-extracted fMRI features for multiple subjects
2. Loads shared CLIP embeddings (the common target space)
3. Fits per-subject ridge regression: fMRI -> shared latent space
4. Evaluates alignment quality via cosine similarity and retrieval metrics
5. Compares ridge alignment vs. learned adapters (from V54a)

Usage:
    python scripts/evaluation/investigate_ridge_alignment.py \
        --subjects subj01 subj02 subj05 subj07 \
        --target-subject subj01 \
        --alpha-values 0.1 1.0 10.0 100.0 1000.0 \
        --output-dir outputs/ridge_alignment_investigation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_fmri_features(
    subject: str,
    cache_root: Path,
) -> np.ndarray:
    """Load pre-extracted fMRI features for a subject."""
    feat_path = cache_root / f"subject={subject}" / "fmri_features.npy"
    if not feat_path.exists():
        raise FileNotFoundError(f"Features not found: {feat_path}")
    features = np.load(feat_path)
    logger.info("Loaded %s: shape=%s", subject, features.shape)
    return features


def load_trial_meta(
    subject: str,
    cache_root: Path,
) -> pd.DataFrame:
    """Load trial metadata for a subject."""
    meta_path = cache_root / f"subject={subject}" / "trial_meta.parquet"
    if not meta_path.exists():
        idx_path = Path("data/indices/nsd_index") / f"subject={subject}" / "index.parquet"
        if idx_path.exists():
            return pd.read_parquet(idx_path)
        raise FileNotFoundError(f"Metadata not found: {meta_path} or {idx_path}")
    return pd.read_parquet(meta_path)


def load_clip_embeddings(
    clip_cache_path: str = "outputs/clip_cache/clip.parquet",
    embedding_column: str = "fused",
) -> Dict[int, np.ndarray]:
    """Load CLIP embeddings indexed by nsdId."""
    df = pd.read_parquet(clip_cache_path)

    col = None
    for candidate in [embedding_column, "embedding", "fused", "final"]:
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        if emb_cols:
            embeddings = df[emb_cols].values.astype(np.float32)
            nsd_ids = df["nsdId"].values
            return dict(zip(nsd_ids, embeddings))
        raise ValueError(f"No embedding column found in {clip_cache_path}")

    if df[col].dtype == object:
        embeddings = np.stack(df[col].values).astype(np.float32)
    else:
        embeddings = df[col].values.astype(np.float32)

    nsd_ids = df["nsdId"].values
    return dict(zip(nsd_ids, embeddings))


def build_paired_data(
    features: np.ndarray,
    meta: pd.DataFrame,
    clip_embeddings: Dict[int, np.ndarray],
    average_repetitions: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build paired (fMRI, CLIP) matrices for ridge regression.

    Returns:
        X: (N, V) fMRI features
        Y: (N, D) CLIP embeddings
        nsd_ids: (N,) corresponding nsdIds
    """
    if "nsdId" not in meta.columns:
        for col in ["nsd_id", "nsdid", "stimulus_id"]:
            if col in meta.columns:
                meta = meta.rename(columns={col: "nsdId"})
                break

    valid_mask = meta["nsdId"].isin(clip_embeddings.keys())
    meta_valid = meta[valid_mask].reset_index(drop=True)
    feat_valid = features[valid_mask.values]

    if average_repetitions:
        grouped = meta_valid.groupby("nsdId")
        unique_ids = sorted(grouped.groups.keys())
        X_list, Y_list = [], []
        for nsd_id in unique_ids:
            idx = grouped.groups[nsd_id]
            X_list.append(feat_valid[idx].mean(axis=0))
            Y_list.append(clip_embeddings[nsd_id])
        return (
            np.array(X_list, dtype=np.float32),
            np.array(Y_list, dtype=np.float32),
            np.array(unique_ids, dtype=np.int64),
        )
    else:
        X = feat_valid
        Y = np.array(
            [clip_embeddings[nid] for nid in meta_valid["nsdId"]],
            dtype=np.float32,
        )
        return X, Y, meta_valid["nsdId"].values


def fit_ridge(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    alpha: float = 1.0,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit ridge regression analytically: W = (X^T X + alpha I)^{-1} X^T Y.

    Args:
        X_train: (N, V) fMRI features
        Y_train: (N, D) CLIP embeddings
        alpha: regularization strength

    Returns:
        W: (V, D) weight matrix
        bias: (D,) bias vector
    """
    X_mean = X_train.mean(axis=0)
    Y_mean = Y_train.mean(axis=0)
    X_c = X_train - X_mean
    Y_c = Y_train - Y_mean

    X_t = torch.from_numpy(X_c).to(device).float()
    Y_t = torch.from_numpy(Y_c).to(device).float()

    V = X_t.shape[1]
    I = torch.eye(V, device=device) * alpha
    XtX = X_t.T @ X_t + I
    XtY = X_t.T @ Y_t

    logger.info(
        "Solving ridge (V=%d, D=%d, N=%d, alpha=%.1f)...",
        V, Y_t.shape[1], X_t.shape[0], alpha,
    )
    W = torch.linalg.solve(XtX, XtY)

    W_np = W.cpu().numpy()
    bias = Y_mean - X_mean @ W_np

    return W_np, bias


def predict_ridge(
    X: np.ndarray,
    W: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Apply ridge regression to get predictions."""
    return X @ W + bias


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize along last axis."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


def compute_retrieval_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    ks: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """Compute retrieval metrics (R@K, MRR, median rank)."""
    pred_norm = l2_normalize(predictions)
    tgt_norm = l2_normalize(targets)

    sim = pred_norm @ tgt_norm.T
    N = sim.shape[0]

    ranks = np.zeros(N, dtype=np.int64)
    for i in range(N):
        sorted_idx = np.argsort(-sim[i])
        rank = np.where(sorted_idx == i)[0][0] + 1
        ranks[i] = rank

    metrics = {}
    for k in ks:
        metrics[f"R@{k}"] = float(np.mean(ranks <= k))
    metrics["MRR"] = float(np.mean(1.0 / ranks))
    metrics["MedianRank"] = float(np.median(ranks))
    metrics["MeanRank"] = float(np.mean(ranks))

    cosine_sims = np.array([sim[i, i] for i in range(N)])
    metrics["mean_cosine"] = float(np.mean(cosine_sims))
    metrics["std_cosine"] = float(np.std(cosine_sims))

    return metrics


def investigate_single_subject(
    subject: str,
    cache_root: Path,
    clip_embeddings: Dict[int, np.ndarray],
    alpha_values: List[float],
    val_fraction: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
) -> Dict[str, any]:
    """Run ridge alignment investigation for a single subject."""
    features = load_fmri_features(subject, cache_root)
    meta = load_trial_meta(subject, cache_root)
    X, Y, nsd_ids = build_paired_data(features, meta, clip_embeddings)

    rng = np.random.RandomState(seed)
    n_total = len(X)
    n_val = int(n_total * val_fraction)
    perm = rng.permutation(n_total)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    logger.info(
        "%s: %d train, %d val images, %d voxels, %d embedding dim",
        subject, len(train_idx), len(val_idx), X.shape[1], Y.shape[1],
    )

    results = {
        "subject": subject,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_voxels": X.shape[1],
        "embedding_dim": Y.shape[1],
        "alpha_sweep": {},
    }

    best_val_r1 = 0.0
    best_alpha = alpha_values[0]

    for alpha in alpha_values:
        logger.info("  alpha=%.1f ...", alpha)
        W, bias = fit_ridge(X_train, Y_train, alpha=alpha, device=device)

        train_pred = predict_ridge(X_train, W, bias)
        val_pred = predict_ridge(X_val, W, bias)

        train_metrics = compute_retrieval_metrics(train_pred, Y_train)
        val_metrics = compute_retrieval_metrics(val_pred, Y_val)

        results["alpha_sweep"][str(alpha)] = {
            "train": train_metrics,
            "val": val_metrics,
        }

        logger.info(
            "    train R@1=%.3f, val R@1=%.3f, val cosine=%.4f",
            train_metrics["R@1"],
            val_metrics["R@1"],
            val_metrics["mean_cosine"],
        )

        if val_metrics["R@1"] > best_val_r1:
            best_val_r1 = val_metrics["R@1"]
            best_alpha = alpha

    results["best_alpha"] = best_alpha
    results["best_val_r1"] = best_val_r1
    logger.info(
        "%s: best alpha=%.1f → val R@1=%.3f",
        subject, best_alpha, best_val_r1,
    )

    return results


def investigate_cross_subject(
    subjects: List[str],
    target_subject: str,
    cache_root: Path,
    clip_embeddings: Dict[int, np.ndarray],
    alpha: float,
    val_fraction: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
) -> Dict[str, any]:
    """Test cross-subject generalization: train on other subjects, test on target."""
    all_X_train, all_Y_train = [], []

    for subj in subjects:
        if subj == target_subject:
            continue
        features = load_fmri_features(subj, cache_root)
        meta = load_trial_meta(subj, cache_root)
        X, Y, _ = build_paired_data(features, meta, clip_embeddings)
        all_X_train.append(X)
        all_Y_train.append(Y)
        logger.info("  source %s: %d images", subj, len(X))

    target_features = load_fmri_features(target_subject, cache_root)
    target_meta = load_trial_meta(target_subject, cache_root)
    X_target, Y_target, _ = build_paired_data(
        target_features, target_meta, clip_embeddings,
    )

    rng = np.random.RandomState(seed)
    n_total = len(X_target)
    n_val = int(n_total * val_fraction)
    perm = rng.permutation(n_total)
    target_val_idx = perm[:n_val]
    target_train_idx = perm[n_val:]

    logger.info(
        "Cross-subject: %d source subjects, target=%s (%d train, %d val)",
        len(subjects) - 1, target_subject, len(target_train_idx), len(target_val_idx),
    )

    X_source = np.concatenate(all_X_train, axis=0)
    Y_source = np.concatenate(all_Y_train, axis=0)

    max_voxels = max(X_source.shape[1], X_target.shape[1])
    if X_source.shape[1] != max_voxels:
        X_source = np.pad(X_source, ((0, 0), (0, max_voxels - X_source.shape[1])))
    if X_target.shape[1] != max_voxels:
        X_target = np.pad(X_target, ((0, 0), (0, max_voxels - X_target.shape[1])))

    W_source, bias_source = fit_ridge(X_source, Y_source, alpha=alpha, device=device)
    source_pred_on_target_val = predict_ridge(X_target[target_val_idx], W_source, bias_source)
    cross_metrics = compute_retrieval_metrics(source_pred_on_target_val, Y_target[target_val_idx])

    W_target, bias_target = fit_ridge(
        X_target[target_train_idx], Y_target[target_train_idx],
        alpha=alpha, device=device,
    )
    target_pred_on_val = predict_ridge(X_target[target_val_idx], W_target, bias_target)
    within_metrics = compute_retrieval_metrics(target_pred_on_val, Y_target[target_val_idx])

    logger.info(
        "Cross-subject → target val R@1=%.3f | Within-subject R@1=%.3f",
        cross_metrics["R@1"], within_metrics["R@1"],
    )

    return {
        "target_subject": target_subject,
        "source_subjects": [s for s in subjects if s != target_subject],
        "alpha": alpha,
        "n_source_images": len(X_source),
        "n_target_train": len(target_train_idx),
        "n_target_val": len(target_val_idx),
        "cross_subject_metrics": cross_metrics,
        "within_subject_metrics": within_metrics,
        "gap_r1": within_metrics["R@1"] - cross_metrics["R@1"],
    }


def main():
    parser = argparse.ArgumentParser(description="Ridge alignment investigation")
    parser.add_argument(
        "--subjects", nargs="+", default=["subj01", "subj02", "subj05", "subj07"],
    )
    parser.add_argument("--target-subject", default="subj01")
    parser.add_argument(
        "--alpha-values", nargs="+", type=float,
        default=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
    )
    parser.add_argument("--cache-root", default="cache/preextracted")
    parser.add_argument("--clip-cache", default="outputs/clip_cache/clip.parquet")
    parser.add_argument("--embedding-column", default="fused")
    parser.add_argument("--output-dir", default="outputs/ridge_alignment_investigation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading CLIP embeddings from %s ...", args.clip_cache)
    clip_embeddings = load_clip_embeddings(args.clip_cache, args.embedding_column)
    logger.info("Loaded %d CLIP embeddings", len(clip_embeddings))

    cache_root = Path(args.cache_root)
    all_results = {"per_subject": {}, "cross_subject": None}

    for subj in args.subjects:
        feat_path = cache_root / f"subject={subj}" / "fmri_features.npy"
        if not feat_path.exists():
            logger.warning("Skipping %s: features not found at %s", subj, feat_path)
            continue

        results = investigate_single_subject(
            subject=subj,
            cache_root=cache_root,
            clip_embeddings=clip_embeddings,
            alpha_values=args.alpha_values,
            device=args.device,
            seed=args.seed,
        )
        all_results["per_subject"][subj] = results

    available_subjects = [
        s for s in args.subjects
        if (cache_root / f"subject={s}" / "fmri_features.npy").exists()
    ]
    if len(available_subjects) >= 2 and args.target_subject in available_subjects:
        best_alpha = all_results["per_subject"].get(
            args.target_subject, {},
        ).get("best_alpha", 100.0)

        cross_results = investigate_cross_subject(
            subjects=available_subjects,
            target_subject=args.target_subject,
            cache_root=cache_root,
            clip_embeddings=clip_embeddings,
            alpha=best_alpha,
            device=args.device,
            seed=args.seed,
        )
        all_results["cross_subject"] = cross_results

    out_path = output_dir / "ridge_alignment_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)

    logger.info("\n" + "=" * 60)
    logger.info("RIDGE ALIGNMENT INVESTIGATION SUMMARY")
    logger.info("=" * 60)
    for subj, res in all_results["per_subject"].items():
        logger.info(
            "  %s: best_alpha=%.1f, val R@1=%.3f",
            subj, res["best_alpha"], res["best_val_r1"],
        )
    if all_results["cross_subject"]:
        cr = all_results["cross_subject"]
        logger.info(
            "  Cross-subject → %s: R@1=%.3f (within=%.3f, gap=%.3f)",
            cr["target_subject"],
            cr["cross_subject_metrics"]["R@1"],
            cr["within_subject_metrics"]["R@1"],
            cr["gap_r1"],
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
