#!/usr/bin/env python3
"""
Training Signal Diagnostic: Tier 2 checks for R@1 = 0 debugging
================================================================

Requires GPU.  Runs short training experiments to verify the model can
learn *something* from the data and that gradients flow correctly.

Checks:
  2.1  Pure MSE Sanity — train with MSE only, does val MSE decrease?
  2.2  Overfitting Test — can the model memorize 50 images?
  2.3  Gradient Health — are gradients vanishing or exploding?
  2.4  Temperature Tracking — does the learned logit scale collapse?

Usage:
    python scripts/diagnostics/training_signal_diagnostic.py \
        --subject subj01 \
        [--config configs/experiments/B0v4_deterministic.yaml] \
        [--output outputs/diagnostics/training_report.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("training_diagnostic")

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fmri2img.models.unified_model import create_model
from fmri2img.eval.embedding_eval import compute_retrieval_metrics, compute_collapse_diagnostics


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers shared across checks
# ---------------------------------------------------------------------------

def _load_dataset(subject: str, config: Dict[str, Any]):
    """Load PreextractedNSDDataset matching train_unified.py logic."""
    primary = Path(f"data/indices/nsd_index/subject={subject}/index.parquet")
    legacy = primary.parent / "index_full.parquet"
    index_path = primary if primary.exists() else (legacy if legacy.exists() else None)
    if index_path is None:
        raise FileNotFoundError("No index file found")
    index_df = pd.read_parquet(index_path)

    emb_path = _find_embeddings_path()
    if emb_path is None:
        raise FileNotFoundError("No CLIP cache found")
    embeddings_df = pd.read_parquet(emb_path)

    cache_root = os.environ.get("CACHE_ROOT", "cache")
    feat_path = Path(cache_root) / "preextracted" / f"subject={subject}" / "fmri_features.npy"
    if not feat_path.exists():
        raise FileNotFoundError(f"Pre-extracted features not found: {feat_path}")

    # Inline PreextractedNSDDataset to avoid circular import issues
    features = np.load(feat_path, mmap_mode=None)
    index_df = index_df.reset_index(drop=True)

    if len(features) != len(index_df):
        raise ValueError(
            f"Feature rows ({len(features)}) != index rows ({len(index_df)})"
        )

    # Build embedding lookup
    if "nsdId" in embeddings_df.columns:
        emb_lookup = {int(r["nsdId"]): i for i, (_, r) in enumerate(embeddings_df.iterrows())}
    else:
        emb_lookup = {i: i for i in range(len(embeddings_df))}

    emb_col = None
    for col in ("final", "embedding", "clip_embedding", "clip512"):
        if col in embeddings_df.columns:
            emb_col = col
            break
    if emb_col is None:
        raise ValueError("No embedding column found")

    class _Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(features)
        def __getitem__(self, idx):
            fmri = torch.from_numpy(features[idx].astype(np.float32))
            nsd_id = int(index_df.iloc[idx]["nsdId"])
            emb_idx = emb_lookup.get(nsd_id)
            if emb_idx is None:
                raise KeyError(f"nsdId={nsd_id} not in CLIP cache")
            raw = embeddings_df.iloc[emb_idx][emb_col]
            emb = torch.from_numpy(np.asarray(raw, dtype=np.float32))
            return fmri, emb

    return _Dataset(), index_df, features


def _find_embeddings_path() -> Optional[Path]:
    candidates = [
        Path("cache/clip_embeddings/nsd_clipcache_multilayer.parquet"),
        Path("cache/clip_embeddings/nsd_clipvitl14.parquet"),
        Path("cache/clip_embeddings/embeddings_ViT-B-32.parquet"),
        Path("cache/clip_embeddings/text_clip.parquet"),
        Path("outputs/clip_cache/clip.parquet"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _make_split(index_df, config):
    """Image-level train/val split matching train_unified.py."""
    data_cfg = config.get("data", {})
    seed = data_cfg.get("seed", 42)
    val_frac = data_cfg.get("val_fraction", 0.1)
    exclude_shared = data_cfg.get("exclude_shared1000", True)

    _idx = index_df.reset_index(drop=True)
    pool_mask = np.ones(len(_idx), dtype=bool)
    if exclude_shared and "shared1000" in _idx.columns:
        pool_mask &= ~_idx["shared1000"].astype(bool).values
    pool_indices = np.where(pool_mask)[0]

    pool_nsd = _idx.iloc[pool_indices]["nsdId"].values
    unique_imgs = np.unique(pool_nsd)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_imgs)
    n_val = max(1, int(len(unique_imgs) * val_frac))
    val_set = set(unique_imgs[:n_val])
    train_set = set(unique_imgs[n_val:])

    train_idx = [i for i in pool_indices if _idx.iloc[i]["nsdId"] in train_set]
    val_idx = [i for i in pool_indices if _idx.iloc[i]["nsdId"] in val_set]
    return train_idx, val_idx


def _build_simple_model(fmri_dim: int, emb_dim: int, device: str) -> nn.Module:
    """Build a deterministic MLP model for diagnostic training."""
    config = {
        "type": "deterministic",
        "encoder": {
            "encoder_type": "mlp",
            "input_dim": fmri_dim,
            "hidden_dims": [4096, 2048],
            "use_residual": True,
            "dropout": 0.1,
            "activation": "gelu",
        },
        "decoder": {
            "output_dim": emb_dim,
            "hidden_dims": [1024],
        },
    }
    return create_model(config).to(device)


def _apply_zscore(features: np.ndarray, train_idx: List[int]):
    """Per-voxel z-scoring using training set statistics, in-place."""
    train_feats = features[train_idx]
    mean = train_feats.mean(axis=0)
    std = train_feats.std(axis=0)
    std[std < 1e-6] = 1.0
    features -= mean
    features /= std
    return mean, std


# ---------------------------------------------------------------------------
# Check 2.1 — Pure MSE Sanity
# ---------------------------------------------------------------------------

def check_2_1_pure_mse(
    dataset, index_df, features, config, device, n_epochs: int = 20
) -> CheckResult:
    """Train with MSE only — val MSE must decrease."""
    name = "2.1 Pure MSE Sanity"
    log.info("Running %s (%d epochs)...", name, n_epochs)

    train_idx, val_idx = _make_split(index_df, config)
    _apply_zscore(features, train_idx)

    fmri_dim = features.shape[1]
    sample_emb = dataset[train_idx[0]][1]
    emb_dim = sample_emb.shape[0]

    model = _build_simple_model(fmri_dim, emb_dim, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.MSELoss()

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=128, shuffle=False, num_workers=0)

    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for fmri, emb in train_loader:
            fmri, emb = fmri.to(device), emb.to(device)
            pred = model(fmri)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = criterion(pred, emb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / max(n_batches, 1))

        # Val
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for fmri, emb in val_loader:
                fmri, emb = fmri.to(device), emb.to(device)
                pred = model(fmri)
                if isinstance(pred, tuple):
                    pred = pred[0]
                val_loss += criterion(pred, emb).item()
                n_val += 1
        val_losses.append(val_loss / max(n_val, 1))
        log.info("  Epoch %2d: train_mse=%.6f  val_mse=%.6f", epoch, train_losses[-1], val_losses[-1])

    details = {
        "n_epochs": n_epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_mse_start": train_losses[0],
        "train_mse_end": train_losses[-1],
        "val_mse_start": val_losses[0],
        "val_mse_end": val_losses[-1],
        "val_mse_reduction": val_losses[0] - val_losses[-1],
    }

    reduction = val_losses[0] - val_losses[-1]
    if reduction < 0.001:
        return CheckResult(
            name, "FAIL",
            f"Val MSE barely decreased: {val_losses[0]:.6f} -> {val_losses[-1]:.6f} "
            f"(delta={reduction:.6f}). fMRI signal may be uninformative.",
            details,
        )
    return CheckResult(
        name, "PASS",
        f"Val MSE: {val_losses[0]:.6f} -> {val_losses[-1]:.6f} (delta={reduction:.6f})",
        details,
    )


# ---------------------------------------------------------------------------
# Check 2.2 — Overfitting Test
# ---------------------------------------------------------------------------

def check_2_2_overfit(
    dataset, index_df, features, config, device,
    n_images: int = 50, n_epochs: int = 300,
) -> CheckResult:
    """Can the model memorize a tiny subset?"""
    name = "2.2 Overfitting Test"
    log.info("Running %s (%d images, %d epochs)...", name, n_images, n_epochs)

    # Select a tiny subset: first n_images unique nsdIds
    _idx = index_df.reset_index(drop=True)
    unique = _idx["nsdId"].unique()[:n_images]
    tiny_idx = [i for i in range(len(_idx)) if _idx.iloc[i]["nsdId"] in set(unique)]
    # Deduplicate to one trial per image
    seen = set()
    deduped = []
    for i in tiny_idx:
        nid = int(_idx.iloc[i]["nsdId"])
        if nid not in seen:
            seen.add(nid)
            deduped.append(i)
    tiny_idx = deduped[:n_images]

    _apply_zscore(features, tiny_idx)

    fmri_dim = features.shape[1]
    sample_emb = dataset[tiny_idx[0]][1]
    emb_dim = sample_emb.shape[0]

    model = _build_simple_model(fmri_dim, emb_dim, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    criterion = nn.MSELoss()
    loader = DataLoader(Subset(dataset, tiny_idx), batch_size=len(tiny_idx), shuffle=True, num_workers=0)

    train_losses = []
    r1_history = []

    for epoch in range(n_epochs):
        model.train()
        for fmri, emb in loader:
            fmri, emb = fmri.to(device), emb.to(device)
            pred = model(fmri)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = criterion(pred, emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        if (epoch + 1) % 50 == 0 or epoch == 0:
            model.eval()
            all_preds, all_gts = [], []
            with torch.no_grad():
                for fmri, emb in loader:
                    fmri = fmri.to(device)
                    pred = model(fmri)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    all_preds.append(pred.cpu().numpy())
                    all_gts.append(emb.numpy())
            preds_np = np.concatenate(all_preds)
            gts_np = np.concatenate(all_gts)
            metrics = compute_retrieval_metrics(preds_np, gts_np, ks=(1, 5), normalize=True)
            r1 = metrics["top1_accuracy"]
            r1_history.append({"epoch": epoch + 1, "r1": r1, "mse": train_losses[-1]})
            log.info("  Epoch %3d: MSE=%.6f  R@1=%.2f%%", epoch + 1, train_losses[-1], r1 * 100)

    final_r1 = r1_history[-1]["r1"] if r1_history else 0.0
    details = {
        "n_images": len(tiny_idx),
        "n_epochs": n_epochs,
        "mse_start": train_losses[0],
        "mse_end": train_losses[-1],
        "r1_history": r1_history,
        "final_r1": final_r1,
    }

    if final_r1 < 0.5:
        return CheckResult(
            name, "FAIL",
            f"Cannot memorize {len(tiny_idx)} images: final R@1 = {final_r1*100:.1f}% "
            f"(expected >90%). Architecture or gradient issue.",
            details,
        )
    return CheckResult(
        name, "PASS",
        f"Memorized {len(tiny_idx)} images: R@1 = {final_r1*100:.1f}%",
        details,
    )


# ---------------------------------------------------------------------------
# Check 2.3 — Gradient Health
# ---------------------------------------------------------------------------

def check_2_3_gradient_health(
    dataset, index_df, features, config, device,
) -> CheckResult:
    """Log gradient norms per layer for 1 epoch."""
    name = "2.3 Gradient Health"
    log.info("Running %s...", name)

    train_idx, _ = _make_split(index_df, config)
    _apply_zscore(features, train_idx)

    fmri_dim = features.shape[1]
    sample_emb = dataset[train_idx[0]][1]
    emb_dim = sample_emb.shape[0]

    model = _build_simple_model(fmri_dim, emb_dim, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.MSELoss()
    loader = DataLoader(Subset(dataset, train_idx[:512]), batch_size=64, shuffle=True, num_workers=0)

    grad_norms: Dict[str, List[float]] = {}

    model.train()
    for fmri, emb in loader:
        fmri, emb = fmri.to(device), emb.to(device)
        pred = model(fmri)
        if isinstance(pred, tuple):
            pred = pred[0]
        loss = criterion(pred, emb)
        optimizer.zero_grad()
        loss.backward()

        for pname, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.norm(2).item()
                grad_norms.setdefault(pname, []).append(norm)

        optimizer.step()

    # Summarize
    summary = {}
    vanishing = []
    exploding = []
    for pname, norms in grad_norms.items():
        avg = float(np.mean(norms))
        mx = float(np.max(norms))
        summary[pname] = {"mean": avg, "max": mx}
        if avg < 1e-8:
            vanishing.append(pname)
        if mx > 100:
            exploding.append(pname)

    # Top-level stats
    all_means = [v["mean"] for v in summary.values()]
    overall_mean = float(np.mean(all_means)) if all_means else 0.0
    overall_max = float(np.max([v["max"] for v in summary.values()])) if summary else 0.0

    details = {
        "per_layer": summary,
        "overall_mean_grad_norm": overall_mean,
        "overall_max_grad_norm": overall_max,
        "vanishing_layers": vanishing,
        "exploding_layers": exploding,
    }

    if vanishing:
        return CheckResult(
            name, "FAIL",
            f"Vanishing gradients in {len(vanishing)} layers: {vanishing[:3]}...",
            details,
        )
    if exploding:
        return CheckResult(
            name, "WARN",
            f"Large gradients in {len(exploding)} layers (max={overall_max:.2f}): {exploding[:3]}...",
            details,
        )
    return CheckResult(
        name, "PASS",
        f"Gradients healthy: mean={overall_mean:.6f}, max={overall_max:.4f}",
        details,
    )


# ---------------------------------------------------------------------------
# Check 2.4 — Temperature Tracking
# ---------------------------------------------------------------------------

def check_2_4_temperature(
    dataset, index_df, features, config, device, n_epochs: int = 10,
) -> CheckResult:
    """Track learnable logit_scale during InfoNCE training."""
    name = "2.4 Temperature Tracking"
    log.info("Running %s (%d epochs)...", name, n_epochs)

    from fmri2img.losses.infonce_queue import InfoNCEQueueLoss

    train_idx, _ = _make_split(index_df, config)
    _apply_zscore(features, train_idx)

    fmri_dim = features.shape[1]
    sample_emb = dataset[train_idx[0]][1]
    emb_dim = sample_emb.shape[0]

    model = _build_simple_model(fmri_dim, emb_dim, device)
    infonce = InfoNCEQueueLoss(temperature=0.07, learnable_temperature=True, use_queue=False).to(device)

    all_params = list(model.parameters()) + list(infonce.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=1e-4, weight_decay=0.01)
    loader = DataLoader(Subset(dataset, train_idx[:2048]), batch_size=64, shuffle=True, num_workers=0)

    temp_history = []

    for epoch in range(n_epochs):
        model.train()
        for fmri, emb in loader:
            fmri, emb = fmri.to(device), emb.to(device)
            pred = model(fmri)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = infonce(pred, emb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

        logit_scale_val = infonce.logit_scale.item()
        temp_val = 1.0 / (np.exp(logit_scale_val) + 1e-10)
        temp_history.append({
            "epoch": epoch + 1,
            "logit_scale": logit_scale_val,
            "effective_temperature": temp_val,
        })
        log.info("  Epoch %2d: logit_scale=%.4f  temperature=%.6f", epoch + 1, logit_scale_val, temp_val)

    details = {
        "initial_logit_scale": np.log(1.0 / 0.07),
        "history": temp_history,
    }

    final_ls = temp_history[-1]["logit_scale"]
    if final_ls < 0.5:
        return CheckResult(
            name, "FAIL",
            f"Temperature collapsed: logit_scale={final_ls:.4f} (< 0.5). "
            f"Contrastive learning is ineffective.",
            details,
        )
    if final_ls >= 4.6:
        return CheckResult(
            name, "WARN",
            f"logit_scale hit ceiling: {final_ls:.4f} (clamp=4.6). "
            f"Temperature may be too sharp.",
            details,
        )
    return CheckResult(
        name, "PASS",
        f"logit_scale={final_ls:.4f}, temperature={temp_history[-1]['effective_temperature']:.6f}",
        details,
    )


# ---------------------------------------------------------------------------
# Check 2.5 — Post-training collapse detection (bonus)
# ---------------------------------------------------------------------------

def check_2_5_collapse_after_training(
    dataset, index_df, features, config, device, n_epochs: int = 10,
) -> CheckResult:
    """Train briefly and check if predictions collapse."""
    name = "2.5 Post-Training Collapse Detection"
    log.info("Running %s (%d epochs)...", name, n_epochs)

    train_idx, val_idx = _make_split(index_df, config)
    _apply_zscore(features, train_idx)

    fmri_dim = features.shape[1]
    sample_emb = dataset[train_idx[0]][1]
    emb_dim = sample_emb.shape[0]

    model = _build_simple_model(fmri_dim, emb_dim, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.MSELoss()
    train_loader = DataLoader(Subset(dataset, train_idx[:4096]), batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx[:512]), batch_size=128, shuffle=False, num_workers=0)

    for epoch in range(n_epochs):
        model.train()
        for fmri, emb in train_loader:
            fmri, emb = fmri.to(device), emb.to(device)
            pred = model(fmri)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = criterion(pred, emb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for fmri, emb in val_loader:
            fmri = fmri.to(device)
            pred = model(fmri)
            if isinstance(pred, tuple):
                pred = pred[0]
            all_preds.append(pred.cpu().numpy())
            all_gts.append(emb.numpy())
    preds_np = np.concatenate(all_preds)
    gts_np = np.concatenate(all_gts)

    diag = compute_collapse_diagnostics(preds_np, gts_np, normalize=True)
    metrics = compute_retrieval_metrics(preds_np, gts_np, ks=(1, 5, 10), normalize=True)

    details = {
        "n_epochs": n_epochs,
        "collapse": diag,
        "retrieval": metrics,
    }

    log.info("  After %d epochs: R@1=%.2f%%, avg_pairwise_cos=%.4f, collapse_ratio=%.3f",
             n_epochs, metrics["top1_accuracy"] * 100, diag["avg_pairwise_sim"], diag["collapse_ratio"])

    if diag["avg_pairwise_sim"] > 0.95:
        return CheckResult(name, "FAIL", f"Predictions collapsed after {n_epochs} epochs", details)
    if diag["avg_pairwise_sim"] > 0.85:
        return CheckResult(name, "WARN", f"Near-collapse after {n_epochs} epochs", details)
    return CheckResult(
        name, "PASS",
        f"R@1={metrics['top1_accuracy']*100:.2f}%, avg_cos={diag['avg_pairwise_sim']:.4f}",
        details,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Training signal diagnostic (Tier 2)")
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--config", default=None, help="Experiment YAML")
    parser.add_argument("--output", default="outputs/diagnostics/training_report.json")
    parser.add_argument("--device", default=None, help="cuda / cpu (auto-detect)")
    parser.add_argument("--skip-overfit", action="store_true", help="Skip the slow overfitting test")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=" * 70)
    log.info("TRAINING SIGNAL DIAGNOSTIC — Tier 2")
    log.info("=" * 70)
    log.info("Subject: %s  Device: %s", args.subject, device)

    if args.config and Path(args.config).exists():
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "data": {"seed": 42, "val_fraction": 0.1, "exclude_shared1000": True,
                     "average_repetitions": False},
            "training": {"batch_size": 64},
        }

    dataset, index_df, features = _load_dataset(args.subject, config)
    log.info("Dataset: %d trials, %d voxels", len(dataset), features.shape[1])

    results: List[CheckResult] = []

    # -- 2.1 Pure MSE --
    # Reload features fresh for each check (z-scoring is in-place)
    feat_backup = features.copy()

    r = check_2_1_pure_mse(dataset, index_df, features, config, device)
    results.append(r)
    log.info("[%s] %s: %s", r.status, r.name, r.message)
    np.copyto(features, feat_backup)

    # -- 2.2 Overfit --
    if not args.skip_overfit:
        r = check_2_2_overfit(dataset, index_df, features, config, device)
        results.append(r)
        log.info("[%s] %s: %s", r.status, r.name, r.message)
        np.copyto(features, feat_backup)
    else:
        results.append(CheckResult("2.2 Overfitting Test", "SKIP", "Skipped via --skip-overfit", {}))

    # -- 2.3 Gradient Health --
    r = check_2_3_gradient_health(dataset, index_df, features, config, device)
    results.append(r)
    log.info("[%s] %s: %s", r.status, r.name, r.message)
    np.copyto(features, feat_backup)

    # -- 2.4 Temperature --
    r = check_2_4_temperature(dataset, index_df, features, config, device)
    results.append(r)
    log.info("[%s] %s: %s", r.status, r.name, r.message)
    np.copyto(features, feat_backup)

    # -- 2.5 Collapse after training --
    r = check_2_5_collapse_after_training(dataset, index_df, features, config, device)
    results.append(r)
    log.info("[%s] %s: %s", r.status, r.name, r.message)

    # -- Summary --
    log.info("\n" + "=" * 70)
    log.info("TRAINING DIAGNOSTIC SUMMARY")
    log.info("=" * 70)

    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_warn = sum(1 for r in results if r.status == "WARN")
    n_pass = sum(1 for r in results if r.status == "PASS")

    for r in results:
        tag = {"PASS": "OK ", "FAIL": "!! ", "WARN": "?? ", "SKIP": "-- "}
        log.info("  [%s] %s", tag.get(r.status, "  ") + r.status, r.name)
        if r.status in ("FAIL", "WARN"):
            log.info("         %s", r.message)

    log.info("\nTotals: %d PASS, %d FAIL, %d WARN", n_pass, n_fail, n_warn)

    # Save report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "subject": args.subject,
        "device": device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {"pass": n_pass, "fail": n_fail, "warn": n_warn},
        "checks": [
            {"name": r.name, "status": r.status, "message": r.message, "details": r.details}
            for r in results
        ],
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Report saved to %s", out_path)


if __name__ == "__main__":
    main()
