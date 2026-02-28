#!/usr/bin/env python3
"""
Pipeline Diagnostic: Systematic R@1 = 0 Root Cause Analysis
============================================================

Runs a sequence of data-integrity and embedding-space checks to identify
why all models produce near-chance retrieval.  Designed to execute on
JupyterHub with real NSD data -- no GPU required for Tier 1 checks.

Usage:
    python scripts/diagnostics/pipeline_diagnostic.py \
        --subject subj01 \
        [--config configs/experiments/B0v4_deterministic.yaml] \
        [--output outputs/diagnostics/pipeline_report.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("pipeline_diagnostic")

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    status: str          # PASS, FAIL, WARN, SKIP
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tier 1 — Data Integrity
# ---------------------------------------------------------------------------

def check_1_1_index_source(subject: str) -> CheckResult:
    """Verify which index file exists and whether it was built correctly."""
    name = "1.1 Index Source Verification"

    primary = Path(f"data/indices/nsd_index/subject={subject}/index.parquet")
    legacy = primary.parent / "index_full.parquet"

    details: Dict[str, Any] = {
        "primary_exists": primary.exists(),
        "legacy_exists": legacy.exists(),
    }

    if primary.exists():
        index_path = primary
        details["used_file"] = str(primary)
    elif legacy.exists():
        index_path = legacy
        details["used_file"] = str(legacy)
        log.warning("Primary index.parquet missing — training would fall back to index_full.parquet!")
    else:
        return CheckResult(name, "FAIL", "No index file found at all", details)

    df = pd.read_parquet(index_path)
    details["n_rows"] = len(df)
    details["n_unique_nsdId"] = int(df["nsdId"].nunique())
    details["has_session_col"] = "session" in df.columns

    # Detect the buggy cycling pattern from build_full_subj01_index.py:
    # In that script, early rows have nsdIds = sorted unique list cycled by
    # global_trial_idx, so the first ~N_unique rows are monotonically sorted.
    first_ids = df["nsdId"].values[:200]
    is_monotonic = bool(np.all(np.diff(first_ids) >= 0))
    has_repeats_early = len(set(first_ids)) < len(first_ids)
    details["first_200_nsdIds_monotonic"] = is_monotonic
    details["first_200_nsdIds_has_repeats"] = has_repeats_early

    # A behaviorally-derived index will have repeats early (same stimulus
    # shown multiple times across sessions) and non-monotonic nsdIds (trials
    # are ordered by session/run, not by stimulus ID).
    if is_monotonic and not has_repeats_early:
        return CheckResult(
            name, "FAIL",
            f"Index appears built by buggy cycling script (first 200 nsdIds are "
            f"monotonically sorted with no repeats). File: {index_path}",
            details,
        )

    if not details["has_session_col"]:
        return CheckResult(
            name, "WARN",
            f"Index is missing 'session' column — per-session z-scoring will "
            f"fall back to global mode. File: {index_path}",
            details,
        )

    status = "PASS"
    msg = (
        f"Using {index_path.name}: {len(df)} trials, "
        f"{df['nsdId'].nunique()} unique stimuli, session column present"
    )
    if not primary.exists():
        status = "WARN"
        msg = f"WARN: using legacy fallback ({legacy.name}). " + msg

    return CheckResult(name, status, msg, details)


def check_1_2_oracle_retrieval(
    subject: str,
    embeddings_df: pd.DataFrame,
    index_df: pd.DataFrame,
    features_path: Optional[Path],
    config: Dict[str, Any],
) -> CheckResult:
    """Oracle test: use GT embeddings as predictions — R@1 must be 100 %."""
    from fmri2img.eval.embedding_eval import compute_retrieval_metrics

    name = "1.2 Oracle Retrieval Test"

    # We only need the CLIP side — no fMRI features required.
    # Reconstruct the same split the training script would use.
    data_cfg = config.get("data", {})
    seed = data_cfg.get("seed", 42)
    val_frac = data_cfg.get("val_fraction", 0.1)
    exclude_shared = data_cfg.get("exclude_shared1000", True)

    # Build embedding lookup
    if "nsdId" in embeddings_df.columns:
        emb_lookup = {
            int(row["nsdId"]): i
            for i, (_, row) in enumerate(embeddings_df.iterrows())
        }
    else:
        emb_lookup = {i: i for i in range(len(embeddings_df))}

    # Determine embedding column
    emb_col = None
    for col in ("final", "embedding", "clip_embedding", "clip512"):
        if col in embeddings_df.columns:
            emb_col = col
            break
    if emb_col is None:
        return CheckResult(name, "SKIP", "Cannot determine embedding column", {})

    # Pool indices (exclude shared1000 if configured)
    _idx = index_df.reset_index(drop=True)
    pool_mask = np.ones(len(_idx), dtype=bool)
    if exclude_shared and "shared1000" in _idx.columns:
        pool_mask &= ~_idx["shared1000"].astype(bool).values

    pool_indices = np.where(pool_mask)[0]

    # Image-level split
    pool_nsd_ids = _idx.iloc[pool_indices]["nsdId"].values
    unique_images = np.unique(pool_nsd_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_images)
    n_val = max(1, int(len(unique_images) * val_frac))
    val_image_set = set(unique_images[:n_val])
    val_indices = [i for i in pool_indices if _idx.iloc[i]["nsdId"] in val_image_set]

    if len(val_indices) < 2:
        return CheckResult(name, "SKIP", f"Only {len(val_indices)} val trials", {})

    # If average_repetitions is set, average per unique nsdId
    avg_reps = data_cfg.get("average_repetitions", False)

    # Collect GT embeddings for val set
    val_embeddings = []
    val_nsd_ids_used = []
    seen_nsd = set()

    for vi in val_indices:
        nsd_id = int(_idx.iloc[vi]["nsdId"])
        if avg_reps and nsd_id in seen_nsd:
            continue
        seen_nsd.add(nsd_id)

        emb_idx = emb_lookup.get(nsd_id)
        if emb_idx is None:
            continue
        raw = embeddings_df.iloc[emb_idx][emb_col]
        val_embeddings.append(np.asarray(raw, dtype=np.float32))
        val_nsd_ids_used.append(nsd_id)

    gt = np.stack(val_embeddings)
    metrics = compute_retrieval_metrics(gt, gt, ks=(1, 5, 10), normalize=True)

    details = {
        "n_val_images": len(gt),
        "metrics": metrics,
        "embedding_dim": gt.shape[1],
    }

    r1 = metrics["top1_accuracy"]
    if r1 < 0.999:
        return CheckResult(
            name, "FAIL",
            f"Oracle R@1 = {r1*100:.2f}% (expected 100%). "
            f"Data alignment is broken!",
            details,
        )
    return CheckResult(
        name, "PASS",
        f"Oracle R@1 = {r1*100:.1f}% on {len(gt)} val images — alignment OK",
        details,
    )


def check_1_3_spot_check(
    subject: str,
    embeddings_df: pd.DataFrame,
    index_df: pd.DataFrame,
    n_checks: int = 5,
) -> CheckResult:
    """Load images on-the-fly and compare fresh CLIP embeddings to cache."""
    name = "1.3 fMRI-CLIP Spot Check"

    try:
        from fmri2img.utils.clip_utils import load_clip_model
    except ImportError:
        return CheckResult(name, "SKIP", "open_clip / clip_utils not importable", {})

    # Determine embedding column
    emb_col = None
    for col in ("final", "embedding", "clip_embedding", "clip512"):
        if col in embeddings_df.columns:
            emb_col = col
            break
    if emb_col is None:
        return CheckResult(name, "SKIP", "Cannot determine embedding column", {})

    # Build embedding lookup
    if "nsdId" in embeddings_df.columns:
        emb_lookup = {
            int(row["nsdId"]): i
            for i, (_, row) in enumerate(embeddings_df.iterrows())
        }
    else:
        emb_lookup = {i: i for i in range(len(embeddings_df))}

    # Try to locate the HDF5 stimulus file
    import h5py
    hdf5_path = os.getenv("NSD_HDF5", "cache/nsd_hdf5/nsd_stimuli.hdf5")
    if not Path(hdf5_path).exists():
        # Try NSD_DATA_ROOT
        nsd_root = os.environ.get("NSD_DATA_ROOT", "")
        alt = Path(nsd_root) / "nsddata_stimuli" / "stimuli" / "nsd" / "nsd_stimuli.hdf5"
        if alt.exists():
            hdf5_path = str(alt)
        else:
            return CheckResult(
                name, "SKIP",
                f"HDF5 stimuli not found at {hdf5_path} or {alt}",
                {},
            )

    import torch
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, clip_cfg = load_clip_model(device=device)
    model.eval()

    _idx = index_df.reset_index(drop=True)
    rng = np.random.default_rng(42)
    check_indices = rng.choice(len(_idx), size=min(n_checks, len(_idx)), replace=False)

    cosines = []
    failures = []

    with h5py.File(hdf5_path, "r") as hf:
        ds = hf["imgBrick"]
        log.info("imgBrick shape: %s", ds.shape)

        for ci in check_indices:
            nsd_id = int(_idx.iloc[ci]["nsdId"])
            emb_idx = emb_lookup.get(nsd_id)
            if emb_idx is None:
                failures.append({"trial": int(ci), "nsdId": nsd_id, "error": "not in cache"})
                continue

            cached = np.asarray(
                embeddings_df.iloc[emb_idx][emb_col], dtype=np.float32
            )

            # Load image and compute fresh CLIP embedding
            arr = ds[nsd_id]
            img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
            img_t = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad(), torch.amp.autocast(device, enabled=(device == "cuda")):
                fresh = model.encode_image(img_t)
                fresh = fresh / fresh.norm(dim=-1, keepdim=True)
            fresh_np = fresh.cpu().numpy().flatten().astype(np.float32)

            cos = float(np.dot(cached, fresh_np) / (
                np.linalg.norm(cached) * np.linalg.norm(fresh_np) + 1e-10
            ))
            cosines.append(cos)
            log.info("  trial=%d  nsdId=%d  cos(cached, fresh)=%.6f", ci, nsd_id, cos)

    details = {
        "n_checked": len(cosines),
        "cosines": cosines,
        "failures": failures,
        "min_cosine": float(min(cosines)) if cosines else None,
        "max_cosine": float(max(cosines)) if cosines else None,
    }

    if not cosines:
        return CheckResult(name, "SKIP", "No spot checks completed", details)

    min_cos = min(cosines)
    if min_cos < 0.99:
        return CheckResult(
            name, "FAIL",
            f"Spot check FAILED: min cosine = {min_cos:.6f} (expected > 0.99). "
            f"CLIP cache may not match the images.",
            details,
        )
    return CheckResult(
        name, "PASS",
        f"{len(cosines)}/{n_checks} checks passed, min cosine = {min_cos:.6f}",
        details,
    )


def check_1_4_clip_cache(embeddings_df: pd.DataFrame, emb_path: Optional[Path]) -> CheckResult:
    """Verify CLIP cache dimensionality, norms, and source path."""
    name = "1.4 CLIP Cache Verification"

    details: Dict[str, Any] = {"path": str(emb_path) if emb_path else "None"}

    if emb_path is None:
        return CheckResult(name, "FAIL", "No CLIP cache found by find_embeddings_path()", details)

    # Determine which column is used
    emb_col = None
    for col in ("final", "embedding", "clip_embedding", "clip512"):
        if col in embeddings_df.columns:
            emb_col = col
            break
    details["embedding_column"] = emb_col

    if emb_col is None:
        return CheckResult(name, "FAIL", "No recognizable embedding column in cache", details)

    # Sample to check dimension + norms
    sample = np.asarray(embeddings_df.iloc[0][emb_col], dtype=np.float32)
    dim = sample.shape[0]
    details["embedding_dim"] = dim

    # Check norms for first 100 rows
    norms = []
    for i in range(min(100, len(embeddings_df))):
        v = np.asarray(embeddings_df.iloc[i][emb_col], dtype=np.float32)
        norms.append(float(np.linalg.norm(v)))
    norms_arr = np.array(norms)
    details["norm_mean"] = float(norms_arr.mean())
    details["norm_std"] = float(norms_arr.std())
    details["norm_min"] = float(norms_arr.min())
    details["norm_max"] = float(norms_arr.max())
    details["n_embeddings"] = len(embeddings_df)

    has_nsdId = "nsdId" in embeddings_df.columns or "nsd_id" in embeddings_df.columns
    details["has_nsdId_column"] = has_nsdId

    issues = []
    if dim == 512:
        issues.append(f"Dimension is 512 — likely ViT-B/32 cache, but model expects 768 (ViT-L/14)")
    elif dim != 768:
        issues.append(f"Unexpected dimension {dim} (expected 768 for ViT-L/14)")

    if norms_arr.std() > 0.01:
        issues.append(f"Norms are not uniform (std={norms_arr.std():.4f}) — embeddings may not be L2-normalized")

    if abs(norms_arr.mean() - 1.0) > 0.01:
        issues.append(f"Mean norm = {norms_arr.mean():.4f}, expected 1.0")

    if not has_nsdId:
        issues.append("Missing nsdId column — lookup will use positional indexing (fragile)")

    # Check if path suggests a stale/wrong cache
    path_str = str(emb_path)
    if "ViT-B-32" in path_str or "vit-b-32" in path_str.lower():
        issues.append(f"Path contains 'ViT-B-32' — wrong CLIP model")
    if "text_clip" in path_str:
        issues.append("Path suggests text embeddings, not image embeddings")

    if issues:
        return CheckResult(name, "FAIL", "; ".join(issues), details)

    return CheckResult(
        name, "PASS",
        f"{dim}-D embeddings, norm=1.0, {len(embeddings_df)} entries from {emb_path.name}",
        details,
    )


def check_1_5_duplicate_negatives(
    index_df: pd.DataFrame,
    config: Dict[str, Any],
) -> CheckResult:
    """Count how often the same nsdId appears multiple times in a batch."""
    name = "1.5 Duplicate Negatives in Batch"

    data_cfg = config.get("data", {})
    batch_size = config.get("training", {}).get("batch_size", 64)
    seed = data_cfg.get("seed", 42)
    val_frac = data_cfg.get("val_fraction", 0.1)
    exclude_shared = data_cfg.get("exclude_shared1000", True)
    avg_reps = data_cfg.get("average_repetitions", False)

    _idx = index_df.reset_index(drop=True)

    # Build pool (exclude shared1000)
    pool_mask = np.ones(len(_idx), dtype=bool)
    if exclude_shared and "shared1000" in _idx.columns:
        pool_mask &= ~_idx["shared1000"].astype(bool).values

    pool_indices = np.where(pool_mask)[0]
    pool_nsd_ids = _idx.iloc[pool_indices]["nsdId"].values
    unique_images = np.unique(pool_nsd_ids)

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_images)
    n_val = max(1, int(len(unique_images) * val_frac))
    train_image_set = set(unique_images[n_val:])
    train_indices = np.array([i for i in pool_indices if _idx.iloc[i]["nsdId"] in train_image_set])

    if avg_reps:
        return CheckResult(name, "PASS", "average_repetitions=True, one sample per image — no duplicates", {})

    # Simulate random batches (DataLoader with shuffle)
    rng2 = np.random.default_rng(0)
    n_batches = min(500, len(train_indices) // batch_size)
    if n_batches == 0:
        return CheckResult(name, "SKIP", "Not enough data for even 1 batch", {})

    shuffled = train_indices.copy()
    rng2.shuffle(shuffled)

    batches_with_dup = 0
    total_dup_pairs = 0
    max_dups_in_batch = 0

    for b in range(n_batches):
        batch_idx = shuffled[b * batch_size : (b + 1) * batch_size]
        batch_nsd = _idx.iloc[batch_idx]["nsdId"].values
        counts = Counter(batch_nsd)
        n_dup = sum(c - 1 for c in counts.values() if c > 1)
        if n_dup > 0:
            batches_with_dup += 1
            total_dup_pairs += n_dup
            max_dups_in_batch = max(max_dups_in_batch, n_dup)

    frac_dup = batches_with_dup / n_batches
    avg_dup = total_dup_pairs / n_batches

    details = {
        "batch_size": batch_size,
        "n_batches_simulated": n_batches,
        "n_train_trials": len(train_indices),
        "n_train_images": len(train_image_set),
        "trials_per_image_avg": len(train_indices) / len(train_image_set) if train_image_set else 0,
        "batches_with_duplicates": batches_with_dup,
        "fraction_batches_with_dup": frac_dup,
        "avg_duplicate_pairs_per_batch": avg_dup,
        "max_duplicates_in_one_batch": max_dups_in_batch,
    }

    msg = (
        f"{frac_dup*100:.1f}% of batches have duplicate nsdIds "
        f"(avg {avg_dup:.1f} dup pairs/batch, max {max_dups_in_batch})"
    )

    if frac_dup > 0.5:
        return CheckResult(
            name, "WARN",
            msg + " — InfoNCE treats duplicates as false negatives, which hurts learning",
            details,
        )
    return CheckResult(name, "PASS" if frac_dup < 0.05 else "INFO", msg, details)


# ---------------------------------------------------------------------------
# Tier 3 — Embedding Space
# ---------------------------------------------------------------------------

def check_3_1_collapse_detection(
    embeddings_df: pd.DataFrame,
    index_df: pd.DataFrame,
    config: Dict[str, Any],
    checkpoint_path: Optional[Path] = None,
) -> CheckResult:
    """Check for mode collapse in model predictions (requires a checkpoint)."""
    name = "3.1 Prediction Collapse Detection"

    if checkpoint_path is None or not checkpoint_path.exists():
        # Fall back: analyze GT embedding space properties
        emb_col = None
        for col in ("final", "embedding", "clip_embedding", "clip512"):
            if col in embeddings_df.columns:
                emb_col = col
                break
        if emb_col is None:
            return CheckResult(name, "SKIP", "No embedding column + no checkpoint", {})

        n_sample = min(500, len(embeddings_df))
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(embeddings_df), size=n_sample, replace=False)
        gt = np.stack([
            np.asarray(embeddings_df.iloc[i][emb_col], dtype=np.float32)
            for i in sample_idx
        ])
        norms = np.linalg.norm(gt, axis=1, keepdims=True)
        gt = gt / np.maximum(norms, 1e-8)

        sim_matrix = gt @ gt.T
        mask = ~np.eye(n_sample, dtype=bool)
        avg_cos = float(sim_matrix[mask].mean())
        std_per_dim = float(gt.std(axis=0).mean())

        # Effective rank via SVD
        U, S, Vt = np.linalg.svd(gt, full_matrices=False)
        p = S / S.sum()
        eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))

        details = {
            "mode": "gt_only_no_checkpoint",
            "n_samples": n_sample,
            "gt_avg_pairwise_cosine": avg_cos,
            "gt_std_per_dim": std_per_dim,
            "gt_effective_rank": eff_rank,
        }

        return CheckResult(
            name, "INFO",
            f"No checkpoint — GT space analysis: avg_cos={avg_cos:.4f}, "
            f"std/dim={std_per_dim:.4f}, eff_rank={eff_rank:.1f}. "
            f"Re-run with --checkpoint to check predictions.",
            details,
        )

    # With checkpoint: load model, run val predictions, compute collapse metrics
    import torch
    try:
        from fmri2img.eval.embedding_eval import compute_collapse_diagnostics
    except ImportError:
        return CheckResult(name, "SKIP", "Cannot import collapse diagnostics", {})

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Try to extract predictions from checkpoint if stored
    if "val_predictions" in ckpt and "val_ground_truth" in ckpt:
        preds = np.array(ckpt["val_predictions"])
        gts = np.array(ckpt["val_ground_truth"])
    else:
        return CheckResult(
            name, "SKIP",
            "Checkpoint does not contain val_predictions/val_ground_truth. "
            "Run training_signal_diagnostic.py for live collapse analysis.",
            {"checkpoint": str(checkpoint_path)},
        )

    diag = compute_collapse_diagnostics(preds, gts, normalize=True)
    details = {
        "checkpoint": str(checkpoint_path),
        **diag,
    }

    if diag["avg_pairwise_sim"] > 0.95:
        return CheckResult(
            name, "FAIL",
            f"SEVERE mode collapse: avg pairwise cosine = {diag['avg_pairwise_sim']:.4f}",
            details,
        )
    if diag["avg_pairwise_sim"] > 0.90:
        return CheckResult(
            name, "WARN",
            f"Likely collapse: avg pairwise cosine = {diag['avg_pairwise_sim']:.4f}",
            details,
        )
    return CheckResult(
        name, "PASS",
        f"avg_pairwise_cos = {diag['avg_pairwise_sim']:.4f}, "
        f"collapse_ratio = {diag['collapse_ratio']:.3f}",
        details,
    )


def check_3_2_cosine_distribution(
    embeddings_df: pd.DataFrame,
    index_df: pd.DataFrame,
    config: Dict[str, Any],
    checkpoint_path: Optional[Path] = None,
) -> CheckResult:
    """Analyze positive vs negative cosine distributions."""
    name = "3.2 Cosine Distribution Analysis"

    if checkpoint_path is None or not checkpoint_path.exists():
        return CheckResult(
            name, "SKIP",
            "No checkpoint provided. Run training_signal_diagnostic.py for live analysis.",
            {},
        )

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "val_predictions" not in ckpt or "val_ground_truth" not in ckpt:
        return CheckResult(name, "SKIP", "Checkpoint lacks val predictions", {})

    preds = np.array(ckpt["val_predictions"])
    gts = np.array(ckpt["val_ground_truth"])

    # Normalize
    preds = preds / (np.linalg.norm(preds, axis=1, keepdims=True) + 1e-8)
    gts = gts / (np.linalg.norm(gts, axis=1, keepdims=True) + 1e-8)

    # Positive cosines: cos(pred[i], gt[i])
    pos_cos = np.sum(preds * gts, axis=1)

    # Negative cosines: cos(pred[i], gt[j]) for random j != i
    rng = np.random.default_rng(42)
    n = len(preds)
    neg_cos = []
    for i in range(n):
        j = rng.integers(0, n - 1)
        if j >= i:
            j += 1
        neg_cos.append(float(np.dot(preds[i], gts[j])))
    neg_cos = np.array(neg_cos)

    separation = float(pos_cos.mean() - neg_cos.mean())

    details = {
        "n_samples": n,
        "pos_cosine_mean": float(pos_cos.mean()),
        "pos_cosine_std": float(pos_cos.std()),
        "neg_cosine_mean": float(neg_cos.mean()),
        "neg_cosine_std": float(neg_cos.std()),
        "separation": separation,
    }

    if separation < 0.01:
        return CheckResult(
            name, "FAIL",
            f"No separation: pos_cos={pos_cos.mean():.4f}, neg_cos={neg_cos.mean():.4f}, "
            f"gap={separation:.4f}",
            details,
        )
    return CheckResult(
        name, "PASS",
        f"pos_cos={pos_cos.mean():.4f}, neg_cos={neg_cos.mean():.4f}, "
        f"separation={separation:.4f}",
        details,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def load_default_config() -> Dict[str, Any]:
    """Minimal config mirroring v4 defaults for split logic."""
    return {
        "data": {
            "seed": 42,
            "val_fraction": 0.1,
            "exclude_shared1000": True,
            "average_repetitions": False,
        },
        "training": {"batch_size": 64},
    }


def find_embeddings_path() -> Optional[Path]:
    """Mirror the logic in train_unified.py."""
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


def main():
    parser = argparse.ArgumentParser(description="Pipeline diagnostic for R@1 = 0 debugging")
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--config", default=None, help="Experiment YAML (for split params)")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint for Tier 3")
    parser.add_argument("--output", default="outputs/diagnostics/pipeline_report.json")
    parser.add_argument("--skip-spot-check", action="store_true",
                        help="Skip Check 1.3 (requires HDF5 + CLIP model)")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("PIPELINE DIAGNOSTIC — R@1 Root Cause Analysis")
    log.info("=" * 70)
    log.info("Subject: %s", args.subject)

    # Load config
    if args.config and Path(args.config).exists():
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        log.info("Config: %s", args.config)
    else:
        config = load_default_config()
        log.info("Config: defaults (no YAML provided)")

    # Load embeddings
    emb_path = find_embeddings_path()
    if emb_path is not None:
        log.info("CLIP cache: %s", emb_path)
        embeddings_df = pd.read_parquet(emb_path)
    else:
        log.error("No CLIP cache found!")
        embeddings_df = pd.DataFrame()

    # Load index
    primary = Path(f"data/indices/nsd_index/subject={args.subject}/index.parquet")
    legacy = primary.parent / "index_full.parquet"
    if primary.exists():
        index_df = pd.read_parquet(primary)
    elif legacy.exists():
        index_df = pd.read_parquet(legacy)
    else:
        log.error("No index file found")
        index_df = pd.DataFrame()

    features_path = Path(
        os.environ.get("CACHE_ROOT", "cache")
    ) / "preextracted" / f"subject={args.subject}" / "fmri_features.npy"

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None

    # -----------------------------------------------------------------------
    # Run checks
    # -----------------------------------------------------------------------
    results: List[CheckResult] = []

    log.info("\n" + "=" * 70)
    log.info("TIER 1: DATA INTEGRITY")
    log.info("=" * 70)

    # 1.1
    r = check_1_1_index_source(args.subject)
    results.append(r)
    log.info("[%s] %s: %s", r.status, r.name, r.message)

    # 1.2
    if len(embeddings_df) > 0 and len(index_df) > 0:
        r = check_1_2_oracle_retrieval(
            args.subject, embeddings_df, index_df, features_path, config,
        )
    else:
        r = CheckResult("1.2 Oracle Retrieval Test", "SKIP", "Missing embeddings or index", {})
    results.append(r)
    log.info("[%s] %s: %s", r.status, r.name, r.message)

    # 1.3
    if args.skip_spot_check:
        r = CheckResult("1.3 fMRI-CLIP Spot Check", "SKIP", "Skipped via --skip-spot-check", {})
    elif len(embeddings_df) > 0 and len(index_df) > 0:
        r = check_1_3_spot_check(args.subject, embeddings_df, index_df)
    else:
        r = CheckResult("1.3 fMRI-CLIP Spot Check", "SKIP", "Missing data", {})
    results.append(r)
    log.info("[%s] %s: %s", r.status, r.name, r.message)

    # 1.4
    r = check_1_4_clip_cache(embeddings_df, emb_path)
    results.append(r)
    log.info("[%s] %s: %s", r.status, r.name, r.message)

    # 1.5
    if len(index_df) > 0:
        r = check_1_5_duplicate_negatives(index_df, config)
    else:
        r = CheckResult("1.5 Duplicate Negatives", "SKIP", "No index", {})
    results.append(r)
    log.info("[%s] %s: %s", r.status, r.name, r.message)

    log.info("\n" + "=" * 70)
    log.info("TIER 3: EMBEDDING SPACE")
    log.info("=" * 70)

    # 3.1
    r = check_3_1_collapse_detection(embeddings_df, index_df, config, checkpoint_path)
    results.append(r)
    log.info("[%s] %s: %s", r.status, r.name, r.message)

    # 3.2
    r = check_3_2_cosine_distribution(embeddings_df, index_df, config, checkpoint_path)
    results.append(r)
    log.info("[%s] %s: %s", r.status, r.name, r.message)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("DIAGNOSTIC SUMMARY")
    log.info("=" * 70)

    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_warn = sum(1 for r in results if r.status == "WARN")
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_skip = sum(1 for r in results if r.status in ("SKIP", "INFO"))

    for r in results:
        tag = {"PASS": "OK ", "FAIL": "!! ", "WARN": "?? ", "SKIP": "-- ", "INFO": "ii "}
        log.info("  [%s] %s", tag.get(r.status, "  ") + r.status, r.name)
        if r.status in ("FAIL", "WARN"):
            log.info("         %s", r.message)

    log.info("")
    log.info("Totals: %d PASS, %d FAIL, %d WARN, %d SKIP/INFO", n_pass, n_fail, n_warn, n_skip)

    if n_fail > 0:
        log.info("\nACTION REQUIRED: Fix the FAIL items above before running experiments.")
    elif n_warn > 0:
        log.info("\nWARNINGS present — review before running experiments.")
    else:
        log.info("\nAll checks passed. If R@1 is still ~0, run training_signal_diagnostic.py (Tier 2).")

    # Save JSON report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "subject": args.subject,
        "config": args.config,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {"pass": n_pass, "fail": n_fail, "warn": n_warn, "skip": n_skip},
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
