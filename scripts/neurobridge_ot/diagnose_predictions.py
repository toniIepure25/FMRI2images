#!/usr/bin/env python3
"""Prediction collapse diagnostics for NeuroBridge-OT.

Loads a checkpoint and analyzes predictions for signs of embedding collapse:
- Uniform predictions (all outputs identical)
- Low variance in predictions
- Hubness (few predictions dominate nearest neighbors)
- Norm collapse (all predictions same norm)
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def diagnose_predictions(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    subjects: List[str] = None,
    n_samples: int = 2000,
    output_dir: str = "experimental_results/neurobridge_ot_debug/collapse_diagnostics",
    split: str = "val",
):
    """Run prediction collapse diagnostics."""
    import yaml
    from fmri2img.data.neurobridge_dataset import NeuroBridgeDataset, neurobridge_collate
    from fmri2img.models.neurobridge_ot.model import NeuroBridgeOTModel

    if subjects is None:
        subjects = ["subj01"]

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    import pandas as pd
    clip_path = Path(os.environ.get("CLIP_CACHE", "outputs/clip_cache/clip_multilayer.parquet"))
    if not clip_path.exists():
        clip_path = Path("outputs/clip_cache/clip.parquet")
    clip_df = pd.read_parquet(clip_path)
    embedding_col = config.get("data", {}).get("embedding_column", "layer_18_proj")

    cache_root = Path(os.environ.get("CACHE_ROOT", "cache")) / "preextracted"
    index_root = Path("data/indices/nsd_index")

    dataset = NeuroBridgeDataset(
        subjects=subjects,
        cache_root=cache_root,
        index_root=index_root,
        embeddings_df=clip_df,
        embedding_column=embedding_col,
        exclude_shared1000=True,
        split_by_image=True,
        val_ratio=0.10,
        seed=42,
    )

    if split == "val":
        indices = dataset.val_indices[:n_samples]
    else:
        indices = dataset.train_indices[:n_samples]

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=64, collate_fn=neurobridge_collate, num_workers=0)

    # Load model
    model_config = config.get("model", {})
    model = NeuroBridgeOTModel(model_config).to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded checkpoint from %s", checkpoint_path)
    else:
        logger.info("No checkpoint loaded — diagnosing randomly initialized model")

    model.eval()

    # Collect predictions and targets
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            fmri = batch["fmri"].to(device).float()
            clip_target = batch["clip_target"].to(device).float()
            subject_ids = batch["subject_id"].to(device)

            outputs = model(fmri=fmri, roi_indices={}, subject_id=subject_ids)
            all_preds.append(outputs["clip_embedding"].cpu())
            all_targets.append(clip_target.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    N, D = preds.shape
    logger.info("Collected %d predictions (dim=%d) and %d targets", N, D, targets.shape[0])

    # === Diagnostics ===
    diagnostics = {}

    # 1. Norm statistics
    pred_norms = np.linalg.norm(preds, axis=1)
    target_norms = np.linalg.norm(targets, axis=1)
    diagnostics["prediction_norm"] = {
        "mean": float(pred_norms.mean()),
        "std": float(pred_norms.std()),
        "min": float(pred_norms.min()),
        "max": float(pred_norms.max()),
    }
    diagnostics["target_norm"] = {
        "mean": float(target_norms.mean()),
        "std": float(target_norms.std()),
        "min": float(target_norms.min()),
        "max": float(target_norms.max()),
    }

    # 2. Per-dimension variance
    pred_var_per_dim = preds.var(axis=0)
    diagnostics["prediction_variance"] = {
        "mean_across_dims": float(pred_var_per_dim.mean()),
        "min_across_dims": float(pred_var_per_dim.min()),
        "max_across_dims": float(pred_var_per_dim.max()),
        "n_zero_variance_dims": int((pred_var_per_dim < 1e-8).sum()),
        "total_dims": D,
    }

    # 3. Pairwise cosine similarity (subsample if large)
    subsample = min(N, 500)
    idx = np.random.choice(N, subsample, replace=False)
    preds_sub = preds[idx]
    targets_sub = targets[idx]

    preds_norm_sub = preds_sub / (np.linalg.norm(preds_sub, axis=1, keepdims=True) + 1e-8)
    targets_norm_sub = targets_sub / (np.linalg.norm(targets_sub, axis=1, keepdims=True) + 1e-8)

    pred_sim = preds_norm_sub @ preds_norm_sub.T
    target_sim = targets_norm_sub @ targets_norm_sub.T

    # Upper triangle only (exclude diagonal)
    triu_idx = np.triu_indices(subsample, k=1)
    pred_pairwise = pred_sim[triu_idx]
    target_pairwise = target_sim[triu_idx]

    diagnostics["prediction_pairwise_cosine"] = {
        "mean": float(pred_pairwise.mean()),
        "std": float(pred_pairwise.std()),
        "min": float(pred_pairwise.min()),
        "max": float(pred_pairwise.max()),
    }
    diagnostics["target_pairwise_cosine"] = {
        "mean": float(target_pairwise.mean()),
        "std": float(target_pairwise.std()),
        "min": float(target_pairwise.min()),
        "max": float(target_pairwise.max()),
    }

    # 4. Prediction-vs-target cosine (per sample)
    pred_target_cos = (preds_norm_sub * targets_norm_sub).sum(axis=1)
    diagnostics["pred_vs_target_cosine"] = {
        "mean": float(pred_target_cos.mean()),
        "std": float(pred_target_cos.std()),
        "min": float(pred_target_cos.min()),
        "max": float(pred_target_cos.max()),
    }

    # 5. Score matrix and retrieval
    score_matrix = preds_norm_sub @ targets_norm_sub.T
    diagnostics["score_matrix"] = {
        "mean": float(score_matrix.mean()),
        "std": float(score_matrix.std()),
        "min": float(score_matrix.min()),
        "max": float(score_matrix.max()),
    }

    # Top-1 scores
    top1_scores = score_matrix.max(axis=1)
    diagnostics["top1_scores"] = {
        "mean": float(top1_scores.mean()),
        "std": float(top1_scores.std()),
        "min": float(top1_scores.min()),
        "max": float(top1_scores.max()),
    }

    # 6. Nearest neighbor uniqueness (hubness)
    nn_indices = score_matrix.argmax(axis=1)
    unique_nn = len(set(nn_indices.tolist()))
    diagnostics["hubness"] = {
        "unique_nearest_neighbors": unique_nn,
        "total_queries": subsample,
        "unique_ratio": unique_nn / subsample,
        "most_common_nn_count": int(np.bincount(nn_indices).max()),
    }

    # 7. Collapse detection
    collapse_detected = False
    collapse_reasons = []

    if diagnostics["prediction_pairwise_cosine"]["mean"] > 0.95:
        collapse_detected = True
        collapse_reasons.append(
            f"Predictions nearly identical: pairwise cosine mean={diagnostics['prediction_pairwise_cosine']['mean']:.4f}"
        )

    if diagnostics["prediction_variance"]["mean_across_dims"] < 1e-6:
        collapse_detected = True
        collapse_reasons.append(
            f"Prediction variance near zero: {diagnostics['prediction_variance']['mean_across_dims']:.2e}"
        )

    if diagnostics["hubness"]["unique_ratio"] < 0.1:
        collapse_detected = True
        collapse_reasons.append(
            f"Severe hubness: only {unique_nn}/{subsample} unique nearest neighbors"
        )

    if diagnostics["pred_vs_target_cosine"]["std"] < 0.01:
        collapse_detected = True
        collapse_reasons.append(
            f"Pred-vs-target cosine has no variance (std={diagnostics['pred_vs_target_cosine']['std']:.4f})"
        )

    diagnostics["collapse_detected"] = collapse_detected
    diagnostics["collapse_reasons"] = collapse_reasons
    diagnostics["n_samples"] = N
    diagnostics["split"] = split
    diagnostics["checkpoint"] = checkpoint_path or "none"

    # Write results
    with open(output_path / "collapse_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    logger.info("=== COLLAPSE DIAGNOSTICS ===")
    logger.info("Collapse detected: %s", collapse_detected)
    if collapse_reasons:
        for reason in collapse_reasons:
            logger.warning("  COLLAPSE: %s", reason)
    logger.info("Pred norm: mean=%.4f, std=%.4f", pred_norms.mean(), pred_norms.std())
    logger.info("Pred pairwise cos: mean=%.4f, std=%.4f",
                diagnostics["prediction_pairwise_cosine"]["mean"],
                diagnostics["prediction_pairwise_cosine"]["std"])
    logger.info("Pred-vs-target cos: mean=%.4f, std=%.4f",
                diagnostics["pred_vs_target_cosine"]["mean"],
                diagnostics["pred_vs_target_cosine"]["std"])
    logger.info("Unique NNs: %d/%d (%.1f%%)", unique_nn, subsample, 100 * unique_nn / subsample)

    return diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroBridge-OT prediction collapse diagnostics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--subjects", nargs="+", default=["subj01"])
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--output-dir",
                        default="experimental_results/neurobridge_ot_debug/collapse_diagnostics")
    args = parser.parse_args()

    diagnose_predictions(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        subjects=args.subjects,
        n_samples=args.n_samples,
        split=args.split,
        output_dir=args.output_dir,
    )
