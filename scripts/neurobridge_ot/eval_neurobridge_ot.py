#!/usr/bin/env python3
"""
NeuroBridge-OT Evaluation Script
===================================

Evaluate a trained NeuroBridge-OT checkpoint on a subject/split/gallery.

Usage:
    python scripts/neurobridge_ot/eval_neurobridge_ot.py \
        --checkpoint experimental_results/neurobridge_ot/best.pt \
        --subjects subj01 \
        --split val \
        --csls-k 3
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fmri2img.models.neurobridge_ot.model import NeuroBridgeOTModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def csls_retrieval(
    query: torch.Tensor, gallery: torch.Tensor, k: int = 3
) -> torch.Tensor:
    """CSLS-corrected similarity for hubness reduction.

    Args:
        query: (N, D) query embeddings.
        gallery: (M, D) gallery embeddings.
        k: Number of neighbors for mean similarity.

    Returns:
        CSLS score matrix (N, M).
    """
    sim = query @ gallery.T  # (N, M)

    # Mean similarity of each gallery item to its k-nearest queries
    topk_gallery, _ = sim.topk(k, dim=0)  # (k, M)
    r_gallery = topk_gallery.mean(dim=0)  # (M,)

    # Mean similarity of each query to its k-nearest gallery items
    topk_query, _ = sim.topk(k, dim=1)  # (N, k)
    r_query = topk_query.mean(dim=1)  # (N,)

    csls = 2 * sim - r_query.unsqueeze(1) - r_gallery.unsqueeze(0)
    return csls


def compute_retrieval_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    csls_k: int = 0,
) -> Dict[str, float]:
    """Compute retrieval metrics.

    Args:
        preds: Predicted embeddings (N, D).
        targets: Target/gallery embeddings (N, D).
        csls_k: If > 0, use CSLS scoring.

    Returns:
        Dict of metrics.
    """
    preds_norm = F.normalize(preds, dim=-1)
    targets_norm = F.normalize(targets, dim=-1)

    if csls_k > 0:
        sim_matrix = csls_retrieval(preds_norm, targets_norm, k=csls_k)
    else:
        sim_matrix = preds_norm @ targets_norm.T

    N = sim_matrix.shape[0]
    # For each query, find where the GT (diagonal) ranks
    sorted_indices = sim_matrix.argsort(dim=-1, descending=True)
    gt_indices = torch.arange(N, device=sim_matrix.device)
    ranks = (sorted_indices == gt_indices.unsqueeze(1)).nonzero(as_tuple=True)[1].float()

    metrics = {
        "r@1": (ranks < 1).float().mean().item(),
        "r@5": (ranks < 5).float().mean().item(),
        "r@10": (ranks < 10).float().mean().item(),
        "mrr": (1.0 / (ranks + 1)).mean().item(),
        "median_rank": ranks.median().item(),
        "mean_rank": ranks.mean().item(),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="NeuroBridge-OT Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--subjects", nargs="+", default=["subj01"])
    parser.add_argument("--target-subject", type=str, default=None)
    parser.add_argument("--split", choices=["val", "test", "shared1000"], default="val")
    parser.add_argument("--gallery", type=str, default=None, help="Path to gallery .npy")
    parser.add_argument("--csls-k", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-batches", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Model
    model_config = config.get("model", {})
    model = NeuroBridgeOTModel(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded model from epoch %d (best_metric=%.4f)",
                ckpt.get("epoch", -1), ckpt.get("best_metric", 0))

    # Load data
    import pandas as pd
    from fmri2img.data.neurobridge_dataset import NeuroBridgeDataset, neurobridge_collate
    from torch.utils.data import DataLoader, Subset

    clip_path = Path(os.environ.get("CLIP_CACHE", "outputs/clip_cache/clip_multilayer.parquet"))
    if not clip_path.exists():
        clip_path = Path("outputs/clip_cache/clip.parquet")
    clip_df = pd.read_parquet(clip_path)

    cache_root = Path(os.environ.get("CACHE_ROOT", "cache")) / "preextracted"
    index_root = Path("data/indices/nsd_index")

    dataset = NeuroBridgeDataset(
        subjects=args.subjects,
        cache_root=cache_root,
        index_root=index_root,
        embeddings_df=clip_df,
        exclude_shared1000=False,
        split_by_image=True,
        seed=args.seed,
        protocol="multi_subject_seen_subject",
    )

    # Select split
    if args.split == "val":
        indices = dataset.val_indices
    elif args.split == "shared1000":
        indices = dataset.get_shared1000_indices()
    else:
        indices = dataset.train_indices

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset, batch_size=64, shuffle=False,
        collate_fn=neurobridge_collate, num_workers=4,
    )

    # Predict
    all_preds = []
    all_targets = []
    all_kappas = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            fmri = batch["fmri"].to(device)
            subject_ids = batch["subject_id"].to(device)

            outputs = model(fmri=fmri, roi_indices={}, subject_id=subject_ids)
            all_preds.append(outputs["clip_embedding"].cpu())
            all_targets.append(batch["clip_target"])

            if "vmf_kappa" in outputs:
                all_kappas.append(outputs["vmf_kappa"].cpu())

            if args.limit_batches and batch_idx >= args.limit_batches - 1:
                break

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    # Metrics
    metrics = compute_retrieval_metrics(preds, targets, csls_k=args.csls_k)
    metrics["n_samples"] = len(preds)
    metrics["split"] = args.split
    metrics["csls_k"] = args.csls_k

    if all_kappas:
        kappas = torch.cat(all_kappas)
        metrics["kappa_mean"] = kappas.mean().item()
        metrics["kappa_std"] = kappas.std().item()
        metrics["kappa_min"] = kappas.min().item()
        metrics["kappa_max"] = kappas.max().item()

    # Output
    logger.info("Results: %s", json.dumps(metrics, indent=2))

    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(output_dir / "predictions.npy", preds.numpy())
    np.save(output_dir / "targets.npy", targets.numpy())

    if all_kappas:
        np.save(output_dir / "kappa_values.npy", torch.cat(all_kappas).numpy())

    logger.info("Saved to %s", output_dir)


if __name__ == "__main__":
    main()
