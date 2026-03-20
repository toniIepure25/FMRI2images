#!/usr/bin/env python3
"""Generate predictions for a specific split (train/val/shared1000) from a saved checkpoint.

This script loads a trained model checkpoint and runs inference on the requested
data split, saving predictions in the same format as train_unified.py's
post-training prediction saving. Useful for generating train-split predictions
needed by V39 union-shortlist reranker.

Usage:
    # Generate train predictions for V35 triple-head model
    python scripts/evaluation/generate_split_predictions.py \
        --config configs/experiments/V38_legacy_compact_distill.yaml \
        --checkpoint experimental_results/V35_legacy_teacher_distill/subj01/checkpoint_best.pt \
        --output-dir experimental_results/V35_legacy_teacher_distill/subj01 \
        --split train --subject subj01

    # Generate train predictions for N1v28a legacy model
    python scripts/evaluation/generate_split_predictions.py \
        --config configs/experiments/N1v28a_dual_head.yaml \
        --checkpoint experimental_results/N1v28a_dual_head/subj01/checkpoint_best.pt \
        --output-dir experimental_results/N1v28a_dual_head/subj01 \
        --split train --subject subj01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _build_model(config: dict, input_dim: int, device: str) -> torch.nn.Module:
    """Build model from config."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from fmri2img.models.unified_model import UnifiedModel

    model = UnifiedModel(config, input_dim=input_dim)
    model = model.to(device)
    model.eval()
    return model


def _load_dataset(config: dict, subject: str, split: str):
    """Load dataset for the requested split.

    Returns:
        dataset: PyTorch dataset
        nsd_ids: (N,) array of nsd_ids per trial (for image-level averaging)
        index_df: DataFrame with trial metadata
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from fmri2img.data.preextracted_dataset import PreextractedNSDDataset

    data_cfg = config.get("data", {})

    # Build dataset
    dataset = PreextractedNSDDataset(
        subject=subject,
        embedding_column=data_cfg.get("embedding_column", "embedding"),
        split_by_image=data_cfg.get("split_by_image", True),
        exclude_shared1000=data_cfg.get("exclude_shared1000", True),
        val_fraction=data_cfg.get("val_fraction", 0.1),
        seed=data_cfg.get("seed", 42),
        average_repetitions=data_cfg.get("average_repetitions", False),
    )

    # Get train or val subset
    if split == "train":
        indices = dataset.train_indices
    elif split == "val":
        indices = dataset.val_indices
    else:
        raise ValueError(f"Unsupported split: {split}. Use 'train' or 'val'.")

    subset = torch.utils.data.Subset(dataset, indices)
    nsd_ids = dataset.index_df.iloc[list(indices)]["nsdId"].values

    return subset, nsd_ids, dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate predictions for a data split from a saved checkpoint"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment YAML config")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint_best.pt")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory (e.g. experimental_results/V35/subj01)")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"],
                        help="Which split to generate predictions for")
    parser.add_argument("--subject", type=str, default="subj01",
                        help="Subject ID")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Inference batch size")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Load dataset
    logger.info("Loading %s split for %s...", args.split, args.subject)
    dataset, nsd_ids, full_dataset = _load_dataset(config, args.subject, args.split)
    logger.info("Dataset: %d trials", len(dataset))

    # Determine input dim
    sample = dataset[0]
    if isinstance(sample, dict):
        input_dim = sample["fmri"].shape[0]
    else:
        input_dim = sample[0].shape[0]
    logger.info("Input dim: %d", input_dim)

    # Build model
    model = _build_model(config, input_dim, device)
    model_type = getattr(model, "model_type", "deterministic")
    logger.info("Model type: %s", model_type)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded checkpoint from epoch %d", ckpt.get("epoch", -1))

    # Apply z-scoring if configured
    zscore_mode = config.get("data", {}).get("zscore_mode", None)
    normalize_fmri = config.get("data", {}).get("normalize_fmri", False)
    zscore_stats_path = output_dir / "zscore_stats"

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Run inference
    logger.info("Running inference on %s split (%d batches)...", args.split, len(loader))
    all_preds = []
    all_gts = []
    all_kappas = []
    all_rich_preds = []
    all_rich_gts = []
    all_rerank_preds = []
    all_rerank_gts = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if isinstance(batch, dict):
                fmri = batch["fmri"].to(device, dtype=torch.float32)
                gt = batch["embedding"].numpy() if "embedding" in batch else None
            elif isinstance(batch, (list, tuple)):
                fmri = batch[0].to(device, dtype=torch.float32)
                gt = batch[1].numpy() if len(batch) > 1 else None
            else:
                fmri = batch.to(device, dtype=torch.float32)
                gt = None

            output = model(fmri)

            if isinstance(output, tuple):
                pred = output[0]
                aux = output[1] if len(output) > 1 else {}

                if isinstance(aux, dict):
                    kappa = aux.get("kappa", aux.get("concentration"))
                    if kappa is not None:
                        all_kappas.append(kappa.squeeze(-1).cpu().numpy())

                    # Rich/rerank predictions (vmf_triple)
                    if "rich_pred" in aux:
                        all_rich_preds.append(aux["rich_pred"].cpu().numpy())
                    if "rich_gt" in aux:
                        all_rich_gts.append(aux["rich_gt"].cpu().numpy())
                    if "rerank_pred" in aux:
                        all_rerank_preds.append(aux["rerank_pred"].cpu().numpy())
                    if "rerank_gt" in aux:
                        all_rerank_gts.append(aux["rerank_gt"].cpu().numpy())
                elif torch.is_tensor(aux):
                    all_kappas.append(aux.squeeze(-1).cpu().numpy())
            else:
                pred = output

            all_preds.append(pred.cpu().numpy())
            if gt is not None:
                all_gts.append(gt)

            if (batch_idx + 1) % 50 == 0:
                logger.info("  Batch %d/%d", batch_idx + 1, len(loader))

    preds = np.concatenate(all_preds)
    gts = np.concatenate(all_gts) if all_gts else None
    kappas = np.concatenate(all_kappas) if all_kappas else None

    logger.info("Raw predictions: %s", preds.shape)

    # Image-level averaging (dedup repetitions by nsdId)
    unique_ids = np.unique(nsd_ids)
    n_images = len(unique_ids)
    logger.info("Unique images: %d (from %d trials)", n_images, len(nsd_ids))

    preds_img = np.zeros((n_images, preds.shape[1]), dtype=np.float32)
    gts_img = np.zeros((n_images, gts.shape[1]), dtype=np.float32) if gts is not None else None
    kappas_img = np.zeros(n_images, dtype=np.float32) if kappas is not None else None

    for i, uid in enumerate(unique_ids):
        mask = nsd_ids == uid
        preds_img[i] = preds[mask].mean(axis=0)
        if gts is not None:
            gts_img[i] = gts[mask][0]
        if kappas is not None:
            kappas_img[i] = kappas[mask[:len(kappas)]].mean()

    # L2-normalize after averaging
    nrm = np.linalg.norm(preds_img, axis=-1, keepdims=True)
    preds_img = preds_img / np.maximum(nrm, 1e-8)

    prefix = args.split

    # Save predictions
    np.save(metrics_dir / f"{prefix}_predictions.npy", preds_img)
    logger.info("Saved %s predictions %s", prefix, preds_img.shape)

    if gts_img is not None:
        np.save(metrics_dir / f"{prefix}_ground_truth.npy", gts_img)
        logger.info("Saved %s ground truth %s", prefix, gts_img.shape)

    if kappas_img is not None:
        np.save(metrics_dir / f"{prefix}_kappas.npy", kappas_img)
        logger.info("Saved %s kappas %s", prefix, kappas_img.shape)

    # Save nsd_ids
    np.save(metrics_dir / f"{prefix}_nsd_ids.npy", unique_ids)
    logger.info("Saved %s nsd_ids %s", prefix, unique_ids.shape)

    # vmf_triple: save compact/rerank/rich predictions
    if model_type == "vmf_triple":
        np.save(metrics_dir / f"{prefix}_predictions_compact.npy", preds_img)
        np.save(metrics_dir / f"{prefix}_ground_truth_compact.npy", gts_img)
        logger.info("Saved %s compact predictions (same as main) %s", prefix, preds_img.shape)

        # Rich predictions
        if all_rich_preds:
            rich_preds = np.concatenate(all_rich_preds)
            rich_gts = np.concatenate(all_rich_gts) if all_rich_gts else None
            rp_img = np.zeros((n_images, rich_preds.shape[1]), dtype=np.float32)
            rg_img = np.zeros((n_images, rich_gts.shape[1]), dtype=np.float32) if rich_gts is not None else None
            for i, uid in enumerate(unique_ids):
                mask = nsd_ids == uid
                rp_img[i] = rich_preds[mask].mean(axis=0)
                if rg_img is not None:
                    rg_img[i] = rich_gts[mask][0]
            np.save(metrics_dir / f"{prefix}_predictions_rich.npy", rp_img)
            if rg_img is not None:
                np.save(metrics_dir / f"{prefix}_ground_truth_rich.npy", rg_img)
            logger.info("Saved %s rich predictions %s", prefix, rp_img.shape)

        # Rerank predictions
        if all_rerank_preds:
            rerank_preds = np.concatenate(all_rerank_preds)
            rerank_gts = np.concatenate(all_rerank_gts) if all_rerank_gts else None
            rrp_img = np.zeros((n_images, rerank_preds.shape[1]), dtype=np.float32)
            rrg_img = np.zeros((n_images, rerank_gts.shape[1]), dtype=np.float32) if rerank_gts is not None else None
            for i, uid in enumerate(unique_ids):
                mask = nsd_ids == uid
                rrp_img[i] = rerank_preds[mask].mean(axis=0)
                if rrg_img is not None:
                    rrg_img[i] = rerank_gts[mask][0]
            rrp_img = rrp_img / np.maximum(np.linalg.norm(rrp_img, axis=-1, keepdims=True), 1e-8)
            np.save(metrics_dir / f"{prefix}_predictions_rerank.npy", rrp_img)
            if rrg_img is not None:
                np.save(metrics_dir / f"{prefix}_ground_truth_rerank.npy", rrg_img)
            logger.info("Saved %s rerank predictions %s", prefix, rrp_img.shape)

    logger.info("Done. All outputs in %s", metrics_dir)


if __name__ == "__main__":
    main()
