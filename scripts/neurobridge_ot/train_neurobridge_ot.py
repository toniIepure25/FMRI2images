#!/usr/bin/env python3
"""
NeuroBridge-OT Training Script
================================

Full training entrypoint for the NeuroBridge-OT architecture.
Supports single-subject, multi-subject, and adaptation protocols.

Usage:
    python scripts/neurobridge_ot/train_neurobridge_ot.py \
        --config configs/experiments/neurobridge_ot/neurobridge_ot_full.yaml \
        --subjects subj01 subj02 subj05 subj07 \
        --gpu 0 \
        --seed 42
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fmri2img.data.neurobridge_dataset import (
    BalancedSubjectSampler,
    NeuroBridgeDataset,
    neurobridge_collate,
)
from fmri2img.models.neurobridge_ot.losses import NeuroBridgeOTLoss
from fmri2img.models.neurobridge_ot.model import NeuroBridgeOTModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", config_path)
    return config


def build_roi_indices(subjects: List[str], nsd_root: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    """Build ROI indices for all subjects."""
    try:
        from fmri2img.data.roi_utils import build_roi_index
    except ImportError:
        logger.warning("build_roi_index not available, using dummy indices")
        return {}

    all_indices = {}
    for subj in subjects:
        try:
            _, roi_indices = build_roi_index(subj)
            all_indices[subj] = {
                name: torch.from_numpy(idx) if isinstance(idx, np.ndarray) else idx
                for name, idx in roi_indices.items()
            }
            logger.info("Built ROI indices for %s: %d ROIs", subj, len(roi_indices))
        except Exception as e:
            logger.warning("Failed to build ROI indices for %s: %s", subj, e)
    return all_indices


def load_clip_embeddings(clip_cache_path: Path, embedding_column: str = "fused"):
    """Load CLIP embeddings from parquet cache."""
    import pandas as pd
    df = pd.read_parquet(clip_cache_path)
    logger.info("Loaded CLIP cache: %d entries, columns: %s", len(df), list(df.columns)[:5])
    return df


def setup_output_dir(config: Dict[str, Any], args) -> Path:
    """Create output directory for this run."""
    output_root = Path(os.environ.get("OUTPUT_ROOT", "experimental_results"))
    exp_name = config.get("experiment", {}).get("name", "neurobridge_ot")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / exp_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    return output_dir


def save_run_manifest(output_dir: Path, config: Dict, args, extra: Dict = None):
    """Save run manifest with full provenance."""
    import subprocess
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except Exception:
        git_hash = "unknown"
        git_branch = "unknown"

    manifest = {
        "taxonomy_label": config.get("protocol", {}).get("taxonomy_label", "unknown"),
        "model_name": "NeuroBridge-OT",
        "git_commit": git_hash,
        "git_branch": git_branch,
        "config_path": str(args.config),
        "training_subjects": args.subjects,
        "source_subjects": args.source_subjects if args.source_subjects else args.subjects,
        "target_subject": args.target_subject,
        "evaluation_subjects": [args.target_subject] if args.target_subject else args.subjects,
        "target_subject_seen_during_training": config.get("protocol", {}).get("target_subject_seen_during_training", None),
        "target_calibration_fmri_used": config.get("protocol", {}).get("target_calibration_fmri_used", False),
        "target_labels_used": config.get("protocol", {}).get("target_labels_used", None),
        "target_adapter_fitted": config.get("protocol", {}).get("target_adapter_fitted", False),
        "target_learned_parameters_used": config.get("protocol", {}).get("target_learned_parameters_used", None),
        "split_by_image": config.get("data", {}).get("split_by_image", True),
        "exclude_shared1000": config.get("data", {}).get("exclude_shared1000", True),
        "gallery_size": None,  # Populated after evaluation
        "csls_k": config.get("evaluation", {}).get("csls_k", 3),
        "seed": args.seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
    }
    if extra:
        manifest.update(extra)

    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: NeuroBridgeOTLoss,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    roi_indices: Dict[str, Dict[str, torch.Tensor]],
    config: Dict[str, Any],
    epoch: int,
) -> Dict[str, float]:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    loss_components = {}
    n_batches = 0
    grad_accum = config.get("training", {}).get("gradient_accumulation_steps", 1)
    use_amp = config.get("training", {}).get("amp", True)
    amp_dtype = torch.bfloat16 if config.get("training", {}).get("mixed_precision_dtype") == "bf16" else torch.float16

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        fmri = batch["fmri"].to(device)
        clip_target = batch["clip_target"].to(device)
        subject_ids = batch["subject_id"].to(device)
        nsd_ids = batch["nsd_id"].to(device)

        # Get ROI indices for this batch (use first subject's indices for now)
        subj_names = batch["subject_name"]
        batch_roi_indices = {}
        if roi_indices and subj_names[0] in roi_indices:
            batch_roi_indices = roi_indices[subj_names[0]]

        # Forward pass
        with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = model(
                fmri=fmri,
                roi_indices=batch_roi_indices,
                subject_id=subject_ids,
            )

            targets = {
                "clip_target": clip_target,
                "subject_id": subject_ids,
                "nsd_id": nsd_ids,
            }
            if "token_target" in batch:
                targets["token_target"] = batch["token_target"].to(device)

            losses = loss_fn(
                model_outputs=outputs,
                targets=targets,
                ot_cost=outputs.get("ot_cost"),
            )

        loss = losses["total_loss"] / grad_accum

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step
        if (batch_idx + 1) % grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += losses["total_loss"].item()
        for k, v in losses.items():
            if k != "total_loss" and isinstance(v, torch.Tensor):
                loss_components[k] = loss_components.get(k, 0.0) + v.item()
        n_batches += 1

        # Limit batches for smoke tests
        limit = config.get("training", {}).get("limit_batches")
        if limit and batch_idx >= limit - 1:
            break

    avg_loss = total_loss / max(n_batches, 1)
    avg_components = {k: v / max(n_batches, 1) for k, v in loss_components.items()}
    avg_components["total_loss"] = avg_loss
    return avg_components


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: NeuroBridgeOTLoss,
    device: torch.device,
    roi_indices: Dict[str, Dict[str, torch.Tensor]],
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Run validation and compute retrieval metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        fmri = batch["fmri"].to(device)
        clip_target = batch["clip_target"].to(device)
        subject_ids = batch["subject_id"].to(device)
        nsd_ids = batch["nsd_id"].to(device)

        subj_names = batch["subject_name"]
        batch_roi_indices = {}
        if roi_indices and subj_names[0] in roi_indices:
            batch_roi_indices = roi_indices[subj_names[0]]

        outputs = model(fmri=fmri, roi_indices=batch_roi_indices, subject_id=subject_ids)
        targets = {"clip_target": clip_target, "subject_id": subject_ids, "nsd_id": nsd_ids}
        losses = loss_fn(model_outputs=outputs, targets=targets, ot_cost=outputs.get("ot_cost"))

        total_loss += losses["total_loss"].item()
        all_preds.append(outputs["clip_embedding"].cpu())
        all_targets.append(clip_target.cpu())
        n_batches += 1

        limit = config.get("training", {}).get("limit_batches")
        if limit and batch_idx >= limit - 1:
            break

    avg_loss = total_loss / max(n_batches, 1)

    # Retrieval metrics
    preds = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    preds_norm = F.normalize(preds, dim=-1)
    targets_norm = F.normalize(targets_cat, dim=-1)
    sim_matrix = preds_norm @ targets_norm.T

    N = sim_matrix.shape[0]
    ranks = (sim_matrix.argsort(dim=-1, descending=True) == torch.arange(N).unsqueeze(1)).nonzero(as_tuple=True)[1].float()

    metrics = {
        "val_loss": avg_loss,
        "val_r@1": (ranks < 1).float().mean().item(),
        "val_r@5": (ranks < 5).float().mean().item(),
        "val_r@10": (ranks < 10).float().mean().item(),
        "val_mrr": (1.0 / (ranks + 1)).mean().item(),
        "val_median_rank": ranks.median().item(),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="NeuroBridge-OT Training")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--subjects", nargs="+",
                        default=["subj01", "subj02", "subj03", "subj04",
                                 "subj05", "subj06", "subj07", "subj08"])
    parser.add_argument("--target-subject", type=str, default=None)
    parser.add_argument("--source-subjects", nargs="+", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=None)
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load config
    config = load_config(Path(args.config))
    if args.limit_batches:
        config.setdefault("training", {})["limit_batches"] = args.limit_batches

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = setup_output_dir(config, args)
    logger.info("Output: %s", output_dir)

    # Save config
    with open(output_dir / "config_resolved.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    if args.dry_run:
        logger.info("DRY RUN — stopping before data load")
        save_run_manifest(output_dir, config, args)
        return

    # Load CLIP embeddings
    clip_path = Path(os.environ.get(
        "CLIP_CACHE", "outputs/clip_cache/clip_multilayer.parquet"
    ))
    if not clip_path.exists():
        clip_path = Path("outputs/clip_cache/clip.parquet")
    if not clip_path.exists():
        logger.error("CLIP cache not found at %s", clip_path)
        sys.exit(1)
    clip_df = load_clip_embeddings(clip_path)

    # Build dataset
    cache_root = Path(os.environ.get("CACHE_ROOT", "cache")) / "preextracted"
    index_root = Path("data/indices/nsd_index")
    embedding_col = config.get("data", {}).get("embedding_column", "fused")

    dataset = NeuroBridgeDataset(
        subjects=args.subjects,
        cache_root=cache_root,
        index_root=index_root,
        embeddings_df=clip_df,
        embedding_column=embedding_col,
        exclude_shared1000=config.get("data", {}).get("exclude_shared1000", True),
        split_by_image=config.get("data", {}).get("split_by_image", True),
        val_ratio=config.get("data", {}).get("val_ratio", 0.10),
        seed=args.seed,
        protocol=config.get("protocol", {}).get("taxonomy_label", "multi_subject_seen_subject"),
        target_subject=args.target_subject,
        few_shot_n=config.get("protocol", {}).get("few_shot_n"),
    )

    # Dataloaders
    batch_size = config.get("training", {}).get("batch_size", 64)
    train_subset = Subset(dataset, dataset.train_indices)
    val_subset = Subset(dataset, dataset.val_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        collate_fn=neurobridge_collate, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        collate_fn=neurobridge_collate, num_workers=4, pin_memory=True,
    )
    logger.info("Train: %d batches, Val: %d batches", len(train_loader), len(val_loader))

    # Build ROI indices
    nsd_root = Path(os.environ.get("NSD_DATA_ROOT", "data/nsd"))
    roi_indices = build_roi_indices(args.subjects, nsd_root)

    # Model
    model_config = config.get("model", {})
    model = NeuroBridgeOTModel(model_config).to(device)
    logger.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    # Loss
    loss_config = config.get("loss", {})
    loss_fn = NeuroBridgeOTLoss(loss_config).to(device)

    # Optimizer
    train_config = config.get("training", {})
    lr = float(train_config.get("lr", 1e-4))
    weight_decay = float(train_config.get("weight_decay", 1e-5))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # AMP
    use_amp = train_config.get("amp", True)
    scaler = GradScaler() if use_amp and device.type == "cuda" else None

    # LR Scheduler
    epochs = train_config.get("epochs", 200)
    warmup_epochs = train_config.get("warmup_epochs", 5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_metric = 0.0
    patience = train_config.get("patience", 30)
    patience_counter = 0
    training_log = []

    save_run_manifest(output_dir, config, args)

    for epoch in range(epochs):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, roi_indices, config, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device, roi_indices, config)

        scheduler.step()
        elapsed = time.time() - t0

        # Log
        log_entry = {"epoch": epoch, "elapsed_s": elapsed, "lr": scheduler.get_last_lr()[0]}
        log_entry.update({f"train_{k}": v for k, v in train_metrics.items()})
        log_entry.update(val_metrics)
        training_log.append(log_entry)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_r@1=%.4f | val_loss=%.4f | %.1fs",
            epoch + 1, epochs,
            train_metrics.get("total_loss", 0),
            val_metrics.get("val_r@1", 0),
            val_metrics.get("val_loss", 0),
            elapsed,
        )

        # Early stopping
        current_metric = val_metrics.get("val_r@1", 0)
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric,
                "config": config,
            }, output_dir / "checkpoints" / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "config": config,
        }, output_dir / "checkpoints" / "last.pt")

    # Save training log
    import csv
    log_path = output_dir / "metrics" / "training_log.csv"
    if training_log:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=training_log[0].keys())
            writer.writeheader()
            writer.writerows(training_log)

    # Save final metrics
    final_metrics = training_log[-1] if training_log else {}
    final_metrics["best_val_r@1"] = best_metric
    with open(output_dir / "metrics" / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    logger.info("Training complete. Best val R@1: %.4f", best_metric)
    logger.info("Output: %s", output_dir)


if __name__ == "__main__":
    main()
