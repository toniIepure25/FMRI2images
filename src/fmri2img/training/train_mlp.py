#!/usr/bin/env python3
"""
MLP Encoder Training Script
===========================

Train a lightweight MLP encoder for fMRI → CLIP embedding mapping.

Pipeline:
1. Load canonical index and split train/val/test (same as Ridge)
2. Load preprocessing artifacts (T0+T1+T2)
3. Extract fMRI features and CLIP embeddings
4. Train MLP with early stopping on validation cosine
5. Retrain on train+val for selected epoch count
6. Evaluate on test set (cosine, MSE, retrieval@K)
7. Save model and evaluation report

Scientific Design:
- Model selection on validation cosine; final test reported once; retrain on
  train+val to use full data (standard NSD practice)
- Outputs are L2-normalized so cosine is a proper similarity metric in CLIP space
- Keeps the T0/T1/T2 preprocessing and reliability mask identical to Ridge
- Combined cosine+MSE loss aligns both direction and magnitude with CLIP embeddings

Usage:
    # Quick test with tiny PCA
    python scripts/train_mlp.py --subject subj01 --limit 256 --epochs 10
    
    # Full run
    python scripts/train_mlp.py \\
        --subject subj01 \\
        --use-preproc --pca-k 4096 \\
        --clip-cache outputs/clip_cache/clip.parquet \\
        --hidden 1024 --dropout 0.1 \\
        --lr 1e-3 --wd 1e-4 --epochs 50 --patience 7 \\
        --batch-size 256 --limit 2048
"""

import argparse
import json
import logging
import sys
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Silence nibabel warnings
logging.getLogger("nibabel.global").setLevel(logging.WARNING)

from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.data.preprocess import NSDPreprocessor
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.io.s3 import get_s3_filesystem, NIfTILoader
from fmri2img.models.mlp import MLPEncoder, save_mlp, load_mlp
from fmri2img.models.train_utils import (
    extract_features_and_targets,
    train_val_test_split,
    torch_seed_all,
    cosine_loss,
    compose_loss
)
from fmri2img.models.ridge import evaluate_predictions
from fmri2img.eval.retrieval import retrieval_at_k, compute_ranking_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_epoch(
    model: MLPEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    cosine_weight: float = 1.0,
    mse_weight: float = 0.0,
    infonce_weight: float = 0.0,
    temperature: float = 0.07
) -> tuple[float, dict]:
    """
    Train for one epoch.
    
    Returns:
        total_loss: Average loss over epoch
        components: Dict with average loss components
    """
    model.train()
    total_loss = 0.0
    component_sums = {}
    
    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        
        optimizer.zero_grad()
        Y_pred = model(X_batch)
        
        # Compute loss with components
        loss, components = compose_loss(
            Y_pred, Y_batch,
            cosine_weight=cosine_weight,
            mse_weight=mse_weight,
            infonce_weight=infonce_weight,
            temperature=temperature,
            return_components=True
        )
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate loss and components
        batch_size = len(X_batch)
        total_loss += loss.item() * batch_size
        for key, val in components.items():
            if key not in component_sums:
                component_sums[key] = 0.0
            component_sums[key] += val * batch_size
    
    # Average over epoch
    n_samples = len(loader.dataset)
    avg_loss = total_loss / n_samples
    avg_components = {k: v / n_samples for k, v in component_sums.items()}
    
    return avg_loss, avg_components


@torch.no_grad()
def evaluate_epoch(
    model: MLPEncoder,
    loader: DataLoader,
    device: str
) -> dict:
    """Evaluate model on validation/test set."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_pred = model(X_batch)
        
        all_preds.append(Y_pred.cpu().numpy())
        all_targets.append(Y_batch.numpy())
    
    Y_pred = np.vstack(all_preds)
    Y_true = np.vstack(all_targets)
    
    # Compute metrics (reuse Ridge evaluation)
    metrics = evaluate_predictions(Y_true, Y_pred, normalize=True)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train MLP encoder for fMRI → CLIP")
    
    # Data paths
    parser.add_argument("--index-root", default="data/indices/nsd_index",
                       help="NSD index root directory")
    parser.add_argument("--index-file", help="Path to single index file")
    parser.add_argument("--subject", default="subj01", help="Subject to train on")
    
    # Preprocessing
    parser.add_argument("--use-preproc", action="store_true",
                       help="Use preprocessing pipeline")
    parser.add_argument("--pca-k", type=int, help="PCA components (implies --use-preproc)")
    parser.add_argument("--preproc-dir", default="outputs/preproc",
                       help="Preprocessing artifacts directory")
    parser.add_argument("--clip-cache", default="outputs/clip_cache/clip.parquet",
                       help="Path to CLIP cache")
    
    # Model architecture
    parser.add_argument("--hidden", type=int, default=1024,
                       help="Hidden layer size")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout probability")
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=7,
                       help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size")
    
    # Loss configuration (NOVEL CONTRIBUTIONS)
    parser.add_argument("--cosine-weight", type=float, default=1.0,
                       help="Weight for cosine loss (default: 1.0)")
    parser.add_argument("--mse-weight", type=float, default=0.0,
                       help="Weight for MSE loss (default: 0.0)")
    parser.add_argument("--infonce-weight", type=float, default=0.0,
                       help="Weight for InfoNCE contrastive loss (default: 0.0, NOVEL)")
    parser.add_argument("--temperature", type=float, default=0.07,
                       help="Temperature for InfoNCE softmax (default: 0.07)")
    
    # System
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (cuda/cpu)")
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Output paths
    parser.add_argument("--checkpoint-dir", default="checkpoints/mlp",
                       help="Model checkpoint directory")
    parser.add_argument("--report-dir", default="outputs/reports",
                       help="Evaluation report directory")
    parser.add_argument("--config", default="configs/data.yaml",
                       help="Path to data config file")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch_seed_all(args.seed)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    splits_config = config.get("preprocessing", {}).get("splits", {})
    
    try:
        logger.info("=" * 80)
        logger.info("MLP ENCODER TRAINING")
        logger.info("=" * 80)
        logger.info(f"Subject: {args.subject}")
        logger.info(f"Device: {args.device}")
        logger.info(f"Architecture: {args.hidden}-hidden MLP with {args.dropout:.1f} dropout")
        logger.info(f"Training: lr={args.lr}, wd={args.wd}, epochs={args.epochs}, patience={args.patience}")
        
        # Load subject index
        if args.index_file:
            logger.info(f"Loading index from {args.index_file}")
            df = pd.read_parquet(args.index_file)
        else:
            logger.info(f"Loading index for {args.subject} from {args.index_root}")
            df = read_subject_index(args.index_root, args.subject)
        
        if args.limit:
            df = df.head(args.limit)
            logger.info(f"Limited to {len(df)} samples")
        
        # Split data (same as Ridge for fair comparison)
        train_df, val_df, test_df = train_val_test_split(
            df,
            train_ratio=splits_config.get("train_ratio", 0.8),
            val_ratio=splits_config.get("val_ratio", 0.1),
            test_ratio=splits_config.get("test_ratio", 0.1),
            random_seed=splits_config.get("random_seed", 42)
        )
        
        # Setup preprocessor
        preprocessor = NSDPreprocessor(args.subject, args.preproc_dir)
        if args.use_preproc or args.pca_k:
            if not preprocessor.load_artifacts():
                logger.error("Preprocessing artifacts not found. Run nsd_fit_preproc.py first!")
                return 1
            summary = preprocessor.summary()
            logger.info(f"Loaded preprocessing: {summary['n_voxels_kept']:,} voxels")
            if summary.get('pca_fitted'):
                logger.info(f"  PCA: {summary['pca_components']} components")
        
        # Load CLIP cache
        logger.info(f"Loading CLIP cache from {args.clip_cache}")
        clip_cache = CLIPCache(args.clip_cache).load()
        stats = clip_cache.stats()
        logger.info(f"✅ CLIP cache loaded: {stats['cache_size']} embeddings")
        
        # Initialize NIfTI loader
        s3_fs = get_s3_filesystem()
        nifti_loader = NIfTILoader(s3_fs)
        
        # Extract features
        logger.info("Extracting features and targets...")
        X_train, Y_train, train_ids = extract_features_and_targets(
            train_df, nifti_loader, preprocessor, clip_cache, "train"
        )
        X_val, Y_val, val_ids = extract_features_and_targets(
            val_df, nifti_loader, preprocessor, clip_cache, "validation"
        )
        X_test, Y_test, test_ids = extract_features_and_targets(
            test_df, nifti_loader, preprocessor, clip_cache, "test"
        )
        
        # Convert to PyTorch tensors
        X_train = torch.from_numpy(X_train).float()
        Y_train = torch.from_numpy(Y_train).float()
        X_val = torch.from_numpy(X_val).float()
        Y_val = torch.from_numpy(Y_val).float()
        X_test = torch.from_numpy(X_test).float()
        Y_test = torch.from_numpy(Y_test).float()
        
        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(X_train, Y_train),
            batch_size=args.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, Y_val),
            batch_size=args.batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(X_test, Y_test),
            batch_size=args.batch_size,
            shuffle=False
        )
        
        # Initialize model
        input_dim = X_train.shape[1]
        model = MLPEncoder(input_dim=input_dim, hidden=args.hidden, dropout=args.dropout)
        model = model.to(args.device)
        
        logger.info(f"✅ Model initialized: {input_dim}D → {args.hidden}D → 512D")
        logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # Training loop with early stopping
        logger.info("=" * 80)
        logger.info("TRAINING WITH EARLY STOPPING (validation set)")
        logger.info("=" * 80)
        
        best_val_cosine = -np.inf
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(args.epochs):
            train_loss, train_components = train_epoch(
                model, train_loader, optimizer, args.device,
                cosine_weight=args.cosine_weight,
                mse_weight=args.mse_weight,
                infonce_weight=args.infonce_weight,
                temperature=args.temperature
            )
            val_metrics = evaluate_epoch(model, val_loader, args.device)
            scheduler.step()
            
            val_cosine = val_metrics["cosine"]
            
            # Log training components
            components_str = ", ".join([f"{k}={v:.4f}" for k, v in train_components.items()])
            
            logger.info(
                f"Epoch {epoch+1:3d}/{args.epochs}: "
                f"train_loss={train_loss:.4f} ({components_str}), "
                f"val_cosine={val_cosine:.4f} ± {val_metrics['cosine_std']:.4f}, "
                f"val_mse={val_metrics['mse']:.4f}"
            )
            
            # Early stopping check
            if val_cosine > best_val_cosine:
                best_val_cosine = val_cosine
                best_epoch = epoch + 1
                patience_counter = 0
                logger.info(f"  ✅ New best validation cosine: {best_val_cosine:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(f"  Early stopping triggered after {epoch+1} epochs")
                    break
        
        logger.info(f"✅ Best epoch: {best_epoch} (val cosine={best_val_cosine:.4f})")
        
        # Retrain on train+val for best_epoch epochs
        logger.info("=" * 80)
        logger.info(f"RETRAINING on train+val for {best_epoch} epochs")
        logger.info("=" * 80)
        
        X_trainval = torch.cat([X_train, X_val])
        Y_trainval = torch.cat([Y_train, Y_val])
        
        trainval_loader = DataLoader(
            TensorDataset(X_trainval, Y_trainval),
            batch_size=args.batch_size,
            shuffle=True
        )
        
        # Reinitialize model
        final_model = MLPEncoder(input_dim=input_dim, hidden=args.hidden, dropout=args.dropout)
        final_model = final_model.to(args.device)
        
        final_optimizer = AdamW(final_model.parameters(), lr=args.lr, weight_decay=args.wd)
        final_scheduler = CosineAnnealingLR(final_optimizer, T_max=best_epoch)
        
        for epoch in range(best_epoch):
            train_loss, train_components = train_epoch(
                final_model, trainval_loader, final_optimizer, args.device,
                cosine_weight=args.cosine_weight,
                mse_weight=args.mse_weight,
                infonce_weight=args.infonce_weight,
                temperature=args.temperature
            )
            final_scheduler.step()
            components_str = ", ".join([f"{k}={v:.4f}" for k, v in train_components.items()])
            logger.info(f"Epoch {epoch+1:3d}/{best_epoch}: train_loss={train_loss:.4f} ({components_str})")
        
        # Evaluate on test set
        logger.info("=" * 80)
        logger.info("TEST SET EVALUATION")
        logger.info("=" * 80)
        
        test_metrics = evaluate_epoch(final_model, test_loader, args.device)
        
        logger.info(f"Cosine: {test_metrics['cosine']:.4f} ± {test_metrics['cosine_std']:.4f}")
        logger.info(f"MSE: {test_metrics['mse']:.4f}")
        
        # Retrieval evaluation
        logger.info("\nRetrieval evaluation (test set as gallery)...")
        
        # Get predictions as numpy
        final_model.eval()
        with torch.no_grad():
            Y_test_pred = final_model(X_test.to(args.device)).cpu().numpy()
        
        Y_test_np = Y_test.numpy()
        gt_indices = np.arange(len(Y_test_np))
        
        retrieval_metrics = retrieval_at_k(
            Y_test_pred, Y_test_np, gt_indices, ks=(1, 5, 10)
        )
        
        for k, v in retrieval_metrics.items():
            logger.info(f"{k}: {v:.4f} ({v*100:.2f}%)")
        
        # Additional ranking metrics
        ranking_metrics = compute_ranking_metrics(Y_test_pred, Y_test_np, gt_indices)
        logger.info(f"Mean rank: {ranking_metrics['mean_rank']:.2f}")
        logger.info(f"Median rank: {ranking_metrics['median_rank']:.2f}")
        logger.info(f"MRR: {ranking_metrics['mrr']:.4f}")
        
        # Combine metrics
        test_metrics.update(retrieval_metrics)
        test_metrics.update(ranking_metrics)
        
        # Get preprocessing summary
        preproc_summary = preprocessor.summary() if preprocessor.is_fitted_ else {}
        
        # Save model
        checkpoint_path = Path(args.checkpoint_dir) / args.subject / "mlp.pt"
        
        # Build preprocessing metadata
        preproc_meta = {}
        if preprocessor.is_fitted_:
            preproc_meta = {
                "used_preproc": True,
                "k": preproc_summary.get("pca_components"),
                "reliability_thr": preproc_summary.get("reliability_threshold"),
                "path": str(preprocessor.preproc_dir) if hasattr(preprocessor, "preproc_dir") else str(Path(args.preproc_dir) / args.subject),
                "subject": args.subject
            }
        else:
            preproc_meta = {
                "used_preproc": False,
                "k": None,
                "reliability_thr": None,
                "path": None,
                "subject": args.subject
            }
        
        meta = {
            "input_dim": input_dim,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "best_epoch": best_epoch,
            "best_val_cosine": float(best_val_cosine),
            "lr": args.lr,
            "weight_decay": args.wd,
            "mse_weight": args.mse_weight,
            "subject": args.subject,
            "preproc": preproc_meta,
        }
        
        save_mlp(final_model, str(checkpoint_path), meta)
        logger.info(f"✅ Model saved to {checkpoint_path}")
        
        # Build evaluation report (mirror Ridge format)
        report = {
            "subject": args.subject,
            "model": "MLP",
            "preprocessing": {
                "used": preprocessor.is_fitted_,
                "pca_k": preproc_summary.get("pca_components", None),
                "n_voxels_kept": preproc_summary.get("n_voxels_kept", None),
            },
            "data_splits": {
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "n_train_valid": len(X_train),
                "n_val_valid": len(X_val),
                "n_test_valid": len(X_test),
            },
            "hyperparameters": {
                "hidden": args.hidden,
                "dropout": args.dropout,
                "lr": args.lr,
                "weight_decay": args.wd,
                "mse_weight": args.mse_weight,
                "batch_size": args.batch_size,
                "best_epoch": best_epoch,
            },
            "validation_metrics": {
                "best_cosine": float(best_val_cosine),
            },
            "test_metrics": test_metrics,
            "model_checkpoint": str(checkpoint_path),
        }
        
        # Save report
        report_path = Path(args.report_dir) / args.subject / "mlp_eval.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("✅ Training complete!")
        logger.info(f"Model: {checkpoint_path}")
        logger.info(f"Report: {report_path}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
