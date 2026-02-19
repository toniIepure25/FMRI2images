#!/usr/bin/env python3
"""
Train CLIP → fMRI Encoder for Cycle-Consistency Loss
====================================================

Trains an inverse mapping from CLIP image embeddings to fMRI PCA representations.
This encoder is then used during decoder training to add brain-consistency loss.

Usage:
    # Train with default MLP architecture
    python scripts/train_clip_to_fmri.py \\
        --subject subj01 \\
        --clip-cache outputs/clip_cache/clip.parquet \\
        --preproc-dir outputs/preproc/subj01 \\
        --output checkpoints/clip_to_fmri/subj01/encoder.pt
    
    # Train with linear architecture (fast, interpretable)
    python scripts/train_clip_to_fmri.py \\
        --subject subj01 \\
        --architecture linear \\
        --epochs 20
    
    # Train with residual architecture (most expressive)
    python scripts/train_clip_to_fmri.py \\
        --subject subj01 \\
        --architecture residual \\
        --hidden-dim 1024 \\
        --n-layers 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.data.preprocess import NSDPreprocessor
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.io.s3 import get_s3_filesystem, NIfTILoader
from fmri2img.models.clip_to_fmri_encoder import (
    CLIPToFMRIEncoder,
    save_clip_to_fmri_encoder
)
from fmri2img.models.train_utils import (
    extract_features_and_targets,
    train_val_test_split
)


def train_epoch(
    model: CLIPToFMRIEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for clip_batch, fmri_batch in loader:
        clip_batch = clip_batch.to(device)
        fmri_batch = fmri_batch.to(device)
        
        optimizer.zero_grad()
        fmri_pred = model(clip_batch)
        loss = torch.nn.functional.mse_loss(fmri_pred, fmri_batch)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * len(clip_batch)
    
    return total_loss / len(loader.dataset)


def validate(
    model: CLIPToFMRIEncoder,
    loader: DataLoader,
    device: str
) -> Tuple[float, float]:
    """Validate model. Returns (mse_loss, correlation)."""
    model.eval()
    total_loss = 0.0
    all_pred = []
    all_true = []
    
    with torch.no_grad():
        for clip_batch, fmri_batch in loader:
            clip_batch = clip_batch.to(device)
            fmri_batch = fmri_batch.to(device)
            
            fmri_pred = model(clip_batch)
            loss = torch.nn.functional.mse_loss(fmri_pred, fmri_batch)
            
            total_loss += loss.item() * len(clip_batch)
            all_pred.append(fmri_pred.cpu().numpy())
            all_true.append(fmri_batch.cpu().numpy())
    
    mse_loss = total_loss / len(loader.dataset)
    
    # Compute average correlation across features
    all_pred = np.vstack(all_pred)
    all_true = np.vstack(all_true)
    
    # Pearson correlation per feature, then average
    corrs = []
    for i in range(all_pred.shape[1]):
        corr = np.corrcoef(all_pred[:, i], all_true[:, i])[0, 1]
        if not np.isnan(corr):
            corrs.append(corr)
    
    mean_corr = np.mean(corrs) if corrs else 0.0
    
    return mse_loss, mean_corr


def main():
    parser = argparse.ArgumentParser(
        description="Train CLIP → fMRI encoder for cycle-consistency loss"
    )
    
    # Data arguments
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--index-root", default="data/indices/nsd_index")
    parser.add_argument("--clip-cache", default="outputs/clip_cache/clip.parquet")
    parser.add_argument("--preproc-dir", default="outputs/preproc/subj01")
    
    # Model arguments
    parser.add_argument("--architecture", choices=["linear", "mlp", "residual"],
                       default="mlp", help="Encoder architecture")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=2,
                       help="Number of residual blocks (for residual arch)")
    parser.add_argument("--dropout", type=float, default=0.2)
    
    # Training arguments
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    
    # Output arguments
    parser.add_argument("--output", default="checkpoints/clip_to_fmri/subj01/encoder.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    # Dataset arguments
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info("=" * 70)
    logger.info("Training CLIP → fMRI Encoder")
    logger.info("=" * 70)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Architecture: {args.architecture}")
    logger.info(f"Device: {args.device}")
    
    # Load data
    logger.info("\n1. Loading data...")
    
    # Load index
    df = read_subject_index(args.index_root, args.subject)
    logger.info(f"   Loaded index: {len(df)} trials")
    
    # Split
    train_df, val_df, test_df = train_val_test_split(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.seed
    )
    
    # Load preprocessing
    logger.info(f"   Loading preprocessing from {args.preproc_dir}")
    preprocessor = NSDPreprocessor(args.subject, out_dir=Path(args.preproc_dir).parent)
    if not preprocessor.load_artifacts():
        raise ValueError(f"Failed to load preprocessing artifacts from {args.preproc_dir}")
    
    pca_k = preprocessor.pca_.n_components_ if preprocessor.pca_fitted_ else 0
    logger.info(f"   PCA k: {pca_k}")
    
    # Load CLIP cache
    logger.info(f"   Loading CLIP cache from {args.clip_cache}")
    clip_cache = CLIPCache(args.clip_cache).load()
    
    # Extract features
    logger.info("\n2. Extracting features...")
    s3_fs = get_s3_filesystem()
    nifti_loader = NIfTILoader(s3_fs)
    
    # Note: X is fMRI PCA, Y is CLIP (opposite of decoder!)
    X_train_fmri, Y_train_clip, _ = extract_features_and_targets(
        train_df, nifti_loader, preprocessor, clip_cache, desc="train"
    )
    
    X_val_fmri, Y_val_clip, _ = extract_features_and_targets(
        val_df, nifti_loader, preprocessor, clip_cache, desc="val"
    )
    
    logger.info(f"   Train: fMRI {X_train_fmri.shape}, CLIP {Y_train_clip.shape}")
    logger.info(f"   Val:   fMRI {X_val_fmri.shape}, CLIP {Y_val_clip.shape}")
    
    # Create model (CLIP → fMRI, so inputs and outputs are swapped)
    logger.info("\n3. Creating model...")
    clip_dim = Y_train_clip.shape[1]
    fmri_dim = X_train_fmri.shape[1]
    
    model = CLIPToFMRIEncoder(
        clip_dim=clip_dim,
        fmri_dim=fmri_dim,
        architecture=args.architecture,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(args.device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"   Architecture: {args.architecture}")
    logger.info(f"   Parameters: {n_params:,}")
    logger.info(f"   CLIP dim: {clip_dim}, fMRI dim: {fmri_dim}")
    
    # Create DataLoaders (X=CLIP, Y=fMRI for this encoder)
    train_dataset = TensorDataset(
        torch.from_numpy(Y_train_clip).float(),  # CLIP as input
        torch.from_numpy(X_train_fmri).float()   # fMRI as target
    )
    val_dataset = TensorDataset(
        torch.from_numpy(Y_val_clip).float(),
        torch.from_numpy(X_val_fmri).float()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # Training loop
    logger.info("\n4. Training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, args.device)
        val_loss, val_corr = validate(model, val_loader, args.device)
        scheduler.step()
        
        logger.info(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}, "
            f"val_corr={val_corr:.4f}"
        )
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            output_path = Path(args.output)
            metrics = {
                "best_epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "best_val_corr": val_corr,
                "train_loss": train_loss
            }
            save_clip_to_fmri_encoder(model, output_path, metrics, args.subject)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Training complete!")
    logger.info("=" * 70)
    logger.info(f"Best val loss: {best_val_loss:.6f}")
    logger.info(f"Model saved to: {args.output}")
    logger.info("")
    logger.info("Next step: Use this encoder in decoder training with brain-consistency loss")
    logger.info(f"  --clip-to-fmri-encoder {args.output}")
    logger.info(f"  --brain-consistency-weight 0.1")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
