#!/usr/bin/env python3
"""
CLIP Adapter Training Script
============================

Train a lightweight adapter to map 512-D CLIP embeddings (ViT-B/32) to the
target dimension required by diffusion models (768-D for SD-1.5, 1024-D for SD-2.1).

Pipeline:
1. Load canonical index and split train/val/test (matches encoder training)
2. Load ground-truth ViT-B/32 CLIP embeddings (512-D) from cache
3. Compute target CLIP embeddings from diffusion model's CLIP encoder
4. Train linear adapter with MSE + cosine loss
5. Early stopping on validation cosine similarity
6. Retrain on train+val for selected epoch count
7. Evaluate on test set and save checkpoint + report

Scientific Design:
- Reduces representation gap between encoder output (512-D) and diffusion conditioning
- Target embeddings are from the diffusion model's own CLIP (e.g., OpenCLIP ViT-H/14)
- Trained with combined MSE+cosine loss for both magnitude and angular alignment
- L2-normalized outputs preserve cosine similarity metric in target CLIP space

Usage:
    # Quick test
    python scripts/train_clip_adapter.py \\
        --subject subj01 \\
        --clip-cache outputs/clip_cache/clip.parquet \\
        --model-id stabilityai/stable-diffusion-2-1 \\
        --epochs 10 --limit 256
    
    # Full run
    python scripts/train_clip_adapter.py \\
        --index-root data/indices/nsd_index \\
        --subject subj01 \\
        --clip-cache outputs/clip_cache/clip.parquet \\
        --model-id stabilityai/stable-diffusion-2-1 \\
        --epochs 30 --batch-size 256 \\
        --out checkpoints/clip_adapter/subj01/adapter.pt
"""

import argparse
import json
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import io

# Silence warnings
logging.getLogger("nibabel.global").setLevel(logging.WARNING)

from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.io.s3 import get_s3_filesystem
from fmri2img.models.clip_adapter import CLIPAdapter, save_adapter
from fmri2img.models.train_utils import (
    train_val_test_split,
    torch_seed_all,
    cosine_loss,
    compose_loss
)
from fmri2img.models.ridge import evaluate_predictions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_diffusion_clip_encoder(model_id: str, device: str):
    """
    Load the CLIP image encoder from a diffusion model.
    
    Args:
        model_id: HuggingFace model ID (e.g., "stabilityai/stable-diffusion-2-1")
        device: Device to load on
    
    Returns:
        encoder: CLIP image encoder with .encode_image() method
        target_dim: Output dimension of the CLIP encoder
    """
    from transformers import CLIPVisionModel, CLIPImageProcessor
    
    logger.info(f"Loading CLIP encoder from {model_id}...")
    
    try:
        # Load the vision model from the diffusion pipeline
        vision_model = CLIPVisionModel.from_pretrained(
            model_id,
            subfolder="image_encoder" if "stable-diffusion" in model_id else None
        )
        vision_model = vision_model.to(device)
        vision_model.eval()
        
        # Get the image processor
        processor = CLIPImageProcessor.from_pretrained(
            model_id,
            subfolder="image_encoder" if "stable-diffusion" in model_id else None
        )
        
        # Determine output dimension
        target_dim = vision_model.config.hidden_size
        
        logger.info(f"✅ Loaded CLIP encoder: {target_dim}-D output")
        
        return vision_model, processor, target_dim
        
    except Exception as e:
        logger.warning(f"Could not load image_encoder subfolder: {e}")
        logger.info("Trying to load from feature_extractor...")
        
        # Fallback: Try loading from the main pipeline
        from diffusers import StableDiffusionPipeline
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32
        )
        
        # SD models use text encoder, but we need the CLIP image encoder
        # For SD 2.1: OpenCLIP ViT-H/14 (1024-D projected)
        # For SD 1.5: CLIP ViT-L/14 (768-D projected)
        
        # Try to infer from model_id
        if "2-1" in model_id or "2.1" in model_id:
            target_dim = 1024
            logger.info("Detected SD 2.1 → using OpenCLIP ViT-H/14 (1024-D)")
            # Load full CLIP model to access visual projection (1280→1024)
            from transformers import CLIPModel, CLIPProcessor
            vision_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        elif "1-5" in model_id or "1.5" in model_id:
            target_dim = 768
            logger.info("Detected SD 1.5 → using CLIP ViT-L/14 (768-D)")
            from transformers import CLIPModel, CLIPProcessor
            vision_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        else:
            # Default to 1024 for SD 2.x
            target_dim = 1024
            logger.warning(f"Unknown model, defaulting to 1024-D (OpenCLIP ViT-H/14)")
            from transformers import CLIPModel, CLIPProcessor
            vision_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        
        vision_model = vision_model.to(device)
        vision_model.eval()
        
        return vision_model, processor, target_dim


def compute_target_embeddings_cached(
    nsd_ids: np.ndarray,
    model_id: str,
    s3_fs,
    vision_model,
    processor,
    device: str,
    cache_dir: Path
) -> np.ndarray:
    """
    Compute or load cached target CLIP embeddings for given NSD IDs.
    
    Args:
        nsd_ids: Array of NSD IDs
        model_id: Model identifier for cache naming
        s3_fs: S3 filesystem
        vision_model: CLIP vision model
        processor: CLIP image processor
        device: Device
        cache_dir: Cache directory
    
    Returns:
        target_embeddings: (n_samples, target_dim) array
    """
    # Sanitize model_id for filename
    model_slug = model_id.replace("/", "_").replace("-", "_")
    cache_file = cache_dir / f"target_clip_{model_slug}.parquet"
    
    # Try to load from cache
    if cache_file.exists():
        logger.info(f"Loading cached target embeddings from {cache_file}")
        df_cache = pd.read_parquet(cache_file)
        
        # Check if all IDs are cached
        cached_ids = set(df_cache["nsdId"].values)
        requested_ids = set(nsd_ids)
        
        if requested_ids.issubset(cached_ids):
            logger.info("✅ All requested embeddings found in cache")
            # Extract in order
            df_cache_indexed = df_cache.set_index("nsdId")
            embeddings_list = []
            for nsd_id in nsd_ids:
                emb = df_cache_indexed.loc[nsd_id, "embedding"]
                embeddings_list.append(np.array(emb))
            return np.vstack(embeddings_list)
        else:
            logger.info(f"Cache miss for {len(requested_ids - cached_ids)} IDs, computing...")
            # Load existing cache for merging
            existing_cache = {
                row["nsdId"]: np.array(row["embedding"]) 
                for _, row in df_cache.iterrows()
            }
    else:
        logger.info("No cache found, computing all target embeddings...")
        existing_cache = {}
    
    # Compute missing embeddings
    logger.info(f"Computing target embeddings for {len(nsd_ids)} samples...")
    
    # Load images from NSD using robust loader
    from fmri2img.io.nsd_images import load_nsd_images
    
    # Get IDs that need computation
    ids_to_compute = [nsd_id for nsd_id in nsd_ids if nsd_id not in existing_cache]
    
    if ids_to_compute:
        logger.info(f"Loading {len(ids_to_compute)} images from NSD...")
        nsd_images = load_nsd_images(ids_to_compute, s3_fs=s3_fs, prefer="hdf5")
        logger.info(f"Successfully loaded {len(nsd_images)}/{len(ids_to_compute)} images")
    else:
        nsd_images = {}
    
    all_embeddings = {}
    
    for nsd_id in nsd_ids:
        # Check existing cache first
        if nsd_id in existing_cache:
            all_embeddings[nsd_id] = existing_cache[nsd_id]
            continue
        
        # Check if image was loaded
        if nsd_id not in nsd_images:
            logger.warning(f"Skipping nsdId={nsd_id} (image not available)")
            continue
        
        try:
            img = nsd_images[nsd_id]
            
            # Process image
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Encode - use get_image_features() to get projected embeddings
            with torch.no_grad():
                # This applies the visual projection (1280→1024 for ViT-H/14)
                image_features = vision_model.get_image_features(**inputs)
                embedding = image_features.squeeze(0).cpu().numpy()
                # L2 normalize
                embedding = embedding / np.linalg.norm(embedding)
            
            all_embeddings[nsd_id] = embedding
            
        except Exception as e:
            logger.warning(f"Failed to compute embedding for nsdId={nsd_id}: {e}")
            # Use zero vector as fallback
            target_dim = existing_cache[list(existing_cache.keys())[0]].shape[0] if existing_cache else 1024
            all_embeddings[nsd_id] = np.zeros(target_dim, dtype=np.float32)
    
    # Save updated cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    df_cache_new = pd.DataFrame([
        {"nsdId": nsd_id, "embedding": emb.tolist()}
        for nsd_id, emb in all_embeddings.items()
    ])
    
    df_cache_new.to_parquet(cache_file, index=False)
    logger.info(f"✅ Saved target embeddings cache to {cache_file}")
    
    # Return in requested order
    embeddings_list = [all_embeddings[nsd_id] for nsd_id in nsd_ids]
    return np.vstack(embeddings_list)


def train_epoch(
    model: CLIPAdapter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    mse_weight: float = 0.5
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        
        optimizer.zero_grad()
        Y_pred = model(X_batch)
        
        loss = compose_loss(Y_pred, Y_batch, mse_weight=mse_weight)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_epoch(
    model: CLIPAdapter,
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
    
    # Compute metrics
    metrics = evaluate_predictions(Y_true, Y_pred, normalize=True)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train CLIP adapter")
    
    # Data paths
    parser.add_argument("--index-root", default="data/indices/nsd_index",
                       help="NSD index root directory")
    parser.add_argument("--subject", default="subj01", help="Subject ID")
    parser.add_argument("--clip-cache", required=True,
                       help="Path to ViT-B/32 CLIP cache (512-D)")
    
    # Model
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-2-1",
                       help="Diffusion model ID for target CLIP")
    parser.add_argument("--use-layernorm", action="store_true", default=True,
                       help="Use LayerNorm in adapter")
    parser.add_argument("--no-layernorm", action="store_false", dest="use_layernorm",
                       help="Disable LayerNorm")
    
    # Training
    parser.add_argument("--epochs", type=int, default=30,
                       help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--mse-weight", type=float, default=0.5,
                       help="Weight for MSE term in loss")
    parser.add_argument("--patience", type=int, default=5,
                       help="Early stopping patience")
    
    # System
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (cuda/cpu)")
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Output
    parser.add_argument("--out", required=True,
                       help="Output checkpoint path")
    parser.add_argument("--cache-dir", default="outputs/clip_cache",
                       help="Directory for target embedding cache")
    parser.add_argument("--config", default="configs/data.yaml",
                       help="Data config file")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch_seed_all(args.seed)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    splits_config = config.get("preprocessing", {}).get("splits", {})
    
    try:
        logger.info("=" * 80)
        logger.info("CLIP ADAPTER TRAINING")
        logger.info("=" * 80)
        logger.info(f"Subject: {args.subject}")
        logger.info(f"Model: {args.model_id}")
        logger.info(f"Device: {args.device}")
        
        # Load subject index
        logger.info(f"Loading index for {args.subject} from {args.index_root}")
        df = read_subject_index(args.index_root, args.subject)
        
        if args.limit:
            df = df.head(args.limit)
            logger.info(f"Limited to {len(df)} samples")
        
        # Split data
        train_df, val_df, test_df = train_val_test_split(
            df,
            train_ratio=splits_config.get("train_ratio", 0.8),
            val_ratio=splits_config.get("val_ratio", 0.1),
            test_ratio=splits_config.get("test_ratio", 0.1),
            random_seed=splits_config.get("random_seed", 42)
        )
        
        # Load source CLIP cache (512-D ViT-B/32)
        logger.info(f"Loading source CLIP cache from {args.clip_cache}")
        clip_cache = CLIPCache(args.clip_cache).load()
        stats = clip_cache.stats()
        logger.info(f"✅ Source CLIP cache: {stats['cache_size']} embeddings (512-D)")
        
        # Setup S3 filesystem for image loading
        s3_fs = get_s3_filesystem()
        
        # Load diffusion model's CLIP encoder
        vision_model, processor, target_dim = get_diffusion_clip_encoder(
            args.model_id, args.device
        )
        
        logger.info(f"Adapter architecture: 512-D → {target_dim}-D")
        
        # Get all NSD IDs
        all_nsd_ids = pd.concat([train_df, val_df, test_df])["nsdId"].values
        
        # Compute target embeddings (with caching)
        cache_dir = Path(args.cache_dir)
        target_embeddings_all = compute_target_embeddings_cached(
            all_nsd_ids,
            args.model_id,
            s3_fs,
            vision_model,
            processor,
            args.device,
            cache_dir
        )
        
        # Build mapping nsdId -> embedding
        target_emb_map = {
            nsd_id: target_embeddings_all[i]
            for i, nsd_id in enumerate(all_nsd_ids)
        }
        
        # Extract source and target embeddings for each split
        def extract_embeddings(split_df, split_name):
            source_list = []
            target_list = []
            
            for _, row in split_df.iterrows():
                nsd_id = int(row["nsdId"])
                
                # Get source embedding (512-D)
                source_dict = clip_cache.get([nsd_id])
                source_emb = source_dict.get(nsd_id)
                
                # Get target embedding
                target_emb = target_emb_map.get(nsd_id)
                
                if source_emb is not None and target_emb is not None:
                    source_list.append(source_emb)
                    target_list.append(target_emb)
            
            X = np.vstack(source_list)
            Y = np.vstack(target_list)
            
            logger.info(f"{split_name}: {len(X)} samples, {X.shape[1]}D → {Y.shape[1]}D")
            return X, Y
        
        logger.info("Extracting embeddings for splits...")
        X_train, Y_train = extract_embeddings(train_df, "Train")
        X_val, Y_val = extract_embeddings(val_df, "Val")
        X_test, Y_test = extract_embeddings(test_df, "Test")
        
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
        
        # Initialize adapter
        adapter = CLIPAdapter(
            in_dim=512,
            out_dim=target_dim,
            use_layernorm=args.use_layernorm
        )
        adapter = adapter.to(args.device)
        
        logger.info(f"✅ Adapter initialized: {512}D → {target_dim}D")
        logger.info(f"   Parameters: {sum(p.numel() for p in adapter.parameters()):,}")
        
        # Optimizer and scheduler
        optimizer = AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # Training loop with early stopping
        logger.info("=" * 80)
        logger.info("TRAINING WITH EARLY STOPPING")
        logger.info("=" * 80)
        
        best_val_cosine = -np.inf
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(adapter, train_loader, optimizer, args.device, args.mse_weight)
            val_metrics = evaluate_epoch(adapter, val_loader, args.device)
            scheduler.step()
            
            val_cosine = val_metrics["cosine"]
            
            logger.info(
                f"Epoch {epoch+1:3d}/{args.epochs}: "
                f"train_loss={train_loss:.4f}, "
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
        
        # Reinitialize adapter
        final_adapter = CLIPAdapter(
            in_dim=512,
            out_dim=target_dim,
            use_layernorm=args.use_layernorm
        )
        final_adapter = final_adapter.to(args.device)
        
        final_optimizer = AdamW(final_adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        final_scheduler = CosineAnnealingLR(final_optimizer, T_max=best_epoch)
        
        for epoch in range(best_epoch):
            train_loss = train_epoch(
                final_adapter, trainval_loader, final_optimizer, args.device, args.mse_weight
            )
            final_scheduler.step()
            logger.info(f"Epoch {epoch+1:3d}/{best_epoch}: train_loss={train_loss:.4f}")
        
        # Evaluate on test set
        logger.info("=" * 80)
        logger.info("TEST SET EVALUATION")
        logger.info("=" * 80)
        
        test_metrics = evaluate_epoch(final_adapter, test_loader, args.device)
        
        logger.info(f"Cosine: {test_metrics['cosine']:.4f} ± {test_metrics['cosine_std']:.4f}")
        logger.info(f"MSE: {test_metrics['mse']:.4f}")
        
        # Save adapter
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        meta = {
            "subject": args.subject,
            "model_id": args.model_id,
            "in_dim": 512,
            "out_dim": target_dim,
            "use_layernorm": args.use_layernorm,
            "best_epoch": best_epoch,
            "best_val_cosine": float(best_val_cosine),
            "test_cosine": float(test_metrics["cosine"]),
            "test_mse": float(test_metrics["mse"]),
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "mse_weight": args.mse_weight,
        }
        
        final_adapter.save(str(output_path), meta)
        logger.info(f"✅ Adapter saved to {output_path}")
        
        # Save JSON report
        report = {
            "subject": args.subject,
            "model": "CLIPAdapter",
            "source_dim": 512,
            "target_dim": target_dim,
            "target_model": args.model_id,
            "data_splits": {
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "n_train_valid": len(X_train),
                "n_val_valid": len(X_val),
                "n_test_valid": len(X_test),
            },
            "hyperparameters": {
                "use_layernorm": args.use_layernorm,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "mse_weight": args.mse_weight,
                "batch_size": args.batch_size,
                "best_epoch": best_epoch,
            },
            "validation_metrics": {
                "best_cosine": float(best_val_cosine),
            },
            "test_metrics": test_metrics,
            "model_checkpoint": str(output_path),
        }
        
        report_path = output_path.parent / f"{args.subject}_clip_adapter.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("✅ Training complete!")
        logger.info(f"Adapter: {output_path}")
        logger.info(f"Report: {report_path}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
