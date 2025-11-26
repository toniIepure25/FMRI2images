#!/usr/bin/env python3
"""
SOTA Two-Stage Encoder Training Script
======================================

Train advanced residual encoder for fMRI → CLIP mapping with:
- Two-stage architecture (Stage 1: fMRI → latent, Stage 2: latent → CLIP)
- Multi-objective loss (MSE + Cosine + InfoNCE)
- Optional self-supervised pretraining
- Configurable via Hydra/YAML

Features:
- Residual blocks with LayerNorm and GELU
- InfoNCE contrastive loss for discriminative learning
- Self-supervised pretraining (masked/denoising autoencoder)
- Staged training (pretrain Stage 1, freeze and train Stage 2)
- Backward compatible with simple MLP

Usage:
    # Simple two-stage encoder (no pretraining)
    python scripts/train_two_stage.py \\
        --subject subj01 \\
        --use-preproc --pca-k 512 \\
        --latent-dim 768 --n-blocks 4 \\
        --head-type mlp --head-hidden 512 \\
        --batch-size 128 --epochs 50
    
    # With self-supervised pretraining
    python scripts/train_two_stage.py \\
        --subject subj01 \\
        --use-preproc --pca-k 512 \\
        --latent-dim 768 --n-blocks 4 \\
        --self-supervised --ssl-objective masked --ssl-epochs 20 \\
        --batch-size 128 --epochs 50
    
    # Staged training (pretrain Stage 1, freeze and train Stage 2)
    python scripts/train_two_stage.py \\
        --subject subj01 \\
        --use-preproc --pca-k 512 \\
        --latent-dim 768 --n-blocks 4 \\
        --self-supervised --ssl-epochs 20 \\
        --freeze-stage1 --stage2-epochs 30
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
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import project modules
from fmri2img.data.nsd_index_reader import read_subject_index
from fmri2img.data.preprocess import NSDPreprocessor
from fmri2img.data.clip_cache import CLIPCache
from fmri2img.io.s3 import get_s3_filesystem, NIfTILoader
from fmri2img.models.encoders import (
    TwoStageEncoder,
    MultiLayerTwoStageEncoder,
    SelfSupervisedPretrainer,
    save_two_stage_encoder,
    load_two_stage_encoder,
    load_multilayer_two_stage_encoder
)
from fmri2img.training.losses import MultiLoss, MultiLayerLoss, compute_multiloss
from fmri2img.models.train_utils import (
    extract_features_and_targets,
    train_val_test_split,
    torch_seed_all
)
from fmri2img.models.ridge import evaluate_predictions
from fmri2img.eval.retrieval import retrieval_at_k, compute_ranking_metrics


def train_epoch(
    model: TwoStageEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MultiLoss,
    device: str,
    epoch: int
) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch with multi-objective loss."""
    model.train()
    total_loss = 0.0
    loss_components_sum = {"mse": 0.0, "cosine": 0.0, "info_nce": 0.0}
    n_batches = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for X_batch, Y_batch in pbar:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        
        optimizer.zero_grad()
        Y_pred = model(X_batch)
        
        # Compute loss with components
        loss, components = criterion(Y_pred, Y_batch, return_components=True)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item() * len(X_batch)
        for key in loss_components_sum:
            loss_components_sum[key] += components[key] * len(X_batch)
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "mse": f"{components['mse']:.4f}",
            "cos": f"{components['cosine']:.4f}",
            "nce": f"{components['info_nce']:.4f}"
        })
    
    # Average over all samples
    n_samples = len(loader.dataset)
    avg_loss = total_loss / n_samples
    avg_components = {k: v / n_samples for k, v in loss_components_sum.items()}
    
    return avg_loss, avg_components


@torch.no_grad()
def evaluate_epoch(
    model: TwoStageEncoder,
    loader: DataLoader,
    device: str
) -> Dict:
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


def train_epoch_multilayer(
    model: MultiLayerTwoStageEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MultiLayerLoss,
    device: str,
    epoch: int
) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch with multi-layer supervision."""
    model.train()
    total_loss = 0.0
    loss_components_sum = {"layer_4": 0.0, "layer_8": 0.0, "layer_12": 0.0, "final": 0.0}
    n_batches = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for X_batch, Y_batch_dict in pbar:
        X_batch = X_batch.to(device)
        # Move all target layers to device
        Y_batch_dict = {k: v.to(device) for k, v in Y_batch_dict.items()}
        
        optimizer.zero_grad()
        Y_pred_dict = model(X_batch)
        
        # Compute multi-layer loss with components
        # Pass model for Phase 3 multi-layer InfoNCE
        loss, components = criterion(Y_pred_dict, Y_batch_dict, model=model, return_components=True)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item() * len(X_batch)
        for key in loss_components_sum:
            if key in components:
                loss_components_sum[key] += components[key] * len(X_batch)
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "l4": f"{components.get('layer_4', 0):.3f}",
            "l8": f"{components.get('layer_8', 0):.3f}",
            "l12": f"{components.get('layer_12', 0):.3f}",
            "fin": f"{components.get('final', 0):.3f}"
        })
    
    # Average over all samples
    n_samples = len(loader.dataset)
    avg_loss = total_loss / n_samples
    avg_components = {k: v / n_samples for k, v in loss_components_sum.items()}
    
    return avg_loss, avg_components


@torch.no_grad()
def evaluate_epoch_multilayer(
    model: MultiLayerTwoStageEncoder,
    loader: DataLoader,
    device: str
) -> Dict:
    """Evaluate multi-layer model on validation/test set (using final layer only)."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for X_batch, Y_batch_dict in loader:
        X_batch = X_batch.to(device)
        Y_pred_dict = model(X_batch)
        
        # Use final layer for evaluation
        all_preds.append(Y_pred_dict['final'].cpu().numpy())
        all_targets.append(Y_batch_dict['final'].numpy())
    
    Y_pred = np.vstack(all_preds)
    Y_true = np.vstack(all_targets)
    
    # Compute metrics
    metrics = evaluate_predictions(Y_true, Y_pred, normalize=True)
    
    return metrics


def load_multilayer_clip_cache(cache_path: str) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load multi-layer CLIP cache from parquet file.
    
    Returns:
        Dict mapping nsdId to layer features:
        {nsdId: {'layer_4': array(768,), 'layer_8': ..., 'layer_12': ..., 'final': array(512,)}}
    """
    logger.info(f"Loading multi-layer CLIP cache from {cache_path}...")
    df = pd.read_parquet(cache_path)
    
    cache_dict = {}
    for _, row in df.iterrows():
        nsd_id = int(row['nsdId'])
        cache_dict[nsd_id] = {
            'layer_4': np.array(row['layer_4'], dtype=np.float32),
            'layer_8': np.array(row['layer_8'], dtype=np.float32),
            'layer_12': np.array(row['layer_12'], dtype=np.float32),
            'final': np.array(row['final'], dtype=np.float32)
        }
    
    logger.info(f"  Loaded {len(cache_dict)} multi-layer embeddings")
    return cache_dict


def extract_features_and_multilayer_targets(
    df: pd.DataFrame,
    nifti_loader: NIfTILoader,
    preprocessor: NSDPreprocessor,
    multilayer_cache: Dict[int, Dict[str, np.ndarray]],
    desc: str = "data"
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Extract fMRI features and multi-layer CLIP targets.
    Uses same optimization as extract_features_and_targets: group by file to load each once.
    
    Returns:
        X: fMRI features (N, fmri_dim)
        Y_dict: Dict of CLIP targets {'layer_4': (N, 768), ..., 'final': (N, 512)}
        nsd_ids: NSD stimulus IDs (N,)
    """
    from collections import defaultdict
    
    X_list = []
    Y_dict_lists = {'layer_4': [], 'layer_8': [], 'layer_12': [], 'final': []}
    nsd_ids_list = []
    
    # Group samples by beta_path to load each file only once (OPTIMIZATION)
    samples_by_file = defaultdict(list)
    for idx, row in df.iterrows():
        nsd_id = int(row["nsdId"])
        
        # Skip if no multi-layer embedding
        if nsd_id not in multilayer_cache:
            continue
        
        beta_path = row["beta_path"]
        samples_by_file[beta_path].append({
            'beta_index': int(row["beta_index"]),
            'nsdId': nsd_id,
            'row_idx': idx
        })
    
    logger.info(f"Extracting {desc}: {len(df)} samples from {len(samples_by_file)} unique files")
    
    # Process each beta file once
    pbar = tqdm(samples_by_file.items(), desc=f"Loading {desc}")
    for beta_path, samples in pbar:
        try:
            # Load the 4D beta file ONCE
            img = nifti_loader.load(beta_path)
            data_4d = img.get_fdata()
            
            # Extract all volumes needed from this file
            for sample in samples:
                try:
                    beta_index = sample['beta_index']
                    nsd_id = sample['nsdId']
                    
                    # Extract single volume
                    vol = data_4d[..., beta_index].astype(np.float32)
                    
                    # Apply preprocessing (same as extract_features_and_targets)
                    if preprocessor and preprocessor.is_fitted_:
                        # T0: z-score (online)
                        vol_z = preprocessor.transform_T0(vol)
                        # T1 + T2: scaler + reliability mask + PCA
                        features = preprocessor.transform(vol_z)
                    else:
                        # No preprocessing: flatten
                        features = vol.flatten()
                    
                    # Get multi-layer targets
                    y_dict = multilayer_cache[nsd_id]
                    
                    X_list.append(features)
                    for layer_name in Y_dict_lists:
                        Y_dict_lists[layer_name].append(y_dict[layer_name])
                    nsd_ids_list.append(nsd_id)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract sample {nsd_id} from {beta_path}[{beta_index}]: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Failed to load {beta_path}: {e}")
            continue
    
    X = np.array(X_list, dtype=np.float32)
    Y_dict = {k: np.array(v, dtype=np.float32) for k, v in Y_dict_lists.items()}
    nsd_ids = np.array(nsd_ids_list, dtype=np.int64)
    
    logger.info(f"  {desc}: {len(X)} samples successfully extracted")
    if len(X) > 0:
        logger.info(f"    fMRI shape: {X.shape}")
        for layer_name, layer_data in Y_dict.items():
            logger.info(f"    {layer_name} shape: {layer_data.shape}")
    
    return X, Y_dict, nsd_ids


def pretrain_ssl(
    model: TwoStageEncoder,
    pretrainer: SelfSupervisedPretrainer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int
) -> None:
    """Self-supervised pretraining of Stage 1."""
    logger.info(f"Starting self-supervised pretraining for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.stage1.train()
        total_loss = 0.0
        
        pbar = tqdm(loader, desc=f"SSL Epoch {epoch}/{epochs}", leave=False)
        for X_batch, _ in pbar:
            X_batch = X_batch.to(device)
            
            optimizer.zero_grad()
            
            # Self-supervised forward pass
            x_corrupted, x_reconstructed, x_target = pretrainer(X_batch)
            
            # Reconstruction loss (MSE)
            loss = nn.functional.mse_loss(x_reconstructed, x_target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.stage1.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item() * len(X_batch)
            pbar.set_postfix({"ssl_loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(loader.dataset)
        logger.info(f"SSL Epoch {epoch}/{epochs}: Loss = {avg_loss:.4f}")
    
    logger.info("Self-supervised pretraining completed!")


def load_config_from_yaml(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_config_and_args(config: Dict, args: argparse.Namespace) -> argparse.Namespace:
    """Merge YAML config with command-line arguments (CLI takes precedence)."""
    # Handle nested config structure
    if "preprocessing" in config:
        for key, value in config["preprocessing"].items():
            arg_name = key.replace("-", "_")
            if not hasattr(args, arg_name) or getattr(args, arg_name) is None:
                setattr(args, arg_name, value)
    
    if "encoder" in config:
        for key, value in config["encoder"].items():
            arg_name = key.replace("-", "_")
            if not hasattr(args, arg_name) or getattr(args, arg_name) is None:
                # Handle boolean flags specially
                if key == "self_supervised":
                    if value and not args.self_supervised:
                        setattr(args, arg_name, value)
                elif key == "freeze_stage1":
                    if value and not args.freeze_stage1:
                        setattr(args, arg_name, value)
                else:
                    setattr(args, arg_name, value)
    
    if "loss" in config:
        for key, value in config["loss"].items():
            arg_name = key.replace("-", "_")
            if not hasattr(args, arg_name) or getattr(args, arg_name) is None:
                setattr(args, arg_name, value)
    
    if "training" in config:
        for key, value in config["training"].items():
            # Map config keys to arg names
            key_map = {
                "learning_rate": "lr",
                "weight_decay": "wd"
            }
            arg_name = key_map.get(key, key.replace("-", "_"))
            
            if not hasattr(args, arg_name) or getattr(args, arg_name) is None:
                setattr(args, arg_name, value)
    
    if "dataset" in config:
        if "subject" in config["dataset"] and not hasattr(args, "subject"):
            setattr(args, "subject", config["dataset"]["subject"])
    
    # Multi-layer supervision config
    if "multi_layer" in config:
        ml_config = config["multi_layer"]
        if ml_config.get("enabled", False) and not args.multi_layer:
            args.multi_layer = True
        if "cache_path" in ml_config and not hasattr(args, "multilayer_cache"):
            args.multilayer_cache = ml_config["cache_path"]
    
    # Set use_preproc if pca_k is specified
    if hasattr(args, "pca_k") and args.pca_k is not None:
        args.use_preproc = True
    
    return args


def main():
    parser = argparse.ArgumentParser(description="Train two-stage encoder for fMRI → CLIP")
    
    # Config file support
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML config file (overrides defaults, CLI args override config)")
    
    # Data paths
    parser.add_argument("--index-root", default="data/indices/nsd_index")
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--clip-cache", default="outputs/clip_cache/clip.parquet")
    
    # Preprocessing
    parser.add_argument("--use-preproc", action="store_true")
    parser.add_argument("--pca-k", type=int, help="PCA components (256/512/768)")
    parser.add_argument("--preproc-dir", default="outputs/preproc")
    
    # Model architecture
    parser.add_argument("--latent-dim", type=int, default=512,
                       help="Latent representation dimension (512/768/1024)")
    parser.add_argument("--n-blocks", type=int, default=4,
                       help="Number of residual blocks (3-6)")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--head-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--head-hidden", type=int, default=512)
    
    # Self-supervised pretraining
    parser.add_argument("--self-supervised", action="store_true",
                       help="Enable self-supervised pretraining")
    parser.add_argument("--ssl-objective", choices=["masked", "denoising"], default="masked")
    parser.add_argument("--ssl-epochs", type=int, default=20)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--noise-std", type=float, default=0.1)
    
    # Staged training
    parser.add_argument("--freeze-stage1", action="store_true",
                       help="Freeze Stage 1 after pretraining")
    parser.add_argument("--stage2-epochs", type=int,
                       help="Epochs for Stage 2 training (if freezing Stage 1)")
    
    # Loss function
    parser.add_argument("--mse-weight", type=float, default=0.3)
    parser.add_argument("--cosine-weight", type=float, default=0.3)
    parser.add_argument("--info-nce-weight", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=0.05)
    
    # Multi-layer supervision (Phase 3)
    parser.add_argument("--multi-layer", action="store_true",
                       help="Enable multi-layer CLIP supervision")
    parser.add_argument("--multilayer-cache", type=str,
                       default="cache/clip_embeddings/nsd_clipcache_multilayer.parquet",
                       help="Path to multi-layer CLIP cache")
    
    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    
    # Data limits (for testing)
    parser.add_argument("--limit", type=int, help="Limit samples for quick testing")
    
    # Output
    parser.add_argument("--checkpoint-dir", default="checkpoints/two_stage")
    parser.add_argument("--save-name", help="Custom checkpoint name")
    parser.add_argument("--output-dir", help="Output directory (alternative to checkpoint-dir)")
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config_from_yaml(args.config)
        args = merge_config_and_args(config, args)
        logger.info("Configuration loaded and merged with CLI arguments")
    
    # Use output-dir if provided (for compatibility with config files)
    if args.output_dir:
        args.checkpoint_dir = args.output_dir
    
    # Device setup
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    # Seed for reproducibility
    torch_seed_all(args.seed)
    
    # Load data
    logger.info(f"Loading index for {args.subject}...")
    df = read_subject_index(args.index_root, args.subject)
    
    if args.limit:
        df = df.head(args.limit)
        logger.info(f"Limited to {len(df)} samples for testing")
    
    # Train/val/test split
    train_df, val_df, test_df = train_val_test_split(df, random_seed=args.seed)
    
    # Load CLIP cache
    logger.info("Loading CLIP cache...")
    clip_cache = CLIPCache(args.clip_cache)
    
    # Setup preprocessing
    preprocessor = None
    if args.use_preproc or args.pca_k:
        logger.info("Setting up preprocessing...")
        preprocessor = NSDPreprocessor(args.subject, out_dir=args.preproc_dir)
        
        # Check if artifacts exist
        if not preprocessor.meta_path.exists():
            logger.error(f"Preprocessing artifacts not found at {preprocessor.out_dir}")
            logger.error("Please run preprocessing first:")
            logger.error(f"  python scripts/nsd_fit_preproc.py --subject {args.subject}")
            sys.exit(1)
        
        preprocessor.load_artifacts()
        logger.info(f"Loaded preprocessing: PCA k={preprocessor.pca_info_.get('n_components_eff', 'N/A')}")
    
    # Setup NIfTI loader
    fs = get_s3_filesystem()
    nifti_loader = NIfTILoader(fs)
    
    # Check if multi-layer mode is enabled
    if args.multi_layer:
        logger.info("=" * 70)
        logger.info("MULTI-LAYER SUPERVISION MODE ENABLED")
        logger.info("=" * 70)
        
        # Load multi-layer CLIP cache
        multilayer_cache = load_multilayer_clip_cache(args.multilayer_cache)
        
        # Extract features and multi-layer targets
        logger.info("Extracting training data...")
        X_train, Y_train_dict, _ = extract_features_and_multilayer_targets(
            train_df, nifti_loader, preprocessor, multilayer_cache, desc="train"
        )
        
        logger.info("Extracting validation data...")
        X_val, Y_val_dict, _ = extract_features_and_multilayer_targets(
            val_df, nifti_loader, preprocessor, multilayer_cache, desc="val"
        )
        
        logger.info("Extracting test data...")
        X_test, Y_test_dict, nsd_ids_test = extract_features_and_multilayer_targets(
            test_df, nifti_loader, preprocessor, multilayer_cache, desc="test"
        )
        
        # Create datasets with dict targets
        class MultiLayerDataset(torch.utils.data.Dataset):
            def __init__(self, X, Y_dict):
                self.X = torch.from_numpy(X).float()
                self.Y_dict = {k: torch.from_numpy(v).float() for k, v in Y_dict.items()}
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                return self.X[idx], {k: v[idx] for k, v in self.Y_dict.items()}
        
        train_dataset = MultiLayerDataset(X_train, Y_train_dict)
        val_dataset = MultiLayerDataset(X_val, Y_val_dict)
        test_dataset = MultiLayerDataset(X_test, Y_test_dict)
        
    else:
        # Standard single-layer mode
        logger.info("Standard single-layer CLIP supervision")
        
        # Load regular CLIP cache
        clip_cache = CLIPCache(args.clip_cache)
        
        # Extract features and targets
        logger.info("Extracting training data...")
        X_train, Y_train, _ = extract_features_and_targets(
            train_df, nifti_loader, preprocessor, clip_cache, desc="train"
        )
        
        logger.info("Extracting validation data...")
        X_val, Y_val, _ = extract_features_and_targets(
            val_df, nifti_loader, preprocessor, clip_cache, desc="val"
        )
        
        logger.info("Extracting test data...")
        X_test, Y_test, nsd_ids_test = extract_features_and_targets(
            test_df, nifti_loader, preprocessor, clip_cache, desc="test"
        )
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(Y_train).float()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(Y_val).float()
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_test).float(),
            torch.from_numpy(Y_test).float()
        )
    
    # Create data loaders (same for both modes)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    input_dim = X_train.shape[1]
    
    if args.multi_layer:
        # Get shared_head_backbone from config (Phase 2)
        shared_head_backbone = config.get('encoder', {}).get('shared_head_backbone', False)
        
        logger.info(f"Creating MultiLayerTwoStageEncoder: input_dim={input_dim}, latent_dim={args.latent_dim}, n_blocks={args.n_blocks}")
        logger.info(f"  Shared head backbone: {shared_head_backbone}")
        
        model = MultiLayerTwoStageEncoder(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            head_type=args.head_type,
            head_hidden_dim=args.head_hidden,
            shared_head_backbone=shared_head_backbone
        ).to(device)
    else:
        logger.info(f"Creating TwoStageEncoder: input_dim={input_dim}, latent_dim={args.latent_dim}, n_blocks={args.n_blocks}")
        
        model = TwoStageEncoder(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            head_type=args.head_type,
            head_hidden_dim=args.head_hidden
        ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Self-supervised pretraining
    if args.self_supervised:
        logger.info(f"Setting up self-supervised pretraining ({args.ssl_objective})...")
        
        pretrainer = SelfSupervisedPretrainer(
            encoder=model.stage1,
            reconstruction_dim=input_dim,
            objective=args.ssl_objective,
            mask_ratio=args.mask_ratio,
            noise_std=args.noise_std
        ).to(device)
        
        ssl_optimizer = AdamW(
            pretrainer.parameters(),
            lr=args.lr,
            weight_decay=args.wd
        )
        
        pretrain_ssl(
            model=model,
            pretrainer=pretrainer,
            loader=train_loader,
            optimizer=ssl_optimizer,
            device=device,
            epochs=args.ssl_epochs
        )
    
    # Staged training: freeze Stage 1 if requested
    if args.freeze_stage1:
        model.freeze_stage1()
        if args.stage2_epochs:
            args.epochs = args.stage2_epochs
            logger.info(f"Stage 1 frozen, training Stage 2 for {args.epochs} epochs")
    
    # Setup optimizer and loss
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.wd
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    if args.multi_layer:
        # Multi-layer loss from config
        ml_config = config.get('multi_layer', {})
        layer_weights = ml_config.get('layer_weights', {
            'layer_4': 0.15,
            'layer_8': 0.20,
            'layer_12': 0.25,
            'final': 0.40
        })
        use_learnable_weights = ml_config.get('use_learnable_weights', False)
        use_mse = ml_config.get('use_mse', False)
        mse_weight = ml_config.get('mse_weight', 0.1)
        
        # Phase 3: Multi-layer InfoNCE parameters
        loss_config = config.get('loss', {})
        use_multilayer_infonce = loss_config.get('use_multilayer_infonce', False)
        infonce_weight = loss_config.get('info_nce_weight', 0.4) * 0.5  # Use half of standard InfoNCE weight
        infonce_temperature = loss_config.get('temperature', 0.05)
        infonce_combination = loss_config.get('infonce_combination', 'weighted_pool')
        
        criterion = MultiLayerLoss(
            layer_weights=layer_weights,
            use_mse=use_mse,
            mse_weight=mse_weight,
            use_learnable_weights=use_learnable_weights,
            use_multilayer_infonce=use_multilayer_infonce,
            infonce_weight=infonce_weight,
            infonce_temperature=infonce_temperature,
            infonce_combination=infonce_combination
        )
        
        if use_learnable_weights:
            logger.info(f"Using MultiLayerLoss with LEARNABLE weights (initialized from: {layer_weights})")
        else:
            logger.info(f"Using MultiLayerLoss with FIXED weights: {layer_weights}")
        
        if use_multilayer_infonce:
            logger.info(f"Phase 3: Multi-layer InfoNCE ENABLED (weight={infonce_weight:.3f}, strategy={infonce_combination})")
    else:
        criterion = MultiLoss(
            mse_weight=args.mse_weight,
            cosine_weight=args.cosine_weight,
            info_nce_weight=args.info_nce_weight,
            temperature=args.temperature
        )
        logger.info(f"Loss weights: MSE={args.mse_weight}, Cosine={args.cosine_weight}, InfoNCE={args.info_nce_weight}")
    
    # Training loop with early stopping
    best_val_cosine = -1.0
    best_epoch = 0
    patience_counter = 0
    
    logger.info("Starting training...")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        if args.multi_layer:
            train_loss, train_components = train_epoch_multilayer(
                model, train_loader, optimizer, criterion, device, epoch
            )
            # Validate
            val_metrics = evaluate_epoch_multilayer(model, val_loader, device)
        else:
            train_loss, train_components = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )
            # Validate
            val_metrics = evaluate_epoch(model, val_loader, device)
        
        val_cosine = val_metrics["cosine"]
        
        # Log (handle both single-layer and multi-layer modes)
        if args.multi_layer:
            # Multi-layer components: layer_4, layer_8, layer_12, final
            logger.info(
                f"Epoch {epoch}/{args.epochs}: "
                f"Train Loss={train_loss:.4f} "
                f"(L4={train_components.get('layer_4', 0):.3f}, "
                f"L8={train_components.get('layer_8', 0):.3f}, "
                f"L12={train_components.get('layer_12', 0):.3f}, "
                f"Fin={train_components.get('final', 0):.3f}), "
                f"Val Cosine={val_cosine:.4f}"
            )
            
            # Log effective weights periodically (every 5 epochs) if learnable
            if hasattr(criterion, 'use_learnable_weights') and criterion.use_learnable_weights:
                if epoch % 5 == 0 or epoch == 1:
                    eff_weights = criterion.get_effective_weights()
                    logger.info(
                        f"  → Learned weights: "
                        f"L4={eff_weights['layer_4']:.3f}, "
                        f"L8={eff_weights['layer_8']:.3f}, "
                        f"L12={eff_weights['layer_12']:.3f}, "
                        f"Fin={eff_weights['final']:.3f}"
                    )
        else:
            # Single-layer components: mse, cosine, info_nce
            logger.info(
                f"Epoch {epoch}/{args.epochs}: "
                f"Train Loss={train_loss:.4f} "
                f"(MSE={train_components.get('mse', 0):.4f}, "
                f"Cos={train_components.get('cosine', 0):.4f}, "
                f"NCE={train_components.get('info_nce', 0):.4f}), "
                f"Val Cosine={val_cosine:.4f}"
            )
        
        # Early stopping
        if val_cosine > best_val_cosine:
            best_val_cosine = val_cosine
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            checkpoint_dir = Path(args.checkpoint_dir) / args.subject
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            save_name = args.save_name or "two_stage_best.pt"
            checkpoint_path = checkpoint_dir / save_name
            
            meta = {
                "input_dim": input_dim,
                "latent_dim": args.latent_dim,
                "n_blocks": args.n_blocks,
                "dropout": args.dropout,
                "head_type": args.head_type,
                "head_hidden_dim": args.head_hidden,
                "shared_head_backbone": shared_head_backbone if args.multi_layer else False,
                "best_epoch": best_epoch,
                "best_val_cosine": best_val_cosine,
                "pca_k": args.pca_k,
                "self_supervised": args.self_supervised,
                "ssl_objective": args.ssl_objective if args.self_supervised else None
            }
            
            save_two_stage_encoder(model, str(checkpoint_path), meta)
            logger.info(f"✅ Saved best model (epoch {epoch}, val_cosine={val_cosine:.4f})")
        
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch} (best: {best_epoch})")
                break
        
        scheduler.step()
    
    # Load best model and evaluate on test set
    logger.info(f"Loading best model from epoch {best_epoch}...")
    
    if args.multi_layer:
        model, meta = load_multilayer_two_stage_encoder(str(checkpoint_path), map_location=device)
        model = model.to(device)
        test_metrics = evaluate_epoch_multilayer(model, test_loader, device)
        
        logger.info("=" * 80)
        logger.info("FINAL TEST RESULTS (Multi-Layer)")
        logger.info("=" * 80)
        logger.info(f"Test Cosine (final layer): {test_metrics['cosine']:.4f}")
        
        # Save evaluation report
        report = {
            "model": "MultiLayerTwoStageEncoder",
            "subject": args.subject,
            "architecture": {
                "input_dim": input_dim,
                "latent_dim": args.latent_dim,
                "n_blocks": args.n_blocks,
                "dropout": args.dropout,
                "layer_dims": {
                    "layer_4": 768,
                    "layer_8": 768,
                    "layer_12": 768,
                    "final": 512
                }
            },
            "training": {
                "best_epoch": best_epoch,
                "best_val_cosine": best_val_cosine,
                "multi_layer": True,
                "layer_weights": config["multi_layer"]["layer_weights"]
            },
            "test_metrics": test_metrics,
            "checkpoint": str(checkpoint_path)
        }
    else:
        model, meta = load_two_stage_encoder(str(checkpoint_path), map_location=device)
        model = model.to(device)
        test_metrics = evaluate_epoch(model, test_loader, device)
        
        logger.info("=" * 80)
        logger.info("FINAL TEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"Test Cosine: {test_metrics['cosine']:.4f}")
        logger.info(f"Test MSE: {test_metrics['mse']:.6f}")
        
        # Save evaluation report
        report = {
            "model": "TwoStageEncoder",
            "subject": args.subject,
            "architecture": {
                "input_dim": input_dim,
                "latent_dim": args.latent_dim,
                "n_blocks": args.n_blocks,
                "dropout": args.dropout,
                "head_type": args.head_type
            },
            "training": {
                "best_epoch": best_epoch,
                "best_val_cosine": best_val_cosine,
                "self_supervised": args.self_supervised,
                "ssl_objective": args.ssl_objective if args.self_supervised else None,
                "freeze_stage1": args.freeze_stage1
            },
            "test_metrics": test_metrics,
            "checkpoint": str(checkpoint_path)
        }

    
    report_path = checkpoint_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Saved evaluation report to {report_path}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
