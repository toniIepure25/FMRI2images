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
    ProbabilisticMultiLayerTwoStageEncoder,
    SelfSupervisedPretrainer,
    save_two_stage_encoder,
    load_two_stage_encoder,
    load_multilayer_two_stage_encoder,
    load_probabilistic_encoder
)
from fmri2img.training.losses import MultiLoss, MultiLayerLoss, ProbabilisticMultiLayerLoss, compute_multiloss
from fmri2img.training.phase4_losses import BranchWeightedMultiLayerLoss
from fmri2img.data.streaming_dataset import StreamingMultiLayerDataset
from fmri2img.models.train_utils import (
    extract_features_and_targets,
    train_val_test_split,
    torch_seed_all
)
from fmri2img.models.ridge import evaluate_predictions
from fmri2img.eval.retrieval import retrieval_at_k, compute_ranking_metrics


class LazyMultiLayerDataset(torch.utils.data.Dataset):
    """
    Memory-efficient dataset that loads fMRI data on-the-fly instead of preloading.
    
    This prevents OOM errors when working with large datasets by:
    - Loading beta files only when needed
    - Caching recently used files (LRU cache)
    - Processing one sample at a time
    
    Args:
        df: DataFrame with beta_path, beta_index, nsdId
        nifti_loader: NIfTI file loader
        preprocessor: NSD preprocessor (fitted)
        multilayer_cache: Multi-layer CLIP embeddings
        text_clip_cache: Optional text-CLIP embeddings
        cache_size: Number of beta files to keep in memory (default: 5)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        nifti_loader,
        preprocessor,
        multilayer_cache: Dict[int, Dict[str, np.ndarray]],
        text_clip_cache: Optional[Dict[int, np.ndarray]] = None,
        cache_size: int = 5
    ):
        self.df = df.reset_index(drop=True)
        self.nifti_loader = nifti_loader
        self.preprocessor = preprocessor
        self.multilayer_cache = multilayer_cache
        self.text_clip_cache = text_clip_cache
        
        # Filter samples that have multi-layer embeddings
        valid_indices = []
        for idx, row in self.df.iterrows():
            nsd_id = int(row["nsdId"])
            if nsd_id in multilayer_cache:
                valid_indices.append(idx)
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        logger.info(f"LazyDataset: {len(self.df)} valid samples (have multi-layer embeddings)")
        
        # LRU cache for beta files (keep last N files in memory)
        from collections import OrderedDict
        self.beta_cache = OrderedDict()
        self.cache_size = cache_size
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        beta_path = row["beta_path"]
        beta_index = int(row["beta_index"])
        nsd_id = int(row["nsdId"])
        
        # Load beta file (with LRU caching)
        if beta_path not in self.beta_cache:
            # Load new file
            img = self.nifti_loader.load(beta_path)
            data_4d = img.get_fdata().astype(np.float32)
            
            # Add to cache
            self.beta_cache[beta_path] = data_4d
            
            # Remove oldest if cache full
            if len(self.beta_cache) > self.cache_size:
                self.beta_cache.popitem(last=False)
        else:
            # Move to end (mark as recently used)
            self.beta_cache.move_to_end(beta_path)
            data_4d = self.beta_cache[beta_path]
        
        # Extract volume
        vol = data_4d[..., beta_index]
        
        # Preprocess
        if self.preprocessor and self.preprocessor.is_fitted_:
            vol_z = self.preprocessor.transform_T0(vol)
            features = self.preprocessor.transform(vol_z)
        else:
            features = vol.flatten()
        
        # Get multi-layer targets
        targets = self.multilayer_cache[nsd_id]
        Y_dict = {
            k: torch.from_numpy(v).float() 
            for k, v in targets.items()
        }
        
        # Add text-CLIP if available
        if self.text_clip_cache is not None and nsd_id in self.text_clip_cache:
            Y_dict['text'] = torch.from_numpy(self.text_clip_cache[nsd_id]).float()
        
        X = torch.from_numpy(features).float()
        
        return X, Y_dict


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
    loss_components_sum = {"mse": 0.0, "cosine": 0.0, "info_nce": 0.0, "brain": 0.0}
    n_batches = 0
    
    # Check if brain-consistency is enabled
    use_brain_loss = criterion.brain_consistency_weight > 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for X_batch, Y_batch in pbar:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        
        optimizer.zero_grad()
        Y_pred = model(X_batch)
        
        # Compute loss with components
        # Pass fmri_input if brain-consistency is enabled
        if use_brain_loss:
            loss, components = criterion(
                Y_pred, Y_batch, 
                fmri_input=X_batch,  # Original fMRI PCA for cycle loss
                return_components=True
            )
        else:
            loss, components = criterion(Y_pred, Y_batch, return_components=True)
        
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
        pbar_dict = {
            "loss": f"{loss.item():.4f}",
            "mse": f"{components['mse']:.4f}",
            "cos": f"{components['cosine']:.4f}",
            "nce": f"{components['info_nce']:.4f}"
        }
        if use_brain_loss:
            pbar_dict["brain"] = f"{components['brain']:.4f}"
        pbar.set_postfix(pbar_dict)
    
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
    epoch: int,
    probabilistic: bool = False
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
        
        # Phase 3: Probabilistic model returns (outputs, kl_loss)
        if probabilistic:
            Y_pred_dict, kl_loss = model(X_batch, sample=True, return_kl=True)
            # Pass kl_loss to criterion (not model)
            loss, components = criterion(Y_pred_dict, Y_batch_dict, kl_loss, 
                                         current_epoch=epoch, return_components=True)
        else:
            # Phase 2: Deterministic model returns outputs only
            Y_pred_dict = model(X_batch)
            # Pass model for Phase 2 multi-layer InfoNCE
            loss, components = criterion(Y_pred_dict, Y_batch_dict, model=model, 
                                         current_epoch=epoch, return_components=True)
        
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
        postfix_dict = {
            "loss": f"{loss.item():.4f}",
            "l4": f"{components.get('layer_4', 0):.3f}",
            "l8": f"{components.get('layer_8', 0):.3f}",
            "l12": f"{components.get('layer_12', 0):.3f}",
            "fin": f"{components.get('final', 0):.3f}"
        }
        # Add KL loss for Phase 3
        if probabilistic and 'kl' in components:
            postfix_dict["kl"] = f"{components['kl']:.4f}"
            postfix_dict["β"] = f"{components.get('kl_weight', 0):.4f}"
        pbar.set_postfix(postfix_dict)
    
    # Average over all samples
    n_samples = len(loader.dataset)
    avg_loss = total_loss / n_samples
    avg_components = {k: v / n_samples for k, v in loss_components_sum.items()}
    
    return avg_loss, avg_components


@torch.no_grad()
def evaluate_epoch_multilayer(
    model: MultiLayerTwoStageEncoder,
    loader: DataLoader,
    device: str,
    probabilistic: bool = False
) -> Dict:
    """Evaluate multi-layer model on validation/test set (using final layer only)."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, Y_batch_dict in loader:
            X_batch = X_batch.to(device)
            
            # Phase 3: Probabilistic model returns (outputs, kl_loss)
            if probabilistic:
                Y_pred_dict, _ = model(X_batch, sample=False, return_kl=False)  # Use mean for eval
            else:
                # Phase 2: Deterministic model
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
    desc: str = "data",
    text_clip_cache: Optional[Dict[int, np.ndarray]] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Extract fMRI features and multi-layer CLIP targets.
    Uses same optimization as extract_features_and_targets: group by file to load each once.
    
    Args:
        df: DataFrame with beta_path, beta_index, nsdId
        nifti_loader: NIfTI file loader
        preprocessor: NSD preprocessor
        multilayer_cache: Multi-layer CLIP embeddings {nsd_id: {'layer_4': ..., 'final': ...}}
        desc: Description for logging
        text_clip_cache: Optional text-CLIP embeddings {nsd_id: (512,)} for Phase 2
    
    Returns:
        X: fMRI features (N, fmri_dim)
        Y_dict: Dict of CLIP targets {'layer_4': (N, 768), ..., 'final': (N, 512), 'text': (N, 512)}
        nsd_ids: NSD stimulus IDs (N,)
    """
    from collections import defaultdict
    
    X_list = []
    Y_dict_lists = {'layer_4': [], 'layer_8': [], 'layer_12': [], 'final': []}
    
    # Phase 2: Add text-CLIP support
    if text_clip_cache is not None:
        Y_dict_lists['text'] = []
    
    nsd_ids_list = []
    
    # Group samples by beta_path to load each file only once (OPTIMIZATION)
    samples_by_file = defaultdict(list)
    for idx, row in df.iterrows():
        nsd_id = int(row["nsdId"])
        
        # Skip if no multi-layer embedding
        if nsd_id not in multilayer_cache:
            continue
        
        # Phase 2: Skip if text-CLIP is required but not available
        if text_clip_cache is not None and nsd_id not in text_clip_cache:
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
                    y_dict = multilayer_cache[nsd_id].copy()  # Always copy to avoid modifying cache
                    
                    # Phase 2: Add text-CLIP target (guaranteed to exist due to pre-filtering)
                    if text_clip_cache is not None:
                        y_dict['text'] = text_clip_cache[nsd_id]
                    
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
    parser.add_argument("--streaming", action="store_true",
                       help="Use streaming dataset (memory-efficient for full 30K dataset)")
    
    # Phase 2: Multi-task semantics (text-CLIP)
    parser.add_argument("--predict-text-clip", action="store_true",
                       help="Enable text-CLIP prediction alongside image-CLIP (Phase 2)")
    parser.add_argument("--text-clip-cache", type=str,
                       default="cache/clip_embeddings/text_clip.parquet",
                       help="Path to text-CLIP embeddings (BLIP-2 captions + CLIP text)")
    parser.add_argument("--text-clip-weight", type=float, default=0.3,
                       help="Weight for text-CLIP loss (0.0-1.0)")
    
    # Phase 3: Probabilistic predictions with uncertainty
    parser.add_argument("--probabilistic", action="store_true",
                       help="Enable probabilistic encoder with uncertainty modeling (Phase 3)")
    parser.add_argument("--kl-weight", type=float, default=1e-4,
                       help="Initial weight for KL divergence loss")
    parser.add_argument("--kl-anneal-epochs", type=int, default=10,
                       help="Number of epochs for KL weight annealing")
    
    # Phase 4: Structural vs Semantic Branches
    parser.add_argument("--use-phase4", action="store_true",
                       help="Enable Phase 4 structural/semantic branch architecture")
    parser.add_argument("--structural-dim", type=int, default=256,
                       help="Structural branch latent dimension (for early layers L4/L8)")
    parser.add_argument("--semantic-dim", type=int, default=512,
                       help="Semantic branch latent dimension (for late layers L12/final/text)")
    parser.add_argument("--structural-weight", type=float, default=1.0,
                       help="Weight for structural branch loss (early layers)")
    parser.add_argument("--semantic-weight", type=float, default=1.0,
                       help="Weight for semantic branch loss (late layers + text)")
    
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
    
    # Initialize config as empty dict
    config = {}
    
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
        pca_k = preprocessor.pca_info_.get('k_eff', preprocessor.pca_.n_components_ if preprocessor.pca_ else None)
        logger.info(f"Loaded preprocessing: PCA k={pca_k if pca_k else 'disabled'}")
    
    # Setup NIfTI loader
    fs = get_s3_filesystem()
    nifti_loader = NIfTILoader(fs)
    
    # Check if multi-layer mode is enabled
    if args.multi_layer:
        logger.info("=" * 70)
        logger.info("MULTI-LAYER SUPERVISION MODE ENABLED")
        logger.info("=" * 70)
        
        # Check if streaming mode is enabled
        if args.streaming:
            logger.info("=" * 80)
            logger.info("STREAMING MODE (Memory-Efficient for Full Dataset)")
            logger.info("=" * 80)
            logger.info("  Data will be loaded on-demand during training")
            logger.info("  Memory usage: ~2-4 GB (vs ~15-20 GB eager loading)")
            logger.info("  Trade-off: Slightly slower per epoch due to disk I/O")
            logger.info("  Caches NIfTI files (LRU, 5 files max)")
            
            # Create streaming datasets
            train_dataset = StreamingMultiLayerDataset(
                train_df, nifti_loader, preprocessor,
                multilayer_cache_path=args.multilayer_cache,
                text_clip_cache_path=args.text_clip_cache if args.predict_text_clip else None,
                desc="train"
            )
            val_dataset = StreamingMultiLayerDataset(
                val_df, nifti_loader, preprocessor,
                multilayer_cache_path=args.multilayer_cache,
                text_clip_cache_path=args.text_clip_cache if args.predict_text_clip else None,
                desc="val"
            )
            test_dataset = StreamingMultiLayerDataset(
                test_df, nifti_loader, preprocessor,
                multilayer_cache_path=args.multilayer_cache,
                text_clip_cache_path=args.text_clip_cache if args.predict_text_clip else None,
                desc="test"
            )
            
            logger.info(f"  Train: {len(train_dataset)} samples")
            logger.info(f"  Val: {len(val_dataset)} samples")
            logger.info(f"  Test: {len(test_dataset)} samples")
            
        else:
            # Original eager loading (fast but memory-intensive)
            logger.info("=" * 80)
            logger.info("EAGER LOADING MODE (High Memory)")
            logger.info("=" * 80)
            logger.info("  All data will be loaded into RAM upfront")
            logger.info("  Memory usage: ~15-20 GB for full dataset")
            logger.info("  For low-memory systems, use --streaming flag")
            
            # Load multi-layer CLIP cache
            multilayer_cache = load_multilayer_clip_cache(args.multilayer_cache)
            
            # Phase 2: Load text-CLIP cache if enabled
            text_clip_cache = None
            if args.predict_text_clip and args.text_clip_cache:
                logger.info("=" * 70)
                logger.info("PHASE 2: TEXT-CLIP MULTI-TASK MODE ENABLED")
                logger.info("=" * 70)
                logger.info(f"Loading text-CLIP cache from {args.text_clip_cache}...")
                
                text_clip_df = pd.read_parquet(args.text_clip_cache)
                text_clip_cache = {}
                
                # Handle both nsdId and nsd_id column names (backward compatibility)
                nsd_col = 'nsd_id' if 'nsd_id' in text_clip_df.columns else 'nsdId'
                
                for _, row in text_clip_df.iterrows():
                    nsd_id = int(row[nsd_col])
                    text_emb = np.array(row['text_clip_embedding'], dtype=np.float32)
                    text_clip_cache[nsd_id] = text_emb
                
                logger.info(f"  Loaded {len(text_clip_cache)} text-CLIP embeddings (dim={text_clip_cache[list(text_clip_cache.keys())[0]].shape[0]})")
                logger.info(f"  Text-CLIP weight: {args.text_clip_weight}")
            
            # Extract features and multi-layer targets (eager loading - faster but uses ~13 GB RAM)
            logger.info("Extracting training data...")
            X_train, Y_train_dict, _ = extract_features_and_multilayer_targets(
                train_df, nifti_loader, preprocessor, multilayer_cache, desc="train", text_clip_cache=text_clip_cache
            )
            
            logger.info("Extracting validation data...")
            X_val, Y_val_dict, _ = extract_features_and_multilayer_targets(
                val_df, nifti_loader, preprocessor, multilayer_cache, desc="val", text_clip_cache=text_clip_cache
            )
            
            logger.info("Extracting test data...")
            X_test, Y_test_dict, nsd_ids_test = extract_features_and_multilayer_targets(
                test_df, nifti_loader, preprocessor, multilayer_cache, desc="test", text_clip_cache=text_clip_cache
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
    
    # Create data loaders (single worker for streaming to avoid system overload)
    num_workers = 0  # Single worker - multi-worker can overload memory on limited systems
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    # Create model - get input_dim from first sample
    logger.info("Getting input dimension from first training sample...")
    first_sample_x, _ = train_dataset[0]
    input_dim = first_sample_x.shape[0]
    logger.info(f"Input dimension: {input_dim}")
    
    if args.multi_layer:
        # Get shared_head_backbone from config (Phase 2)
        shared_head_backbone = config.get('encoder', {}).get('shared_head_backbone', False)
        
        # Phase 4: Structural/Semantic Branch Architecture
        if args.use_phase4:
            from fmri2img.models.phase4_encoder import StructuralSemanticEncoder
            
            logger.info("Creating StructuralSemanticEncoder (Phase 4): input_dim={}, latent_dim={}".format(input_dim, args.latent_dim))
            logger.info(f"  Structural branch: {args.structural_dim}-D → layer_4, layer_8")
            logger.info(f"  Semantic branch: {args.semantic_dim}-D → layer_12, final, text")
            logger.info(f"  Probabilistic mode: {args.probabilistic}")
            
            model = StructuralSemanticEncoder(
                input_dim=input_dim,
                latent_dim=args.latent_dim,
                structural_dim=args.structural_dim,
                semantic_dim=args.semantic_dim,
                n_blocks=args.n_blocks,
                dropout=args.dropout,
                predict_text_clip=args.predict_text_clip,
                probabilistic=args.probabilistic,
                kl_weight=args.kl_weight if args.probabilistic else 0.0
            ).to(device)
            
        elif args.probabilistic:
            logger.info(f"Creating ProbabilisticMultiLayerTwoStageEncoder: input_dim={input_dim}, latent_dim={args.latent_dim}, n_blocks={args.n_blocks}")
            logger.info(f"  Shared head backbone: {shared_head_backbone}")
            logger.info(f"  Probabilistic mode: enabled (mu/logvar outputs)")
            
            model = ProbabilisticMultiLayerTwoStageEncoder(
                input_dim=input_dim,
                latent_dim=args.latent_dim,
                n_blocks=args.n_blocks,
                dropout=args.dropout,
                head_hidden_dim=args.head_hidden,
                predict_text_clip=args.predict_text_clip,  # Phase 3: Can still include text-CLIP
                kl_weight=args.kl_weight
            ).to(device)
        else:
            logger.info(f"Creating MultiLayerTwoStageEncoder: input_dim={input_dim}, latent_dim={args.latent_dim}, n_blocks={args.n_blocks}")
            logger.info(f"  Shared head backbone: {shared_head_backbone}")
            
            model = MultiLayerTwoStageEncoder(
                input_dim=input_dim,
                latent_dim=args.latent_dim,
                n_blocks=args.n_blocks,
                dropout=args.dropout,
                head_type=args.head_type,
                head_hidden_dim=args.head_hidden,
                shared_head_backbone=shared_head_backbone,
                predict_text_clip=args.predict_text_clip  # Phase 2: Enable text-CLIP head
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
    
    # Phase 1: Load CLIP→fMRI encoder for brain-consistency loss
    clip_to_fmri_encoder = None
    loss_config = config.get('loss', {})
    brain_consistency_weight = loss_config.get('brain_consistency_weight', 0.0)
    clip_to_fmri_path = loss_config.get('clip_to_fmri_encoder', None)
    
    if brain_consistency_weight > 0:
        if clip_to_fmri_path is None:
            logger.warning(
                f"brain_consistency_weight={brain_consistency_weight} but no clip_to_fmri_encoder path provided. "
                "Disabling brain-consistency loss."
            )
            brain_consistency_weight = 0.0
        else:
            from fmri2img.models.clip_to_fmri_encoder import load_clip_to_fmri_encoder
            
            clip_to_fmri_path = Path(clip_to_fmri_path)
            if not clip_to_fmri_path.exists():
                logger.warning(
                    f"CLIP→fMRI encoder not found at {clip_to_fmri_path}. "
                    f"Train it first with: python scripts/train_clip_to_fmri.py --subject {args.subject}"
                )
                logger.warning("Disabling brain-consistency loss.")
                brain_consistency_weight = 0.0
            else:
                logger.info("=" * 70)
                logger.info("PHASE 1: BRAIN-CONSISTENCY (CYCLE) LOSS ENABLED")
                logger.info("=" * 70)
                logger.info(f"Loading CLIP→fMRI encoder from {clip_to_fmri_path}")
                clip_to_fmri_encoder = load_clip_to_fmri_encoder(str(clip_to_fmri_path), device=device)
                clip_to_fmri_encoder.eval()
                for param in clip_to_fmri_encoder.parameters():
                    param.requires_grad = False
                logger.info(f"Brain-consistency weight: {brain_consistency_weight}")
                logger.info("This will regularize CLIP predictions to be brain-plausible.")
    
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
        use_multilayer_infonce = loss_config.get('use_multilayer_infonce', False)
        infonce_weight = loss_config.get('info_nce_weight', 0.4) * 0.5  # Use half of standard InfoNCE weight
        infonce_temperature = loss_config.get('temperature', 0.05)
        infonce_combination = loss_config.get('infonce_combination', 'weighted_pool')
        
        # Phase 2: Text-CLIP weight
        text_clip_weight = args.text_clip_weight if args.predict_text_clip else 0.0
        
        # Phase 4: Branch-weighted loss for structural/semantic branches
        if args.use_phase4:
            criterion = BranchWeightedMultiLayerLoss(
                layer_weights=layer_weights,
                structural_weight=args.structural_weight,
                semantic_weight=args.semantic_weight,
                use_mse=use_mse,
                mse_weight=mse_weight,
                probabilistic=args.probabilistic,
                kl_weight_max=args.kl_weight if args.probabilistic else 0.0,
                kl_anneal_epochs=args.kl_anneal_epochs if args.probabilistic else 0,
                text_clip_weight=text_clip_weight
            )
            logger.info(f"Using BranchWeightedMultiLayerLoss (Phase 4)")
            logger.info(f"  Structural weight: {args.structural_weight} (L4, L8)")
            logger.info(f"  Semantic weight: {args.semantic_weight} (L12, final, text)")
            if args.probabilistic:
                logger.info(f"  KL weight: {args.kl_weight} (annealing over {args.kl_anneal_epochs} epochs)")
        
        elif args.probabilistic:
            # Phase 3: Probabilistic loss with KL divergence
            # NOTE: ProbabilisticMultiLayerLoss does NOT support:
            #   - use_learnable_weights
            #   - use_multilayer_infonce, infonce_weight, infonce_temperature, infonce_combination
            # These are only for deterministic Phase 2 training
            criterion = ProbabilisticMultiLayerLoss(
                layer_weights=layer_weights,
                use_mse=use_mse,
                mse_weight=mse_weight,
                kl_weight_max=args.kl_weight,
                kl_anneal_epochs=args.kl_anneal_epochs,
                text_clip_weight=text_clip_weight
            )
            logger.info(f"Using ProbabilisticMultiLayerLoss (Phase 3)")
            logger.info(f"  KL weight: {args.kl_weight} (annealing over {args.kl_anneal_epochs} epochs)")
            logger.info(f"  Text-CLIP weight: {text_clip_weight}")
        else:
            # Phase 2: Deterministic multi-layer loss
            criterion = MultiLayerLoss(
                layer_weights=layer_weights,
                use_mse=use_mse,
                mse_weight=mse_weight,
                use_learnable_weights=use_learnable_weights,
                use_multilayer_infonce=use_multilayer_infonce,
                infonce_weight=infonce_weight,
                infonce_temperature=infonce_temperature,
                infonce_combination=infonce_combination,
                text_clip_weight=text_clip_weight  # Phase 2
            )
        
        if use_learnable_weights:
            logger.info(f"Using learnable weights (initialized from: {layer_weights})")
        else:
            logger.info(f"Using fixed weights: {layer_weights}")
        
        if use_multilayer_infonce:
            logger.info(f"Phase 3: Multi-layer InfoNCE ENABLED (weight={infonce_weight:.3f}, strategy={infonce_combination})")
    else:
        criterion = MultiLoss(
            mse_weight=args.mse_weight,
            cosine_weight=args.cosine_weight,
            info_nce_weight=args.info_nce_weight,
            temperature=args.temperature,
            brain_consistency_weight=brain_consistency_weight,
            clip_to_fmri_encoder=clip_to_fmri_encoder
        )
        logger.info(f"Loss weights: MSE={args.mse_weight}, Cosine={args.cosine_weight}, InfoNCE={args.info_nce_weight}")
        if brain_consistency_weight > 0:
            logger.info(f"Brain-consistency weight: {brain_consistency_weight}")
    
    # Training loop with early stopping
    best_val_cosine = -1.0
    best_epoch = 0
    patience_counter = 0
    
    logger.info("Starting training...")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        if args.multi_layer:
            train_loss, train_components = train_epoch_multilayer(
                model, train_loader, optimizer, criterion, device, epoch,
                probabilistic=args.probabilistic
            )
            # Validate
            val_metrics = evaluate_epoch_multilayer(
                model, val_loader, device,
                probabilistic=args.probabilistic
            )
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
            # Single-layer components: mse, cosine, info_nce, brain
            log_msg = (
                f"Epoch {epoch}/{args.epochs}: "
                f"Train Loss={train_loss:.4f} "
                f"(MSE={train_components.get('mse', 0):.4f}, "
                f"Cos={train_components.get('cosine', 0):.4f}, "
                f"NCE={train_components.get('info_nce', 0):.4f}"
            )
            # Add brain loss if enabled
            if brain_consistency_weight > 0:
                log_msg += f", Brain={train_components.get('brain', 0):.4f}"
            log_msg += f"), Val Cosine={val_cosine:.4f}"
            logger.info(log_msg)
        
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
                "probabilistic": args.probabilistic if args.multi_layer else False,
                "predict_text_clip": args.predict_text_clip if args.multi_layer else False,
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
        if args.probabilistic:
            model, meta = load_probabilistic_encoder(str(checkpoint_path), map_location=device)
        else:
            model, meta = load_multilayer_two_stage_encoder(str(checkpoint_path), map_location=device)
        model = model.to(device)
        test_metrics = evaluate_epoch_multilayer(
            model, test_loader, device,
            probabilistic=args.probabilistic
        )
        
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
                "layer_weights": config.get("multi_layer", {}).get("layer_weights", {
                    "layer_4": 0.15,
                    "layer_8": 0.2,
                    "layer_12": 0.25,
                    "final": 0.4
                })
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
