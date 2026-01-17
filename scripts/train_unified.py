"""
Unified Training Script for Research Experiments
================================================

Integrates all research components:
- Embedding preprocessing
- Memory queue
- Multiple loss functions (MSE, InfoNCE, Gaussian NLL, Gaussian-NCE, KL)
- Unified model (deterministic/Gaussian)
- Proper evaluation

Usage:
    python scripts/train_unified.py --config configs/experiments/exp0_baseline.yaml --gpu 0
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fmri2img.models.unified_model import create_model
from src.fmri2img.embedding_preproc import EmbeddingPreprocessor
from src.fmri2img.contrastive.queue import create_memory_queue
from src.fmri2img.losses.infonce_queue import InfoNCEQueueLoss
from src.fmri2img.losses.gaussian_nll import GaussianNLLLoss
from src.fmri2img.losses.gaussian_nce import GaussianNCELoss
from src.fmri2img.training.kl_schedule import KLScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def setup_preprocessing(config: Dict[str, Any], device: str) -> Optional[EmbeddingPreprocessor]:
    """
    Setup embedding preprocessor if enabled.
    
    Args:
        config: Experiment config
        device: Device for preprocessing
    
    Returns:
        preprocessor: EmbeddingPreprocessor instance or None
    """
    preproc_cfg = config.get("preprocessing", {})
    
    if not preproc_cfg.get("enabled", False):
        logger.info("Preprocessing disabled")
        return None
    
    artifact_path = Path(preproc_cfg.get("artifact_path", ""))
    
    if artifact_path.exists():
        # Load existing preprocessor
        logger.info(f"Loading preprocessor from {artifact_path}")
        preprocessor = EmbeddingPreprocessor.load(artifact_path)
    else:
        # Create new preprocessor (will be fit on training data)
        mode = preproc_cfg.get("mode", "center_pcr")
        k_components = preproc_cfg.get("k_components", 8)
        whiten_eps = preproc_cfg.get("whiten_eps", 1e-5)
        
        logger.info(f"Creating new preprocessor: mode={mode}, k={k_components}")
        preprocessor = EmbeddingPreprocessor(
            mode=mode,
            k_components=k_components,
            whiten_eps=whiten_eps
        )
    
    return preprocessor


def setup_losses(config: Dict[str, Any], device: str, queue=None) -> Dict[str, nn.Module]:
    """
    Setup loss functions based on config.
    
    Args:
        config: Experiment config
        device: Device for losses
        queue: Memory queue instance (optional)
    
    Returns:
        losses: Dict of loss name -> loss module
    """
    loss_cfg = config.get("loss", {})
    losses = {}
    
    # MSE loss
    if loss_cfg.get("mse", {}).get("enabled", False):
        losses["mse"] = nn.MSELoss()
        logger.info("✓ MSE loss enabled")
    
    # InfoNCE with queue
    if loss_cfg.get("infonce", {}).get("enabled", False):
        infonce_cfg = loss_cfg["infonce"]
        use_queue = infonce_cfg.get("use_queue", False) and queue is not None
        
        losses["infonce"] = InfoNCEQueueLoss(
            temperature=infonce_cfg.get("temperature", 0.07),
            learnable_temperature=infonce_cfg.get("learnable_temperature", True),
            use_queue=use_queue,
            symmetric=infonce_cfg.get("symmetric", True)
        )
        logger.info(f"✓ InfoNCE loss enabled (queue={use_queue})")
    
    # Gaussian NLL
    if loss_cfg.get("gaussian_nll", {}).get("enabled", False):
        nll_cfg = loss_cfg["gaussian_nll"]
        losses["gaussian_nll"] = GaussianNLLLoss(
            logvar_min=nll_cfg.get("logvar_min", -10.0),
            logvar_max=nll_cfg.get("logvar_max", 5.0),
            learnable_global_scale=nll_cfg.get("learnable_global_scale", False),
            reduction=nll_cfg.get("reduction", "mean")
        )
        logger.info("✓ Gaussian NLL loss enabled")
    
    # Gaussian-NCE
    if loss_cfg.get("gaussian_nce", {}).get("enabled", False):
        gnce_cfg = loss_cfg["gaussian_nce"]
        use_queue = gnce_cfg.get("use_queue", False) and queue is not None
        
        losses["gaussian_nce"] = GaussianNCELoss(
            use_temperature=gnce_cfg.get("use_temperature", True),
            temperature=gnce_cfg.get("temperature", 1.0),
            use_queue=use_queue,
            symmetric=gnce_cfg.get("symmetric", False),
            clamp_logvar=gnce_cfg.get("clamp_logvar", True),
            logvar_min=gnce_cfg.get("logvar_min", -10.0),
            logvar_max=gnce_cfg.get("logvar_max", 5.0)
        )
        logger.info(f"✓ Gaussian-NCE loss enabled (queue={use_queue})")
    
    return losses


def setup_kl_scheduler(config: Dict[str, Any]) -> Optional[KLScheduler]:
    """Setup KL annealing scheduler if enabled."""
    kl_cfg = config.get("loss", {}).get("kl", {})
    
    if not kl_cfg.get("enabled", False):
        return None
    
    scheduler = KLScheduler(
        start_weight=kl_cfg.get("weight_start", 0.0),
        end_weight=kl_cfg.get("weight_end", 0.001),
        n_steps=kl_cfg.get("anneal_steps", 10000),
        anneal_type=kl_cfg.get("anneal_type", "linear"),
        free_bits=kl_cfg.get("free_bits", 0.5),
        free_bits_aggregate=kl_cfg.get("free_bits_type", "dimension")
    )
    logger.info(f"✓ KL scheduler enabled (n_steps={kl_cfg.get('anneal_steps')})")
    return scheduler


def main():
    parser = argparse.ArgumentParser(description="Unified training script")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load config
    config = load_config(Path(args.config))
    
    # Setup device
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create output directory
    exp_name = config["experiment"]["name"]
    output_dir = Path("experimental_results") / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    logger.info(f"Saved config to {config_save_path}")
    
    # Load cached data FIRST to infer dimensions
    logger.info("Loading cached data...")
    
    # Try multiple possible embedding file paths
    possible_paths = [
        Path("cache/clip_embeddings/nsd_clipcache_multilayer.parquet"),
        Path("cache/clip_embeddings/nsd_clipvitl14.parquet"),
        Path("cache/clip_embeddings/embeddings_ViT-B-32.parquet"),
    ]
    
    embeddings_path = None
    for p in possible_paths:
        if p.exists():
            embeddings_path = p
            break
    
    if embeddings_path is None:
        raise FileNotFoundError(f"No embedding cache found. Tried: {[str(p) for p in possible_paths]}")
    
    logger.info(f"Using embeddings from: {embeddings_path}")
    import pandas as pd
    df = pd.read_parquet(embeddings_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Extract embeddings (flexible column detection)
    if 'final' in df.columns:
        # Multilayer cache format: embeddings stored as numpy arrays in 'final' column
        embeddings_list = df['final'].tolist()
        embeddings = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
        logger.info(f"Extracted embeddings from 'final' column: {embeddings.shape}")
    else:
        # Try: emb_000, emb_001, ... or embedding_0, embedding_1, ...
        embedding_cols = [c for c in df.columns if c.startswith('emb_')]
        if not embedding_cols:
            embedding_cols = [c for c in df.columns if c.startswith('embedding_')]
        if not embedding_cols:
            # Try any numeric columns as last resort
            embedding_cols = [c for c in df.columns if df[c].dtype in ['float32', 'float64']]
        
        if not embedding_cols:
            raise ValueError(f"No embedding columns found in {embeddings_path}. Columns: {df.columns.tolist()}")
        
        logger.info(f"Using {len(embedding_cols)} embedding dimensions from column format")
        embeddings = torch.tensor(df[embedding_cols].values, dtype=torch.float32)
    
    # Setup preprocessing
    preprocessor = setup_preprocessing(config, device)
    
    # Setup model
    model_config = config["model"]
    # Infer input_dim from ROI if not specified
    if model_config["encoder"]["input_dim"] is None:
        # TODO: Load from dataset or config
        model_config["encoder"]["input_dim"] = 15724  # Example for nsdgeneral
    
    # Infer output_dim from actual embeddings
    embedding_dim = embeddings.shape[1]
    if model_config["decoder"]["output_dim"] != embedding_dim:
        logger.info(f"Adjusting decoder output_dim from {model_config['decoder']['output_dim']} to {embedding_dim} (from data)")
        model_config["decoder"]["output_dim"] = embedding_dim
    
    model = create_model(model_config).to(device)
    
    # Setup memory queue
    queue = None
    if config.get("queue", {}).get("enabled", False):
        embedding_dim = model_config["decoder"]["output_dim"]
        queue = create_memory_queue(
            config=config["queue"],
            embedding_dim=embedding_dim
        )
        if queue is not None:
            queue = queue.to(device)
            logger.info(f"✓ Memory queue created (size={config['queue'].get('size', 8192)})")
    
    # Setup losses
    losses = setup_losses(config, device, queue)
    loss_weights = {k: v.get("weight", 1.0) for k, v in config.get("loss", {}).items() if isinstance(v, dict)}
    
    # Setup KL scheduler
    kl_scheduler = setup_kl_scheduler(config)
    
    # Setup optimizer
    optimizer_cfg = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg.get("lr", 1e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
        betas=optimizer_cfg.get("betas", [0.9, 0.999])
    )
    
    logger.info("=" * 80)
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Description: {config['experiment']['description']}")
    logger.info("=" * 80)
    logger.info("Starting training...")
    
    # Create dummy fMRI data (TODO: load real fMRI)
    fmri_dim = model_config["encoder"]["input_dim"]
    fmri_data = torch.randn(len(embeddings), fmri_dim)
    
    # Split data
    train_split = config["data"]["train_split"]
    val_split = config["data"]["val_split"]
    n_train = int(len(embeddings) * train_split)
    n_val = int(len(embeddings) * val_split)
    
    train_fmri, train_emb = fmri_data[:n_train], embeddings[:n_train]
    val_fmri, val_emb = fmri_data[n_train:n_train+n_val], embeddings[n_train:n_train+n_val]
    
    # Create dataloaders
    from torch.utils.data import TensorDataset
    train_dataset = TensorDataset(train_fmri, train_emb)
    val_dataset = TensorDataset(val_fmri, val_emb)
    
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    logger.info("Starting training...")
    
    # Training loop
    num_epochs = config["training"]["num_epochs"]
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, losses, loss_weights,
            device, kl_scheduler, queue, preprocessor, global_step
        )
        logger.info(f"Train: {' | '.join([f'{k}={v:.4f}' for k, v in train_metrics.items()])}")
        
        # Validate
        val_metrics = validate(model, val_loader, losses, loss_weights, device, preprocessor, queue)
        logger.info(f"Val:   {' | '.join([f'{k}={v:.4f}' for k, v in val_metrics.items()])}")
        
        # Save checkpoint
        val_loss = val_metrics.get("total_loss", val_metrics.get("mse", float('inf')))
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "checkpoint.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            logger.info(f"✓ Saved best checkpoint: val_loss={val_loss:.4f}")
    
    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info("=" * 80)


def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute KL(q(z|x) || p(z)) for Gaussian posterior."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    losses: Dict[str, nn.Module],
    loss_weights: Dict[str, float],
    device: torch.device,
    kl_scheduler: Optional[KLScheduler],
    queue: Optional[nn.Module],
    preprocessor: Optional[EmbeddingPreprocessor],
    global_step: int
) -> tuple:
    """Train for one epoch."""
    model.train()
    epoch_metrics = {}
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        fmri, gt_embedding = batch
        fmri = fmri.to(device)
        gt_embedding = gt_embedding.to(device)
        
        # Apply preprocessing
        if preprocessor is not None:
            gt_embedding_np = gt_embedding.cpu().numpy()
            gt_embedding_proc = preprocessor.transform(gt_embedding_np)
            gt_embedding = torch.from_numpy(gt_embedding_proc).to(device)
        
        # Forward
        output = model(fmri)
        if isinstance(output, tuple):
            pred, logvar = output
        else:
            pred, logvar = output, None
        
        # Compute losses
        total_loss = 0.0
        batch_metrics = {}
        
        # MSE
        if "mse" in losses and logvar is None:
            mse_loss = losses["mse"](pred, gt_embedding)
            total_loss += loss_weights.get("mse", 1.0) * mse_loss
            batch_metrics["mse"] = mse_loss.item()
        
        # InfoNCE
        if "infonce" in losses and logvar is None:
            infonce_loss = losses["infonce"](pred, gt_embedding, queue=queue)
            total_loss += loss_weights.get("infonce", 1.0) * infonce_loss
            batch_metrics["infonce"] = infonce_loss.item()
            if queue is not None:
                queue.enqueue(gt_embedding)
        
        # Gaussian NLL
        if "gaussian_nll" in losses and logvar is not None:
            nll_loss = losses["gaussian_nll"](pred, logvar, gt_embedding)
            total_loss += loss_weights.get("gaussian_nll", 1.0) * nll_loss
            batch_metrics["nll"] = nll_loss.item()
        
        # Gaussian-NCE
        if "gaussian_nce" in losses and logvar is not None:
            gnce_loss = losses["gaussian_nce"](pred, logvar, gt_embedding, queue=queue)
            total_loss += loss_weights.get("gaussian_nce", 1.0) * gnce_loss
            batch_metrics["gnce"] = gnce_loss.item()
            if queue is not None:
                queue.enqueue(gt_embedding)
        
        # KL
        if kl_scheduler is not None and logvar is not None:
            kl_raw = compute_kl_divergence(pred, logvar)
            kl_weight = kl_scheduler.step()
            kl_loss = kl_weight * kl_raw
            total_loss += kl_loss
            batch_metrics["kl"] = kl_loss.item()
        
        batch_metrics["loss"] = total_loss.item()
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        pbar.set_postfix({k: f"{v:.4f}" for k, v in batch_metrics.items()})
        
        for k, v in batch_metrics.items():
            epoch_metrics.setdefault(k, []).append(v)
        
        global_step += 1
    
    return {k: np.mean(v) for k, v in epoch_metrics.items()}, global_step


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    losses: Dict[str, nn.Module],
    loss_weights: Dict[str, float],
    device: torch.device,
    preprocessor: Optional[EmbeddingPreprocessor],
    queue: Optional[nn.Module]
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    epoch_metrics = {}
    
    with torch.no_grad():
        for batch in dataloader:
            fmri, gt_embedding = batch
            fmri = fmri.to(device)
            gt_embedding = gt_embedding.to(device)
            
            if preprocessor is not None:
                gt_embedding_np = gt_embedding.cpu().numpy()
                gt_embedding_proc = preprocessor.transform(gt_embedding_np)
                gt_embedding = torch.from_numpy(gt_embedding_proc).to(device)
            
            output = model(fmri)
            if isinstance(output, tuple):
                pred, logvar = output
            else:
                pred, logvar = output, None
            
            total_loss = 0.0
            batch_metrics = {}
            
            if "mse" in losses and logvar is None:
                mse_loss = losses["mse"](pred, gt_embedding)
                total_loss += loss_weights.get("mse", 1.0) * mse_loss
                batch_metrics["mse"] = mse_loss.item()
            
            if "infonce" in losses and logvar is None:
                infonce_loss = losses["infonce"](pred, gt_embedding, queue=None)
                total_loss += loss_weights.get("infonce", 1.0) * infonce_loss
                batch_metrics["infonce"] = infonce_loss.item()
            
            if "gaussian_nll" in losses and logvar is not None:
                nll_loss = losses["gaussian_nll"](pred, logvar, gt_embedding)
                total_loss += loss_weights.get("gaussian_nll", 1.0) * nll_loss
                batch_metrics["nll"] = nll_loss.item()
            
            if "gaussian_nce" in losses and logvar is not None:
                gnce_loss = losses["gaussian_nce"](pred, logvar, gt_embedding, queue=None)
                total_loss += loss_weights.get("gaussian_nce", 1.0) * gnce_loss
                batch_metrics["gnce"] = gnce_loss.item()
            
            if logvar is not None:
                kl_raw = compute_kl_divergence(pred, logvar)
                batch_metrics["kl"] = kl_raw.item()
            
            batch_metrics["loss"] = total_loss.item()
            
            for k, v in batch_metrics.items():
                epoch_metrics.setdefault(k, []).append(v)
    
    return {k: np.mean(v) for k, v in epoch_metrics.items()}


if __name__ == "__main__":
    main()
