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
            queue=queue if use_queue else None,
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
        weight_start=kl_cfg.get("weight_start", 0.0),
        weight_end=kl_cfg.get("weight_end", 0.001),
        anneal_steps=kl_cfg.get("anneal_steps", 10000),
        anneal_type=kl_cfg.get("anneal_type", "linear"),
        free_bits=kl_cfg.get("free_bits", 0.5),
        free_bits_type=kl_cfg.get("free_bits_type", "per_dim")
    )
    logger.info(f"✓ KL scheduler enabled (anneal_steps={kl_cfg.get('anneal_steps')})")
    return scheduler


def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence: KL(q(z|x) || N(0,I))
    
    Args:
        mu: Mean (B, D)
        logvar: Log-variance (B, D)
    
    Returns:
        kl_loss: Scalar KL divergence
    """
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kl_div.mean()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    losses: Dict[str, nn.Module],
    loss_weights: Dict[str, float],
    device: str,
    queue=None,
    kl_scheduler: Optional[KLScheduler] = None,
    preprocessor: Optional[EmbeddingPreprocessor] = None,
    global_step: int = 0
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        metrics: Dict of metric name -> average value
    """
    model.train()
    epoch_metrics = {}
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        fmri = batch["fmri"].to(device)
        gt_embedding = batch["embedding"].to(device)
        
        # Apply preprocessing if enabled
        if preprocessor is not None:
            gt_embedding_np = gt_embedding.cpu().numpy()
            gt_embedding_proc = preprocessor.transform(gt_embedding_np)
            gt_embedding = torch.from_numpy(gt_embedding_proc).to(device)
        
        # Forward pass
        pred, logvar = model(fmri)
        
        # Compute losses
        total_loss = 0.0
        batch_metrics = {}
        
        # MSE loss (deterministic models)
        if "mse" in losses and logvar is None:
            mse_loss = losses["mse"](pred, gt_embedding)
            total_loss += loss_weights.get("mse", 1.0) * mse_loss
            batch_metrics["mse"] = mse_loss.item()
        
        # InfoNCE loss (deterministic models)
        if "infonce" in losses and logvar is None:
            infonce_loss = losses["infonce"](pred, gt_embedding, queue=queue)
            total_loss += loss_weights.get("infonce", 1.0) * infonce_loss
            batch_metrics["infonce"] = infonce_loss.item()
            
            # Update queue
            if queue is not None:
                queue.enqueue(gt_embedding)
        
        # Gaussian NLL loss (probabilistic models)
        if "gaussian_nll" in losses and logvar is not None:
            nll_loss = losses["gaussian_nll"](pred, logvar, gt_embedding)
            total_loss += loss_weights.get("gaussian_nll", 1.0) * nll_loss
            batch_metrics["gaussian_nll"] = nll_loss.item()
        
        # Gaussian-NCE loss (probabilistic models)
        if "gaussian_nce" in losses and logvar is not None:
            gnce_loss = losses["gaussian_nce"](pred, logvar, gt_embedding, queue=queue)
            total_loss += loss_weights.get("gaussian_nce", 1.0) * gnce_loss
            batch_metrics["gaussian_nce"] = gnce_loss.item()
            
            # Update queue
            if queue is not None:
                queue.enqueue(gt_embedding)
        
        # KL loss (probabilistic models with annealing)
        if kl_scheduler is not None and logvar is not None:
            kl_raw = compute_kl_divergence(pred, logvar)
            kl_loss = kl_scheduler.apply_loss(kl_raw, pred, logvar, global_step)
            total_loss += kl_loss
            batch_metrics["kl"] = kl_loss.item()
            batch_metrics["kl_raw"] = kl_raw.item()
            batch_metrics["kl_weight"] = kl_scheduler.get_weight(global_step)
        
        batch_metrics["total_loss"] = total_loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update progress bar
        pbar.set_postfix(batch_metrics)
        
        # Accumulate metrics
        for k, v in batch_metrics.items():
            if k not in epoch_metrics:
                epoch_metrics[k] = []
            epoch_metrics[k].append(v)
        
        global_step += 1
    
    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
    return avg_metrics, global_step


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
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    logger.info(f"Saved config to {config_save_path}")
    
    # Setup preprocessing
    preprocessor = setup_preprocessing(config, device)
    
    # Setup model
    model_config = config["model"]
    # Infer input_dim from ROI if not specified
    if model_config["encoder"]["input_dim"] is None:
        # TODO: Load from dataset or config
        model_config["encoder"]["input_dim"] = 15724  # Example for nsdgeneral
    
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
        lr=optimizer_cfg.get("lr", 1e-4),
        weight_decay=optimizer_cfg.get("weight_decay", 0.01),
        betas=optimizer_cfg.get("betas", [0.9, 0.999])
    )
    
    logger.info("=" * 80)
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Description: {config['experiment']['description']}")
    logger.info("=" * 80)
    logger.info("Setup complete. Ready to train!")
    logger.info("")
    logger.info("TODO: Integrate with actual dataset and training loop")
    logger.info("Next steps:")
    logger.info("  1. Create dataset loader (NSDDataset)")
    logger.info("  2. Implement validation loop")
    logger.info("  3. Add checkpointing")
    logger.info("  4. Add tensorboard logging")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
