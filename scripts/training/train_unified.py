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
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from tqdm import tqdm
import pandas as pd
import nibabel as nib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fmri2img.models.unified_model import create_model
from src.fmri2img.embedding_preproc import EmbeddingPreprocessor
from src.fmri2img.contrastive.queue import create_memory_queue
from src.fmri2img.losses.infonce_queue import InfoNCEQueueLoss
from src.fmri2img.losses.gaussian_nll import GaussianNLLLoss
from src.fmri2img.losses.gaussian_nce import GaussianNCELoss
from src.fmri2img.losses.vmf_nce import VonMisesFisherNCELoss, VonMisesFisherNLLLoss, kappa_regularizer
from src.fmri2img.training.kl_schedule import KLScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class NSDDataset(Dataset):
    """Dataset for NSD fMRI and CLIP embeddings."""
    
    def __init__(self, index_df: pd.DataFrame, embeddings_df: pd.DataFrame, 
                 roi_mask_path: Optional[Path] = None):
        """
        Args:
            index_df: DataFrame with columns [nsdId, beta_path, beta_index, ...]
            embeddings_df: DataFrame with nsdId and embedding columns
            roi_mask_path: Optional path to ROI mask for voxel selection
        """
        self.index_df = index_df.reset_index(drop=True)
        self.embeddings_df = embeddings_df
        self.roi_mask = None
        self.beta_cache = {}  # Cache loaded beta files
        
        # Initialize S3 filesystem for MinIO
        import s3fs
        self.s3_fs = s3fs.S3FileSystem(
            key='EMyfxXhCOfYyXvU3O5wn',
            secret='PQpdz1TcUqj8b12kZHb0Sa0UtXMEnuZ4QLwbJmo8',
            client_kwargs={'endpoint_url': 'http://s3hub-intern.cs.ubbcluj.ro:9000'}
        )
        
        # Build nsdId to embedding lookup
        if 'nsdId' in embeddings_df.columns:
            self.embedding_lookup = {
                row['nsdId']: idx 
                for idx, row in embeddings_df.iterrows()
            }
        else:
            # Fallback: assume index matches nsdId
            self.embedding_lookup = {i: i for i in range(len(embeddings_df))}
        
        # Load ROI mask if provided
        if roi_mask_path and roi_mask_path.exists():
            mask_img = nib.load(roi_mask_path)
            mask_data = mask_img.get_fdata()
            # Threshold mask (NSD ROIs are often probabilistic, threshold at 0.5)
            self.roi_mask = mask_data > 0.5
            logger.info(f"Loaded ROI mask: {self.roi_mask.sum():.0f} voxels")
        
        logger.info(f"NSDDataset created: {len(self.index_df)} trials")
    
    def __len__(self):
        return len(self.index_df)
    
    def __getitem__(self, idx):
        row = self.index_df.iloc[idx]
        
        # Load fMRI data
        beta_path = row['beta_path']
        beta_idx = row['beta_index']
        
        # Cache beta file to avoid repeated loading
        if beta_path not in self.beta_cache:
            if beta_path.startswith("s3://"):
                # Load from S3 via temp file in RAM disk
                import os
                temp_path = f"/dev/shm/temp_beta_{os.getpid()}_{hash(beta_path)}.nii.gz"
                try:
                    with self.s3_fs.open(beta_path, "rb") as f_in:
                        with open(temp_path, "wb") as f_out:
                            f_out.write(f_in.read())
                    img = nib.load(temp_path)
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            else:
                # Load local file
                img = nib.load(beta_path)
            self.beta_cache[beta_path] = img.get_fdata()
        
        beta_vol = self.beta_cache[beta_path][..., beta_idx]
        
        # Apply ROI mask if available
        if self.roi_mask is not None:
            fmri = beta_vol[self.roi_mask].flatten()
        else:
            fmri = beta_vol.flatten()
        
        # Get embedding
        nsdId = row['nsdId']
        emb_idx = self.embedding_lookup.get(nsdId, nsdId % len(self.embeddings_df))
        
        # Extract embedding from dataframe
        if 'final' in self.embeddings_df.columns:
            embedding = self.embeddings_df.iloc[emb_idx]['final']
        elif 'embedding' in self.embeddings_df.columns:
            embedding = self.embeddings_df.iloc[emb_idx]['embedding']
        else:
            # Multi-column format
            emb_cols = [c for c in self.embeddings_df.columns if c.startswith('emb_')]
            embedding = self.embeddings_df.iloc[emb_idx][emb_cols].values
        
        # Convert to tensors
        fmri_tensor = torch.tensor(fmri, dtype=torch.float32)
        emb_tensor = torch.tensor(np.array(embedding), dtype=torch.float32)
        
        return fmri_tensor, emb_tensor


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
            logvar_max=gnce_cfg.get("logvar_max", 5.0),
        )
        logger.info(f"Gaussian-NCE loss enabled (queue={use_queue})")

    # Detect whether the vMF decoder outputs log_kappa (legacy) or direct kappa
    model_cfg = config.get("model", {})
    decoder_cfg = model_cfg.get("decoder", {})
    vmf_kappa_is_log = "log_kappa_min" in decoder_cfg or "log_kappa_max" in decoder_cfg
    if model_cfg.get("posterior") == "vmf" or "kappa_min" in decoder_cfg:
        vmf_kappa_is_log = False

    # vMF NLL
    if loss_cfg.get("vmf_nll", {}).get("enabled", False):
        vmf_nll_cfg = loss_cfg["vmf_nll"]
        losses["vmf_nll"] = VonMisesFisherNLLLoss(
            dim=vmf_nll_cfg.get("dim", 768),
            kappa_is_log=vmf_kappa_is_log,
        )
        logger.info(f"vMF-NLL loss enabled (kappa_is_log={vmf_kappa_is_log})")

    # vMF-NCE
    if loss_cfg.get("vmf_nce", {}).get("enabled", False):
        vmf_nce_cfg = loss_cfg["vmf_nce"]
        use_queue = vmf_nce_cfg.get("use_queue", False) and queue is not None
        losses["vmf_nce"] = VonMisesFisherNCELoss(
            tau=vmf_nce_cfg.get("tau", 0.07),
            use_queue=use_queue,
            kappa_is_log=vmf_kappa_is_log,
        )
        logger.info(f"vMF-NCE loss enabled (queue={use_queue}, tau={vmf_nce_cfg.get('tau', 0.07)}, kappa_is_log={vmf_kappa_is_log})")

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
    parser.add_argument("--subject", type=str, default=None, help="Override subject (e.g. subj02)")
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
        Path("cache/clip_embeddings/text_clip.parquet"),
    ]
    
    embeddings_path = None
    for p in possible_paths:
        if p.exists():
            embeddings_path = p
            break
    
    if embeddings_path is None:
        logger.warning(f"No embedding cache found in: {[str(p) for p in possible_paths]}")
        logger.warning("Generating dummy embeddings for testing...")
        # Generate dummy embeddings (512-dim CLIP embeddings, 1000 samples)
        n_samples = 1000
        embedding_dim = 512
        embeddings = torch.randn(n_samples, embedding_dim)
        # Normalize like CLIP embeddings
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        logger.info(f"Generated dummy embeddings: {embeddings.shape}")
    else:
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
        elif 'embedding' in df.columns:
            # Single embedding column format: embeddings stored as lists/arrays
            embeddings_list = df['embedding'].tolist()
            embeddings = torch.tensor(np.array(embeddings_list), dtype=torch.float32)
            logger.info(f"Extracted embeddings from 'embedding' column: {embeddings.shape}")
        else:
            # Try: emb_000, emb_001, ... or embedding_0, embedding_1, ...
            embedding_cols = [c for c in df.columns if c.startswith('emb_')]
            if not embedding_cols:
                embedding_cols = [c for c in df.columns if c.startswith('embedding_')]
            if not embedding_cols:
                # Try any numeric columns as last resort
                embedding_cols = [c for c in df.columns if df[c].dtype in ['float32', 'float64']]
            
            if not embedding_cols:
                logger.warning(f"No embedding columns found in {embeddings_path}. Columns: {df.columns.tolist()}")
                logger.warning("Generating dummy embeddings for testing...")
                n_samples = len(df)
                embedding_dim = 512
                embeddings = torch.randn(n_samples, embedding_dim)
                embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                logger.info(f"Generated dummy embeddings: {embeddings.shape}")
            else:
                logger.info(f"Using {len(embedding_cols)} embedding dimensions from column format")
                embeddings = torch.tensor(df[embedding_cols].values, dtype=torch.float32)
    
    # Setup preprocessing
    preprocessor = setup_preprocessing(config, device)
    
    logger.info("=" * 80)
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Description: {config['experiment']['description']}")
    logger.info("=" * 80)
    
    # Load NSD index
    index_path = Path("data/indices/nsd_index/subject=subj01/index_full.parquet")
    if not index_path.exists():
        logger.error(f"Index not found: {index_path}")
        logger.error("Run: python3 scripts/build_full_subj01_index.py")
        sys.exit(1)
    
    index_df = pd.read_parquet(index_path)
    logger.info(f"Loaded index: {len(index_df)} trials, {index_df['nsdId'].nunique()} unique stimuli")
    
    # Create dataset (embeddings already loaded as DataFrame)
    embeddings_df = pd.read_parquet(embeddings_path)
    
    # Add nsdId if not present (assume sequential)
    if 'nsdId' not in embeddings_df.columns:
        embeddings_df['nsdId'] = range(len(embeddings_df))
        logger.warning("Added sequential nsdId column to embeddings")
    
    # Create full dataset
    roi_mask_path = Path("data/nsd/ppdata/subj01/func1pt8mm/roi_nsdgeneral.nii.gz")
    if roi_mask_path.exists():
        logger.info(f"Using ROI mask: {roi_mask_path}")
        full_dataset = NSDDataset(index_df, embeddings_df, roi_mask_path=roi_mask_path)
    else:
        logger.warning(f"ROI mask not found: {roi_mask_path}")
        logger.warning("Using full brain volume (may cause OOM). Run: bash scripts/download_nsd_roi.sh")
        full_dataset = NSDDataset(index_df, embeddings_df)
    
    # Get fMRI dimension from first sample
    sample_fmri, sample_emb = full_dataset[0]
    fmri_dim = sample_fmri.shape[0]
    embedding_dim = sample_emb.shape[0]
    logger.info(f"Data dimensions: fMRI={fmri_dim}, Embedding={embedding_dim}")
    
    # NOW setup model with correct dimensions
    model_config = config["model"]
    model_config["encoder"]["input_dim"] = fmri_dim
    model_config["decoder"]["output_dim"] = embedding_dim
    logger.info(f"Creating model: encoder {fmri_dim} → decoder {embedding_dim}")
    
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
    
    # Split data
    train_split = config["data"]["train_split"]
    val_split = config["data"]["val_split"]
    n_total = len(full_dataset)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    # Create train/val indices
    indices = torch.randperm(n_total).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # Create dataloaders
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    logger.info("Starting training...")
    
    # Gradient accumulation and mixed-precision
    grad_accum_steps = config["training"].get("gradient_accumulation_steps", 16)
    use_amp = config["training"].get("mixed_precision", False) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger.info(
        f"Gradient accumulation: {grad_accum_steps} steps "
        f"(effective batch {batch_size * grad_accum_steps}) | AMP: {use_amp}"
    )

    # Cosine LR scheduler
    num_epochs = config["training"]["num_epochs"]
    total_steps = num_epochs * len(train_loader) // grad_accum_steps
    warmup_steps = config["training"].get("warmup_epochs", 5) * len(train_loader) // grad_accum_steps
    min_lr = float(config["training"].get("min_lr", 1e-6))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps and warmup_steps > 0:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(min_lr / float(optimizer_cfg.get("lr", 1e-4)),
                    0.5 * (1.0 + math.cos(math.pi * progress)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = config["training"].get("early_stop_patience", 15)
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs} | lr={optimizer.param_groups[0]['lr']:.2e}")

        _vmf_is_log = getattr(model, "vmf_output_is_log", True)
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, losses, loss_weights,
            device, kl_scheduler, queue, preprocessor, global_step,
            grad_accum_steps=grad_accum_steps, scaler=scaler,
            lr_scheduler=lr_scheduler,
            config_ref=config,
            vmf_is_log=_vmf_is_log,
        )
        logger.info(f"Train: {' | '.join([f'{k}={v:.4f}' for k, v in train_metrics.items()])}")

        # Kappa health warnings
        if "kappa_std" in train_metrics:
            if train_metrics["kappa_std"] < 0.01:
                logger.warning("kappa has collapsed (std < 0.01) — model may not be learning uncertainty")
            kappa_upper = getattr(model.decoder, "kappa_max", None)
            kq90 = train_metrics.get("kappa_q90")
            if kappa_upper is not None and kq90 is not None and kq90 > 0.99 * kappa_upper:
                logger.warning(f"kappa saturating at upper bound ({kq90:.1f} / {kappa_upper:.1f})")

        val_metrics = validate(model, val_loader, losses, loss_weights, device, preprocessor, queue)
        logger.info(f"Val:   {' | '.join([f'{k}={v:.4f}' for k, v in val_metrics.items()])}")

        val_loss = val_metrics.get("loss", val_metrics.get("mse", float('inf')))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = output_dir / "checkpoint.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'model_config': config.get("model", {}),
            }, checkpoint_path)
            logger.info(f"Saved best checkpoint: val_loss={val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={early_stop_patience})")
                break

    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info("=" * 80)


import math as math


def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute KL(q(z|x) || p(z)) for Gaussian posterior."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def compute_vmf_kl(mu: torch.Tensor, log_kappa: torch.Tensor, dim: int) -> torch.Tensor:
    """
    KL(vMF(mu, kappa) || Uniform(S^{d-1})).

    Uses the Amos-type approximation for the ratio I_{d/2}(k)/I_{d/2-1}(k).
    """
    kappa = log_kappa.exp().squeeze(-1)  # (B,)
    half_d = dim / 2.0
    log_sphere = half_d * math.log(2 * math.pi) + torch.lgamma(torch.tensor(half_d, device=mu.device)) - torch.tensor(half_d, device=mu.device) * math.log(1.0)
    log_c_kappa = (half_d - 1) * torch.log(kappa + 1e-8) - half_d * math.log(2 * math.pi) - torch.log(
        _ive(half_d - 1, kappa) + 1e-10
    ) - kappa
    kl = kappa * _ive(half_d, kappa) / (_ive(half_d - 1, kappa) + 1e-10) - log_c_kappa + log_sphere
    return kl.mean()


def _ive(v: float, z: torch.Tensor) -> torch.Tensor:
    """Exponentially-scaled modified Bessel I_v(z) * exp(-z)."""
    return torch.special.i1e(z) if abs(v - 1.0) < 0.01 else torch.special.i0e(z)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    losses: Dict[str, nn.Module],
    loss_weights: Dict[str, float],
    device: str,
    kl_scheduler: Optional[KLScheduler],
    queue: Optional[nn.Module],
    preprocessor: Optional[EmbeddingPreprocessor],
    global_step: int,
    grad_accum_steps: int = 1,
    scaler: Optional[torch.amp.GradScaler] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    config_ref: Optional[Dict[str, Any]] = None,
    vmf_is_log: bool = True,
) -> tuple:
    """Train for one epoch with gradient accumulation and optional AMP."""
    model.train()
    epoch_metrics: Dict[str, list] = {}
    use_amp = scaler is not None and scaler.is_enabled()
    model_type = getattr(model, "model_type", "deterministic")

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc="Training")
    for step_in_epoch, batch in enumerate(pbar):
        fmri, gt_embedding = batch
        fmri = fmri.to(device)
        gt_embedding = gt_embedding.to(device)

        if preprocessor is not None:
            gt_embedding_np = gt_embedding.cpu().numpy()
            gt_embedding_proc = preprocessor.transform(gt_embedding_np)
            gt_embedding = torch.from_numpy(gt_embedding_proc).to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(fmri)
            if isinstance(output, tuple):
                pred, aux = output
            else:
                pred, aux = output, None

            total_loss = torch.tensor(0.0, device=device)
            batch_metrics: Dict[str, float] = {}

            is_gaussian = model_type == "gaussian" and aux is not None
            is_vmf = model_type == "vmf" and aux is not None

            # Deterministic losses
            if "mse" in losses and not is_gaussian and not is_vmf:
                mse_loss = losses["mse"](pred, gt_embedding)
                total_loss = total_loss + loss_weights.get("mse", 1.0) * mse_loss
                batch_metrics["mse"] = mse_loss.item()

            if "infonce" in losses and not is_gaussian and not is_vmf:
                infonce_loss = losses["infonce"](pred, gt_embedding, queue=queue)
                total_loss = total_loss + loss_weights.get("infonce", 1.0) * infonce_loss
                batch_metrics["infonce"] = infonce_loss.item()
                if queue is not None:
                    queue.enqueue(gt_embedding)

            # Gaussian losses
            if "gaussian_nll" in losses and is_gaussian:
                nll_loss = losses["gaussian_nll"](pred, aux, gt_embedding)
                total_loss = total_loss + loss_weights.get("gaussian_nll", 1.0) * nll_loss
                batch_metrics["nll"] = nll_loss.item()

            if "gaussian_nce" in losses and is_gaussian:
                gnce_loss = losses["gaussian_nce"](pred, aux, gt_embedding, queue=queue)
                total_loss = total_loss + loss_weights.get("gaussian_nce", 1.0) * gnce_loss
                batch_metrics["gnce"] = gnce_loss.item()
                if queue is not None:
                    queue.enqueue(gt_embedding)

            # vMF losses
            if "vmf_nll" in losses and is_vmf:
                vmf_nll_loss = losses["vmf_nll"](pred, aux, gt_embedding)
                total_loss = total_loss + loss_weights.get("vmf_nll", 1.0) * vmf_nll_loss
                batch_metrics["vmf_nll"] = vmf_nll_loss.item()

            if "vmf_nce" in losses and is_vmf:
                vmf_nce_loss = losses["vmf_nce"](pred, aux, gt_embedding, queue=queue)
                total_loss = total_loss + loss_weights.get("vmf_nce", 1.0) * vmf_nce_loss
                batch_metrics["vmf_nce"] = vmf_nce_loss.item()
                if queue is not None:
                    queue.enqueue(gt_embedding)

            # Kappa regularizer (optional, config-driven)
            kappa_reg_cfg = config_ref.get("loss", {}).get("kappa_reg", {}) if config_ref else {}
            if kappa_reg_cfg.get("enabled", False) and is_vmf and aux is not None:
                kappa_vals = aux.squeeze(-1) if not vmf_is_log else aux.exp().squeeze(-1)
                kr = kappa_regularizer(kappa_vals, kappa_reg_cfg.get("lambda_kappa", 0.01))
                total_loss = total_loss + kr
                batch_metrics["kappa_reg"] = kr.item()

            # Kappa statistics logging
            if is_vmf and aux is not None:
                with torch.no_grad():
                    kappa_vals = aux.squeeze(-1) if not vmf_is_log else aux.exp().squeeze(-1)
                    batch_metrics["kappa_mean"] = kappa_vals.mean().item()
                    batch_metrics["kappa_std"] = kappa_vals.std().item()
                    batch_metrics["kappa_min"] = kappa_vals.min().item()
                    batch_metrics["kappa_max"] = kappa_vals.max().item()
                    if kappa_vals.numel() >= 2:
                        q = torch.quantile(
                            kappa_vals.float(),
                            torch.tensor([0.1, 0.5, 0.9], device=kappa_vals.device),
                        )
                        batch_metrics["kappa_q10"] = q[0].item()
                        batch_metrics["kappa_q50"] = q[1].item()
                        batch_metrics["kappa_q90"] = q[2].item()

            # KL divergence
            if kl_scheduler is not None and is_gaussian:
                kl_raw = compute_kl_divergence(pred, aux)
                kl_weight = kl_scheduler.step()
                total_loss = total_loss + kl_weight * kl_raw
                batch_metrics["kl"] = (kl_weight * kl_raw).item()

            if kl_scheduler is not None and is_vmf:
                kl_raw = compute_vmf_kl(pred, aux, pred.size(-1))
                kl_weight = kl_scheduler.step()
                total_loss = total_loss + kl_weight * kl_raw
                batch_metrics["kl"] = (kl_weight * kl_raw).item()

            # Scale for accumulation
            total_loss = total_loss / grad_accum_steps

        batch_metrics["loss"] = total_loss.item() * grad_accum_steps

        scaler.scale(total_loss).backward() if scaler is not None else total_loss.backward()

        if (step_in_epoch + 1) % grad_accum_steps == 0 or (step_in_epoch + 1) == len(dataloader):
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

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
    device: str,
    preprocessor: Optional[EmbeddingPreprocessor],
    queue: Optional[nn.Module],
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    epoch_metrics: Dict[str, list] = {}
    model_type = getattr(model, "model_type", "deterministic")

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
                pred, aux = output
            else:
                pred, aux = output, None

            total_loss = torch.tensor(0.0, device=device)
            batch_metrics: Dict[str, float] = {}
            is_gaussian = model_type == "gaussian" and aux is not None
            is_vmf = model_type == "vmf" and aux is not None

            if "mse" in losses and not is_gaussian and not is_vmf:
                mse_loss = losses["mse"](pred, gt_embedding)
                total_loss = total_loss + loss_weights.get("mse", 1.0) * mse_loss
                batch_metrics["mse"] = mse_loss.item()

            if "infonce" in losses and not is_gaussian and not is_vmf:
                infonce_loss = losses["infonce"](pred, gt_embedding, queue=None)
                total_loss = total_loss + loss_weights.get("infonce", 1.0) * infonce_loss
                batch_metrics["infonce"] = infonce_loss.item()

            if "gaussian_nll" in losses and is_gaussian:
                nll_loss = losses["gaussian_nll"](pred, aux, gt_embedding)
                total_loss = total_loss + loss_weights.get("gaussian_nll", 1.0) * nll_loss
                batch_metrics["nll"] = nll_loss.item()

            if "gaussian_nce" in losses and is_gaussian:
                gnce_loss = losses["gaussian_nce"](pred, aux, gt_embedding, queue=None)
                total_loss = total_loss + loss_weights.get("gaussian_nce", 1.0) * gnce_loss
                batch_metrics["gnce"] = gnce_loss.item()

            if "vmf_nll" in losses and is_vmf:
                vmf_nll_loss = losses["vmf_nll"](pred, aux, gt_embedding)
                total_loss = total_loss + loss_weights.get("vmf_nll", 1.0) * vmf_nll_loss
                batch_metrics["vmf_nll"] = vmf_nll_loss.item()

            if "vmf_nce" in losses and is_vmf:
                vmf_nce_loss = losses["vmf_nce"](pred, aux, gt_embedding, queue=None)
                total_loss = total_loss + loss_weights.get("vmf_nce", 1.0) * vmf_nce_loss
                batch_metrics["vmf_nce"] = vmf_nce_loss.item()

            if is_gaussian:
                batch_metrics["kl"] = compute_kl_divergence(pred, aux).item()
            if is_vmf:
                batch_metrics["kl"] = compute_vmf_kl(pred, aux, pred.size(-1)).item()

            batch_metrics["loss"] = total_loss.item()
            for k, v in batch_metrics.items():
                epoch_metrics.setdefault(k, []).append(v)

    return {k: np.mean(v) for k, v in epoch_metrics.items()}


if __name__ == "__main__":
    main()
