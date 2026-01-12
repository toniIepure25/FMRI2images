#!/usr/bin/env python3
"""
Training Script for Probabilistic Multi-Layer Two-Stage Encoder (Novel Approach)
=================================================================================

This script trains the NOVEL probabilistic model that predicts uncertainty-aware
CLIP embeddings from fMRI data using variational inference.

**Novel Contributions:**
1. Probabilistic outputs: Predicts μ and σ² instead of point estimates
2. Uncertainty quantification: KL divergence regularization
3. Multiple samples at inference: Better reconstruction quality
4. Variational inference: Principled Bayesian approach

Architecture:
    Stage 1: fMRI (voxels) → Latent h (Residual MLP)
    Stage 2: h → {μ, logσ²} → CLIP embeddings (512-D) for multiple layers
    
Training:
    - Loss = Reconstruction Loss + β·KL(q(z|x) || N(0,I))
    - Reparameterization trick: z = μ + ε·σ, ε ~ N(0,1)
    - KL weight β annealed during training

References:
    - Kingma & Welling (2014): "Auto-Encoding Variational Bayes"
    - MindEye approach: Deterministic embeddings
    - Our approach: Probabilistic embeddings with uncertainty
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fmri2img.models.encoders import ProbabilisticMultiLayerTwoStageEncoder
from src.fmri2img.data.torch_dataset import NSDIterableDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def create_dataloader(config):
    """Create DataLoader with REAL NSD data."""
    log.info("Creating DataLoader with REAL NSD data...")
    
    index_path = Path(config['data']['index_path'])
    
    if not index_path.exists():
        raise FileNotFoundError(
            f"❌ Index file not found: {index_path}\n\n"
            f"Please build the index first:\n"
            f"  python build_minimal_index.py --subject {config['data']['subject']} --session {config['data']['session']}"
        )
    
    log.info(f"  Index path: {index_path}")
    log.info(f"  Subject: {config['data']['subject']}")
    log.info(f"  Session: {config['data']['session']}")
    
    dataset = NSDIterableDataset(
        index_path_or_root=str(index_path),
        subject=config['data']['subject'],
        session=config['data']['session'],
        shuffle=True,
        limit=config['data'].get('limit', None)
    )
    
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], num_workers=0)
    
    log.info(f"✓ DataLoader created successfully")
    return dataloader


def train_epoch(model, dataloader, optimizer, device, epoch, kl_weight):
    """Train for one epoch with probabilistic loss."""
    model.train()
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
    
    for batch in pbar:
        fmri = batch['fmri'].to(device)
        
        # Model expects (batch, n_voxels) - flatten spatial dimensions
        if fmri.ndim == 4:  # (B, H, W, D)
            batch_size = fmri.shape[0]
            fmri = fmri.reshape(batch_size, -1)
        
        # TODO: Load actual CLIP embeddings from cache instead of random targets
        # For now using random targets for smoke testing the training loop
        batch_size = fmri.shape[0]
        target_clip = torch.randn(batch_size, 512, device=device)
        target_clip = target_clip / target_clip.norm(dim=1, keepdim=True)  # L2 normalize
        
        optimizer.zero_grad()
        
        # Forward pass: sample from distribution
        outputs, kl_loss = model(fmri, sample=True, return_kl=True)
        
        # Reconstruction loss: MSE between predicted and target CLIP
        # Use 'final' layer output (main prediction)
        pred_clip = outputs['final']
        recon_loss = nn.functional.mse_loss(pred_clip, target_clip)
        
        # Total loss: reconstruction + KL divergence
        loss = recon_loss + kl_weight * kl_loss
        
        loss.backward()
        optimizer.step()
        
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}',
            'total': f'{loss.item():.4f}',
            'avg_recon': f'{total_recon_loss/num_batches:.4f}',
            'avg_kl': f'{total_kl_loss/num_batches:.4f}'
        })
    
    avg_recon = total_recon_loss / num_batches if num_batches > 0 else 0.0
    avg_kl = total_kl_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_recon, avg_kl


def save_checkpoint(model, optimizer, epoch, recon_loss, kl_loss, run_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
    }
    
    # Save epoch checkpoint
    torch.save(checkpoint, checkpoint_dir / f'epoch_{epoch:02d}.pt')
    
    # Save best model
    if is_best:
        torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
        log.info(f"  ✓ Saved best model (epoch {epoch}, recon={recon_loss:.4f}, kl={kl_loss:.4f})")


def main():
    parser = argparse.ArgumentParser(description='Train probabilistic encoder on NSD data')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Print header
    print("\n" + "="*100)
    print("PROBABILISTIC ENCODER TRAINING (NOVEL APPROACH)".center(100))
    print("="*100)
    print(f"\nExperiment: {config['experiment']['name']}")
    print(f"Subject: {config['data']['subject']}")
    print(f"Session: {config['data']['session']}")
    print("="*100 + "\n")
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{config['experiment']['name']}"
    run_dir = Path(config.get('output_dir', '/bigdata/userhome/students/md5_sd8f61177fd2312b9b32bd118ad1/runs')) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"✓ Created run directory: {run_dir}")
    
    # Save config
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"✓ Using device: {device}")
    if torch.cuda.is_available():
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Create DataLoader
    dataloader = create_dataloader(config)
    
    # Get input dimension from first batch (for model initialization)
    first_batch = next(iter(dataloader))
    fmri_sample = first_batch['fmri']
    if fmri_sample.ndim == 4:  # (B, H, W, D)
        input_dim = fmri_sample.shape[1] * fmri_sample.shape[2] * fmri_sample.shape[3]
    else:
        input_dim = fmri_sample.shape[1]
    
    log.info(f"  Input dimension: {input_dim:,} voxels")
    
    # Create model
    model = ProbabilisticMultiLayerTwoStageEncoder(
        input_dim=input_dim,
        latent_dim=config['model'].get('latent_dim', 512),
        n_blocks=config['model'].get('n_blocks', 4),
        dropout=config['model'].get('dropout', 0.3),
        enabled_layers=['final'],  # Start with just final layer
        predict_text_clip=False,
        kl_weight=config['training'].get('kl_weight', 0.01),
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"✓ Model created with {total_params:,} parameters")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-5)
    )
    
    log.info(f"✓ Training setup complete")
    log.info(f"  Epochs: {config['training']['epochs']}")
    log.info(f"  Batch size: {config['training']['batch_size']}")
    log.info(f"  Learning rate: {config['training']['learning_rate']}")
    log.info(f"  KL weight: {config['training'].get('kl_weight', 0.01)}")
    
    # Training loop
    print("\n" + "="*100)
    print("STARTING TRAINING".center(100))
    print("="*100 + "\n")
    
    best_loss = float('inf')
    metrics_history = []
    
    kl_weight = config['training'].get('kl_weight', 0.01)
    
    for epoch in range(1, config['training']['epochs'] + 1):
        avg_recon, avg_kl = train_epoch(model, dataloader, optimizer, device, epoch, kl_weight)
        
        # Log epoch results
        log.info(f"\nEpoch {epoch}/{config['training']['epochs']} Summary:")
        log.info(f"  Reconstruction Loss: {avg_recon:.4f}")
        log.info(f"  KL Divergence: {avg_kl:.4f}")
        log.info(f"  Total Loss: {avg_recon + kl_weight * avg_kl:.4f}")
        
        # Save metrics
        metrics = {
            'epoch': epoch,
            'recon_loss': avg_recon,
            'kl_loss': avg_kl,
            'total_loss': avg_recon + kl_weight * avg_kl,
            'kl_weight': kl_weight
        }
        metrics_history.append(metrics)
        
        # Save checkpoint
        is_best = avg_recon < best_loss
        if is_best:
            best_loss = avg_recon
        
        save_checkpoint(model, optimizer, epoch, avg_recon, avg_kl, run_dir, is_best)
        
        # Save metrics JSON
        with open(run_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_history, f, indent=2)
    
    print("\n" + "="*100)
    print("TRAINING COMPLETE".center(100))
    print("="*100)
    log.info(f"\n✅ Training complete!")
    log.info(f"  Best reconstruction loss: {best_loss:.4f}")
    log.info(f"  Results saved to: {run_dir}")
    log.info(f"  Best model: {run_dir / 'checkpoints' / 'best_model.pt'}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.error(f"❌ Training failed: {e}", exc_info=True)
        sys.exit(1)
