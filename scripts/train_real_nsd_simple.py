#!/usr/bin/env python3
"""
Quick demo: Train on REAL NSD data (simplified version without optional dependencies).
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fmri2img.data.torch_dataset import NSDIterableDataset
from fmri2img.models.simple_cnn import SimpleCNN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def setup_experiment(config: Dict[str, Any]) -> Tuple[Path, torch.device]:
    """Set up experiment directory and device."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{config['experiment']['name']}"
    run_dir = Path(config.get('output_dir', 'runs')) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    (run_dir / 'logs').mkdir(exist_ok=True)
    (run_dir / 'metrics').mkdir(exist_ok=True)
    
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    log.info(f"✓ Created run directory: {run_dir}")
    
    device = torch.device(config['training'].get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    log.info(f"✓ Using device: {device}")
    
    if device.type == 'cuda':
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    return run_dir, device


def create_dataloader(config: Dict[str, Any]) -> DataLoader:
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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
    
    for batch in pbar:
        fmri = batch['fmri'].to(device)
        # Add channel dimension: [batch, H, W, D] -> [batch, 1, H, W, D]
        if fmri.ndim == 4:
            fmri = fmri.unsqueeze(1)
        batch_size = fmri.shape[0]
        
        # TODO: Load actual CLIP embeddings from cache instead of random targets
        # For now using random targets for smoke testing the training loop
        target = torch.randn(batch_size, 512, device=device)
        target = target / target.norm(dim=1, keepdim=True)  # L2 normalize targets too
        
        optimizer.zero_grad()
        output = model(fmri)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{total_loss/num_batches:.4f}'})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(model, optimizer, epoch, loss, run_dir, is_best=False):
    """Save checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    checkpoint_path = run_dir / 'checkpoints' / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = run_dir / 'checkpoints' / 'best_model.pt'
        torch.save(checkpoint, best_path)
        log.info(f"  ✓ Saved best model: {best_path}")


def save_metrics(losses, run_dir):
    """Save metrics."""
    metrics = {
        'losses': losses,
        'best_loss': min(losses) if losses else float('inf'),
        'best_epoch': (losses.index(min(losses)) + 1) if losses else 0,
        'final_loss': losses[-1] if losses else None,
        'total_epochs': len(losses)
    }
    
    with open(run_dir / 'metrics' / 'summary.json', 'w') as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train on REAL NSD data (simplified)")
    parser.add_argument('--config', required=True, help="Config YAML")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*100)
    print(" " * 25 + "REAL NSD DATA TRAINING (SIMPLIFIED)")
    print("="*100)
    print(f"\nExperiment: {config['experiment']['name']}")
    print(f"Subject: {config['data']['subject']}")
    print(f"Session: {config['data']['session']}")
    print("="*100 + "\n")
    
    run_dir, device = setup_experiment(config)
    
    try:
        dataloader = create_dataloader(config)
        
        model = SimpleCNN(
            input_shape=tuple(config['model']['input_shape']),
            output_dim=config['model']['output_dim']
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        log.info(f"✓ Model created with {total_params:,} parameters")
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
        
        log.info(f"✓ Training setup complete")
        log.info(f"  Epochs: {config['training']['epochs']}")
        log.info(f"  Batch size: {config['training']['batch_size']}")
        log.info(f"  Learning rate: {config['training']['lr']}")
        
        num_epochs = config['training']['epochs']
        best_loss = float('inf')
        losses = []
        
        print("\n" + "="*100)
        print("STARTING TRAINING")
        print("="*100 + "\n")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            avg_loss = train_epoch(model, dataloader, criterion, optimizer, device, epoch)
            losses.append(avg_loss)
            epoch_time = time.time() - start_time
            
            log.info(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")
            
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                log.info(f"  🌟 New best loss: {best_loss:.4f}")
            
            save_checkpoint(model, optimizer, epoch, avg_loss, run_dir, is_best)
            save_metrics(losses, run_dir)
        
        print("\n" + "="*100)
        print("✅ TRAINING COMPLETE!")
        print("="*100)
        print(f"\n🏆 Best Loss: {best_loss:.4f}")
        print(f"📁 Results: {run_dir}")
        print("\n" + "="*100 + "\n")
        
        return 0
        
    except Exception as e:
        log.error(f"❌ Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
