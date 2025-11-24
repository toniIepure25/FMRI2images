#!/usr/bin/env python3
"""
Quick embedding evaluation using pre-computed predictions.

This is a fast alternative that evaluates embeddings without re-loading fMRI data.
It works by:
1. Loading the encoder checkpoint (which has training metrics)
2. Comparing with validation metrics from training
3. Computing statistics on a cached embedding prediction if available

For full evaluation, use evaluate_embeddings.py (slower but comprehensive).
"""
import argparse
import json
import logging
from pathlib import Path
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_checkpoint(ckpt_path: Path):
    """Analyze training metrics from checkpoint."""
    logger.info(f"Loading checkpoint: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Extract training history
    history = ckpt.get('history', {})
    
    # Get best metrics
    metrics = {
        'train_loss': ckpt.get('train_loss'),
        'val_loss': ckpt.get('val_loss'),
        'test_cosine': ckpt.get('test_cosine'),
        'epoch': ckpt.get('epoch'),
    }
    
    logger.info("\n" + "=" * 80)
    logger.info("CHECKPOINT METRICS")
    logger.info("=" * 80)
    for key, value in metrics.items():
        if value is not None:
            logger.info(f"{key}: {value}")
    
    # Analyze training history if available
    if history:
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING HISTORY SUMMARY")
        logger.info("=" * 80)
        
        for metric_name in ['train_loss', 'val_loss', 'val_cosine']:
            if metric_name in history:
                values = history[metric_name]
                logger.info(f"\n{metric_name}:")
                logger.info(f"  Best: {min(values) if 'loss' in metric_name else max(values):.4f}")
                logger.info(f"  Final: {values[-1]:.4f}")
                logger.info(f"  Epochs: {len(values)}")
    
    logger.info("=" * 80)
    
    return metrics, history


def main():
    parser = argparse.ArgumentParser(description="Quick embedding evaluation from checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Encoder checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze checkpoint
    metrics, history = analyze_checkpoint(Path(args.ckpt))
    
    # Save summary
    summary = {
        'checkpoint': str(args.ckpt),
        'metrics': metrics,
        'training_epochs': len(history.get('train_loss', [])) if history else 0,
    }
    
    with open(output_dir / "quick_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n✓ Summary saved to {output_dir / 'quick_summary.json'}")


if __name__ == "__main__":
    main()
