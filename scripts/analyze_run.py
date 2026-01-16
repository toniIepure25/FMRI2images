#!/usr/bin/env python3
"""
Analyze a single experiment run in detail.
Usage: python scripts/analyze_run.py <run_directory>
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_run(run_dir):
    """Analyze a single experiment run."""
    run_path = Path(run_dir)
    
    if not run_path.exists():
        print(f"❌ Run directory not found: {run_dir}")
        return
    
    # Load metrics
    metrics_file = run_path / "metrics" / "summary.json"
    if not metrics_file.exists():
        print(f"❌ Metrics not found: {metrics_file}")
        return
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    # Load config
    config_file = run_path / "config.yaml"
    
    # Print summary
    print("\n" + "="*80)
    print(f"📊 EXPERIMENT ANALYSIS: {run_path.name}")
    print("="*80)
    
    print(f"\n📈 Training Metrics:")
    print(f"   Total Epochs: {len(metrics['losses'])}")
    print(f"   Best Loss: {min(metrics['losses']):.4f} (epoch {metrics['losses'].index(min(metrics['losses'])) + 1})")
    print(f"   Final Loss: {metrics['losses'][-1]:.4f}")
    print(f"   Loss Improvement: {metrics['losses'][0] - metrics['losses'][-1]:.4f}")
    
    print(f"\n📁 Output Files:")
    checkpoints_dir = run_path / "checkpoints"
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob("*.pt"))
        print(f"   Checkpoints: {len(checkpoints)}")
        print(f"   Best model: {'✅' if (checkpoints_dir / 'best_model.pt').exists() else '❌'}")
    
    logs_dir = run_path / "logs"
    if logs_dir.exists():
        print(f"   Log files: {len(list(logs_dir.glob('*.log')))}")
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training loss curve
    epochs = range(1, len(metrics['losses']) + 1)
    axes[0].plot(epochs, metrics['losses'], marker='o', linewidth=2, markersize=6)
    axes[0].axhline(y=min(metrics['losses']), color='r', linestyle='--', 
                    label=f"Best: {min(metrics['losses']):.4f}")
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Loss improvement per epoch
    loss_diffs = [metrics['losses'][i] - metrics['losses'][i-1] 
                  for i in range(1, len(metrics['losses']))]
    axes[1].bar(range(2, len(metrics['losses']) + 1), loss_diffs, color='steelblue')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss Change', fontsize=12)
    axes[1].set_title('Loss Improvement per Epoch', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    output_file = run_path / f"{run_path.name}_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved analysis plot: {output_file}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_run.py <run_directory>")
        sys.exit(1)
    
    analyze_run(sys.argv[1])
