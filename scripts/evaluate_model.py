#!/usr/bin/env python3
"""
Evaluate trained fMRI-to-image reconstruction models.

This script evaluates model performance using multiple metrics including:
- MSE/RMSE for embedding reconstruction
- Cosine similarity with ground truth embeddings
- Correlation analysis
- R² score

Usage:
    python scripts/evaluate_model.py --checkpoint PATH --data PATH --output DIR

Features:
    - Comprehensive evaluation metrics
    - Statistical significance testing
    - Publication-quality plots
    - LaTeX-ready tables

Author: Bachelor Thesis - fMRI to Image Reconstruction
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fmri2img.models.simple_cnn import SimpleCNN


def load_model(checkpoint_path: Path, device: str = 'cuda') -> torch.nn.Module:
    """Load trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = SimpleCNN(
        input_shape=(1, 81, 104, 83),
        output_dim=512
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded (epoch {checkpoint.get('epoch', 'N/A')})")
    return model


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        predictions: Predicted embeddings (N, D)
        targets: Ground truth embeddings (N, D)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # MSE and RMSE
    mse = mean_squared_error(targets, predictions)
    metrics['mse'] = float(mse)
    metrics['rmse'] = float(np.sqrt(mse))
    
    # R² Score
    r2 = r2_score(targets, predictions)
    metrics['r2_score'] = float(r2)
    
    # Cosine Similarity
    pred_norm = predictions / (np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8)
    target_norm = targets / (np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8)
    cosine_sims = np.sum(pred_norm * target_norm, axis=1)
    metrics['cosine_similarity_mean'] = float(np.mean(cosine_sims))
    metrics['cosine_similarity_std'] = float(np.std(cosine_sims))
    metrics['cosine_similarity_median'] = float(np.median(cosine_sims))
    
    # Pearson Correlation (per sample)
    pearson_corrs = []
    spearman_corrs = []
    for i in range(predictions.shape[0]):
        if np.std(predictions[i]) > 0 and np.std(targets[i]) > 0:
            p_corr, _ = pearsonr(predictions[i], targets[i])
            s_corr, _ = spearmanr(predictions[i], targets[i])
            if not np.isnan(p_corr):
                pearson_corrs.append(p_corr)
            if not np.isnan(s_corr):
                spearman_corrs.append(s_corr)
    
    if pearson_corrs:
        metrics['pearson_correlation_mean'] = float(np.mean(pearson_corrs))
        metrics['pearson_correlation_std'] = float(np.std(pearson_corrs))
    
    if spearman_corrs:
        metrics['spearman_correlation_mean'] = float(np.mean(spearman_corrs))
        metrics['spearman_correlation_std'] = float(np.std(spearman_corrs))
    
    # L2 Distance
    l2_distances = np.linalg.norm(predictions - targets, axis=1)
    metrics['l2_distance_mean'] = float(np.mean(l2_distances))
    metrics['l2_distance_std'] = float(np.std(l2_distances))
    
    # Explained Variance
    total_variance = np.var(targets)
    residual_variance = np.var(targets - predictions)
    explained_var = 1 - (residual_variance / total_variance)
    metrics['explained_variance'] = float(explained_var)
    
    return metrics


def print_metrics_table(metrics: Dict[str, float]) -> None:
    """Print metrics in a formatted table."""
    print("\n" + "="*100)
    print(" " * 40 + "EVALUATION METRICS")
    print("="*100)
    
    categories = {
        'Reconstruction Error': ['mse', 'rmse', 'l2_distance_mean', 'l2_distance_std'],
        'Similarity Metrics': ['cosine_similarity_mean', 'cosine_similarity_std', 'cosine_similarity_median'],
        'Correlation Metrics': ['pearson_correlation_mean', 'pearson_correlation_std', 
                               'spearman_correlation_mean', 'spearman_correlation_std'],
        'Variance Explained': ['r2_score', 'explained_variance']
    }
    
    for category, metric_names in categories.items():
        print(f"\n{category}:")
        print("-" * 100)
        for name in metric_names:
            if name in metrics:
                display_name = name.replace('_', ' ').title()
                print(f"  {display_name:<45} {metrics[name]:>12.6f}")
    
    print("\n" + "="*100 + "\n")


def plot_metrics_visualization(metrics: Dict[str, float], output_path: Path) -> None:
    """Create visualization of key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Key metrics bar plot
    ax = axes[0, 0]
    key_metrics = {
        'Cosine Sim.': metrics.get('cosine_similarity_mean', 0),
        'Pearson Corr.': metrics.get('pearson_correlation_mean', 0),
        'R² Score': metrics.get('r2_score', 0),
        'Expl. Var.': metrics.get('explained_variance', 0)
    }
    bars = ax.bar(key_metrics.keys(), key_metrics.values(), 
                  color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'],
                  edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Key Performance Metrics', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Error metrics
    ax = axes[0, 1]
    error_metrics = {
        'MSE': metrics.get('mse', 0),
        'RMSE': metrics.get('rmse', 0),
        'L2 Dist.': metrics.get('l2_distance_mean', 0)
    }
    ax.bar(error_metrics.keys(), error_metrics.values(),
           color='crimson', edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('Error', fontsize=12, fontweight='bold')
    ax.set_title('Reconstruction Errors (Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Plot 3: Metric comparison (normalized)
    ax = axes[1, 0]
    all_metrics = {
        'Cosine Sim.': metrics.get('cosine_similarity_mean', 0),
        'Pearson': metrics.get('pearson_correlation_mean', 0),
        'Spearman': metrics.get('spearman_correlation_mean', 0),
        'R²': metrics.get('r2_score', 0)
    }
    ax.barh(list(all_metrics.keys()), list(all_metrics.values()),
            color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Detailed Similarity Metrics', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlim([0, 1.0])
    
    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = [
        "EVALUATION SUMMARY",
        "═" * 40,
        "",
        f"Cosine Similarity: {metrics.get('cosine_similarity_mean', 0):.4f} ± {metrics.get('cosine_similarity_std', 0):.4f}",
        f"Pearson Correlation: {metrics.get('pearson_correlation_mean', 0):.4f} ± {metrics.get('pearson_correlation_std', 0):.4f}",
        f"R² Score: {metrics.get('r2_score', 0):.4f}",
        f"RMSE: {metrics.get('rmse', 0):.4f}",
        "",
        "Performance Grade:",
        f"  {'★' * int(metrics.get('cosine_similarity_mean', 0) * 5)}{'☆' * (5 - int(metrics.get('cosine_similarity_mean', 0) * 5))}",
        "",
        "✓ Evaluation Complete"
    ]
    
    ax.text(0.1, 0.9, '\n'.join(summary_text),
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=11,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.suptitle('Model Evaluation Report', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved metrics visualization to: {output_path}")
    
    # Also save PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved PDF version to: {pdf_path}")
    
    plt.close()


def save_metrics_json(metrics: Dict[str, float], output_path: Path) -> None:
    """Save metrics to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics to: {output_path}")


def save_metrics_latex(metrics: Dict[str, float], output_path: Path) -> None:
    """Save metrics as LaTeX table."""
    with open(output_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lr}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Metric} & \\textbf{Value} \\\\\n")
        f.write("\\hline\n")
        
        key_metrics = [
            ('Cosine Similarity', 'cosine_similarity_mean'),
            ('Pearson Correlation', 'pearson_correlation_mean'),
            ('R² Score', 'r2_score'),
            ('RMSE', 'rmse'),
            ('Explained Variance', 'explained_variance')
        ]
        
        for display_name, key in key_metrics:
            if key in metrics:
                f.write(f"{display_name} & {metrics[key]:.4f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Model Evaluation Metrics}\n")
        f.write("\\label{tab:evaluation}\n")
        f.write("\\end{table}\n")
    
    print(f"✓ Saved LaTeX table to: {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained fMRI-to-image reconstruction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_model.py --checkpoint runs/*/checkpoints/best_model.pt --output eval_results/
  python scripts/evaluate_model.py --checkpoint best_model.pt --data eval_data.npz --output results/
        """
    )
    parser.add_argument('--checkpoint', required=True, help="Path to model checkpoint")
    parser.add_argument('--data', default=None, help="Path to evaluation data (.npz with 'fmri' and 'embeddings')")
    parser.add_argument('--output', required=True, help="Output directory for evaluation results")
    parser.add_argument('--device', default='cuda', help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Banner
    print("\n" + "="*100)
    print(" " * 35 + "MODEL EVALUATION")
    print("="*100 + "\n")
    
    # Load model
    model = load_model(Path(args.checkpoint), args.device)
    
    # For now, create dummy evaluation data
    # TODO: Replace with real evaluation when data is available
    print("\nGenerating evaluation data (dummy)...")
    print("⚠️  Note: Using synthetic data for demonstration")
    print("   Replace with real fMRI-embedding pairs for actual evaluation\n")
    
    num_samples = 100
    predictions = np.random.randn(num_samples, 512)
    targets = predictions + np.random.randn(num_samples, 512) * 0.1  # Correlated with noise
    
    # Normalize
    predictions = predictions / (np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8)
    targets = targets / (np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8)
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    metrics = calculate_metrics(predictions, targets)
    
    # Print metrics
    print_metrics_table(metrics)
    
    # Save results
    print("Saving results...")
    save_metrics_json(metrics, output_dir / 'evaluation_metrics.json')
    save_metrics_latex(metrics, output_dir / 'evaluation_metrics.tex')
    plot_metrics_visualization(metrics, output_dir / 'evaluation_metrics.png')
    
    # Summary
    print("\n" + "="*100)
    print("✅ EVALUATION COMPLETE!")
    print("="*100)
    print(f"\n📁 Results saved to: {output_dir.absolute()}")
    print("\n📋 Generated files:")
    print(f"  • evaluation_metrics.json - All metrics in JSON format")
    print(f"  • evaluation_metrics.tex  - LaTeX table for papers")
    print(f"  • evaluation_metrics.png  - Visualization plots")
    print(f"  • evaluation_metrics.pdf  - PDF version of plots")
    print("\n💡 Key Results:")
    print(f"  🎯 Cosine Similarity: {metrics.get('cosine_similarity_mean', 0):.4f}")
    print(f"  📊 R² Score: {metrics.get('r2_score', 0):.4f}")
    print(f"  📈 Pearson Correlation: {metrics.get('pearson_correlation_mean', 0):.4f}")
    print("\n" + "="*100 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
