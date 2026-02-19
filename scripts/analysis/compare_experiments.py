#!/usr/bin/env python3
"""
Compare multiple experiment runs and generate comparison plots.

This script loads training metrics from multiple experiment runs,
compares them, and generates publication-quality visualizations.

Usage:
    python scripts/compare_experiments.py run_dir1 run_dir2 run_dir3 [--output OUTPUT_DIR]

Author: Bachelor Thesis - fMRI to Image Reconstruction
"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def load_metrics(run_dir: Path) -> Optional[Dict]:
    """
    Load metrics from a run directory.
    
    Args:
        run_dir: Path to the experiment run directory
        
    Returns:
        Dictionary containing run metadata and metrics, or None if not found
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "metrics" / "summary.json"
    
    if not summary_path.exists():
        print(f"⚠️  Warning: No summary.json found in {run_dir}")
        return None
    
    try:
        with open(summary_path) as f:
            data = json.load(f)
        
        # Extract experiment name from directory
        exp_name = run_dir.name.split('_', 2)[-1] if '_' in run_dir.name else run_dir.name
        
        return {
            'name': exp_name,
            'full_name': run_dir.name,
            'path': str(run_dir),
            'losses': data.get('losses', []),
            'best_loss': min(data.get('losses', [float('inf')])),
            'best_epoch': data.get('losses', []).index(min(data.get('losses', [float('inf')]))) + 1 if data.get('losses') else 0,
            'final_loss': data.get('losses', [])[-1] if data.get('losses') else None,
            'total_epochs': len(data.get('losses', [])),
        }
    except Exception as e:
        print(f"❌ Error loading metrics from {run_dir}: {e}")
        return None

def plot_training_curves(experiments: List[Dict], output_dir: Path) -> None:
    """
    Plot training loss curves for all experiments.
    
    Args:
        experiments: List of experiment data dictionaries
        output_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = sns.color_palette("husl", len(experiments))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for idx, exp in enumerate(experiments):
        epochs = list(range(1, len(exp['losses']) + 1))
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        ax.plot(epochs, exp['losses'], 
                marker=marker, 
                color=color,
                linewidth=2.5,
                markersize=6,
                markevery=max(1, len(epochs) // 10),
                label=exp['name'],
                alpha=0.85)
        
        # Mark best epoch (only if losses exist)
        if exp['losses'] and exp['best_epoch'] > 0:
            best_idx = exp['best_epoch'] - 1
            ax.scatter([exp['best_epoch']], [exp['losses'][best_idx]], 
                      color=color, s=200, marker='*', 
                      edgecolors='black', linewidths=1.5, zorder=10)
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
    ax.set_title('Training Loss Comparison Across Experiments', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, framealpha=0.95, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    
    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved training curves to: {output_path}")
    
    # Also save as PDF for LaTeX
    pdf_path = output_dir / "training_curves.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved PDF version to: {pdf_path}")
    
    plt.close()

def plot_best_loss_comparison(experiments: List[Dict], output_dir: Path) -> None:
    """
    Plot best loss comparison bar chart.
    
    Args:
        experiments: List of experiment data dictionaries
        output_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    names = [exp['name'] for exp in experiments]
    best_losses = [exp['best_loss'] for exp in experiments]
    colors = sns.color_palette("husl", len(experiments))
    
    bars = ax.bar(names, best_losses, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Add value labels on bars
    for bar, loss in zip(bars, best_losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}',
                ha='center', va='bottom', 
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Highlight the best performer
    best_idx = best_losses.index(min(best_losses))
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    ax.set_ylabel('Best Loss Achieved', fontsize=14, fontweight='bold')
    ax.set_title('Best Loss Comparison: Lower is Better', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=11)
    
    # Rotate x-labels if needed
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    
    output_path = output_dir / "best_loss_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved best loss comparison to: {output_path}")
    
    pdf_path = output_dir / "best_loss_comparison.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved PDF version to: {pdf_path}")
    
    plt.close()

def plot_loss_convergence(experiments: List[Dict], output_dir: Path) -> None:
    """
    Plot loss improvement over epochs for each experiment.
    
    Args:
        experiments: List of experiment data dictionaries
        output_dir: Directory to save the plot
    """
    fig, axes = plt.subplots(1, len(experiments), figsize=(6*len(experiments), 5))
    if len(experiments) == 1:
        axes = [axes]
    
    for ax, exp in zip(axes, experiments):
        losses = exp['losses']
        
        # Skip if no losses
        if not losses:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(exp['name'], fontsize=13, fontweight='bold')
            continue
        
        # Calculate improvement from first epoch
        improvements = [(losses[0] - loss) / losses[0] * 100 for loss in losses]
        epochs = list(range(1, len(losses) + 1))
        
        ax.plot(epochs, improvements, marker='o', linewidth=2.5, markersize=6, 
               color='steelblue', markevery=max(1, len(epochs) // 10))
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss Improvement (%)', fontsize=12, fontweight='bold')
        ax.set_title(exp['name'], fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=10)
    
    plt.suptitle('Loss Convergence Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / "loss_convergence.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved loss convergence to: {output_path}")
    
    pdf_path = output_dir / "loss_convergence.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved PDF version to: {pdf_path}")
    
    plt.close()

def create_comparison_table(experiments: List[Dict], output_dir: Path) -> pd.DataFrame:
    """
    Create a detailed comparison table.
    
    Args:
        experiments: List of experiment data dictionaries
        output_dir: Directory to save the table
        
    Returns:
        DataFrame containing the comparison table
    """
    data = []
    for exp in experiments:
        improvement = (exp['losses'][0] - exp['best_loss']) / exp['losses'][0] * 100 if exp['losses'] else 0
        
        data.append({
            'Experiment': exp['name'],
            'Best Loss': f"{exp['best_loss']:.4f}",
            'Best Epoch': exp['best_epoch'],
            'Final Loss': f"{exp['final_loss']:.4f}" if exp['final_loss'] else "N/A",
            'Total Epochs': exp['total_epochs'],
            'Improvement (%)': f"{improvement:.2f}%",
            'Run Directory': exp['full_name']
        })
    
    df = pd.DataFrame(data)
    
    # Sort by best loss
    df = df.sort_values('Best Loss', ascending=True)
    
    # Save as CSV
    csv_path = output_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved comparison table to: {csv_path}")
    
    # Save as LaTeX
    latex_path = output_dir / "comparison_table.tex"
    df.to_latex(latex_path, index=False, column_format='l' + 'c'*(len(df.columns)-1))
    print(f"✓ Saved LaTeX table to: {latex_path}")
    
    # Print to console
    print("\n" + "="*100)
    print("EXPERIMENT COMPARISON TABLE")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")
    
    return df

def generate_statistical_summary(experiments: List[Dict], output_dir: Path) -> None:
    """
    Generate statistical summary of experiments.
    
    Args:
        experiments: List of experiment data dictionaries
        output_dir: Directory to save the summary
    """
    summary = {
        'total_experiments': len(experiments),
        'experiments': []
    }
    
    for exp in experiments:
        exp_stats = {
            'name': exp['name'],
            'best_loss': float(exp['best_loss']),
            'best_epoch': int(exp['best_epoch']),
            'final_loss': float(exp['final_loss']) if exp['final_loss'] else None,
            'total_epochs': int(exp['total_epochs']),
            'loss_std': float(np.std(exp['losses'])),
            'loss_variance': float(np.var(exp['losses'])),
            'convergence_speed': float(exp['losses'][0] - exp['best_loss']) / exp['best_epoch'] if exp['best_epoch'] > 0 else 0
        }
        summary['experiments'].append(exp_stats)
    
    # Find winner
    best_exp = min(experiments, key=lambda x: x['best_loss'])
    summary['winner'] = {
        'name': best_exp['name'],
        'loss': float(best_exp['best_loss'])
    }
    
    # Save JSON
    json_path = output_dir / "statistical_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved statistical summary to: {json_path}")

def generate_report(experiments: List[Dict], output_dir: Path) -> None:
    """
    Generate a markdown report.
    
    Args:
        experiments: List of experiment data dictionaries
        output_dir: Directory to save the report
    """
    report_path = output_dir / "comparison_report.md"
    
    best_exp = min(experiments, key=lambda x: x['best_loss'])
    
    with open(report_path, 'w') as f:
        f.write("# Experiment Comparison Report\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Experiments:** {len(experiments)}\n\n")
        
        f.write("---\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"The **{best_exp['name']}** experiment achieved the best performance with a loss of **{best_exp['best_loss']:.4f}** at epoch {best_exp['best_epoch']}.\n\n")
        
        f.write("---\n\n")
        
        f.write("## Detailed Results\n\n")
        for idx, exp in enumerate(sorted(experiments, key=lambda x: x['best_loss']), 1):
            improvement = (exp['losses'][0] - exp['best_loss']) / exp['losses'][0] * 100
            
            f.write(f"### {idx}. {exp['name']}\n\n")
            f.write(f"- **Best Loss:** {exp['best_loss']:.4f} (Epoch {exp['best_epoch']})\n")
            f.write(f"- **Final Loss:** {exp['final_loss']:.4f}\n")
            f.write(f"- **Total Epochs:** {exp['total_epochs']}\n")
            f.write(f"- **Loss Improvement:** {improvement:.2f}%\n")
            f.write(f"- **Loss Std Dev:** {np.std(exp['losses']):.4f}\n")
            f.write(f"- **Run Directory:** `{exp['path']}`\n\n")
        
        f.write("---\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("### Training Loss Curves\n")
        f.write("![Training Curves](training_curves.png)\n\n")
        f.write("### Best Loss Comparison\n")
        f.write("![Best Loss Comparison](best_loss_comparison.png)\n\n")
        f.write("### Loss Convergence Analysis\n")
        f.write("![Loss Convergence](loss_convergence.png)\n\n")
        
        f.write("---\n\n")
        
        f.write("## Conclusion\n\n")
        f.write(f"Based on the comparison of {len(experiments)} experiments, ")
        f.write(f"**{best_exp['name']}** demonstrates superior performance ")
        f.write(f"with the lowest loss of {best_exp['best_loss']:.4f}. ")
        
        if len(experiments) > 1:
            second_best = sorted(experiments, key=lambda x: x['best_loss'])[1]
            diff = ((second_best['best_loss'] - best_exp['best_loss']) / second_best['best_loss'] * 100)
            f.write(f"This represents a {diff:.2f}% improvement over the second-best method.\n\n")
        
        f.write("---\n\n")
        f.write("*Report generated automatically by experiment comparison script*\n")
    
    print(f"✓ Saved comparison report to: {report_path}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compare multiple experiment runs with publication-quality visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compare_experiments.py run1 run2 run3
  python scripts/compare_experiments.py run1 run2 --output results/comparison
        """
    )
    parser.add_argument('run_dirs', nargs='+', help="Paths to run directories")
    parser.add_argument('--output', '-o', default='experiment_comparison', 
                       help="Output directory for comparison results (default: experiment_comparison)")
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*100)
    print(" " * 30 + "EXPERIMENT COMPARISON TOOL")
    print("="*100 + "\n")
    
    # Load all experiments
    print("📂 Loading Experiments...\n")
    
    experiments = []
    for run_dir in args.run_dirs:
        data = load_metrics(Path(run_dir))
        if data:
            experiments.append(data)
            print(f"  ✓ Loaded: {data['name']} (Best Loss: {data['best_loss']:.4f})")
        else:
            print(f"  ✗ Failed: {run_dir}")
    
    if len(experiments) == 0:
        print("\n❌ No valid experiments found!")
        return 1
    
    print(f"\n✓ Successfully loaded {len(experiments)} experiments\n")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"📁 Output directory: {output_dir.absolute()}\n")
    
    # Generate all analyses
    print("📊 Generating Analysis...\n")
    
    create_comparison_table(experiments, output_dir)
    generate_statistical_summary(experiments, output_dir)
    
    print("\n📈 Generating Visualizations...\n")
    
    plot_training_curves(experiments, output_dir)
    plot_best_loss_comparison(experiments, output_dir)
    plot_loss_convergence(experiments, output_dir)
    
    print("\n📝 Generating Report...\n")
    
    generate_report(experiments, output_dir)
    
    # Final summary
    print("\n" + "="*100)
    print("✅ COMPARISON COMPLETE!")
    print("="*100)
    
    best_exp = min(experiments, key=lambda x: x['best_loss'])
    print(f"\n🏆 Winner: {best_exp['name']} (Loss: {best_exp['best_loss']:.4f})")
    
    print(f"\n📁 All results saved to: {output_dir.absolute()}")
    print("\n📋 Generated files:")
    print(f"  • comparison_report.md           - Detailed markdown report")
    print(f"  • comparison_table.csv           - Data table (CSV format)")
    print(f"  • comparison_table.tex           - LaTeX table for papers")
    print(f"  • statistical_summary.json       - Statistical summary")
    print(f"  • training_curves.png/.pdf       - Training curves plot")
    print(f"  • best_loss_comparison.png/.pdf  - Best loss comparison")
    print(f"  • loss_convergence.png/.pdf      - Convergence analysis")
    
    print("\n" + "="*100 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
