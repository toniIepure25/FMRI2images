"""
Evaluation Report Generation and Plotting
==========================================

Creates publication-quality plots and comprehensive markdown reports
for embedding and probabilistic evaluations.

Author: Research-grade evaluation suite
Date: January 2026
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns

sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.2)


def plot_calibration_curve(
    calibration_coverage: Dict[float, float],
    output_path: Path,
    title: str = "Calibration Curve"
):
    """
    Plot calibration (reliability) diagram.
    
    Args:
        calibration_coverage: Dict mapping nominal -> empirical coverage
        output_path: Output file path
        title: Plot title
    """
    nominal = np.array(sorted(calibration_coverage.keys()))
    empirical = np.array([calibration_coverage[k] for k in nominal])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    
    # Actual calibration
    ax.plot(nominal, empirical, 'o-', label='Model', linewidth=2, markersize=8)
    
    # Fill region
    ax.fill_between(nominal, nominal, empirical, alpha=0.2)
    
    ax.set_xlabel('Nominal Coverage')
    ax.set_ylabel('Empirical Coverage')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_risk_coverage_curve(
    coverage_levels: List[float],
    risks: List[float],
    output_path: Path,
    title: str = "Risk-Coverage Curve"
):
    """
    Plot risk vs coverage curve.
    
    Args:
        coverage_levels: Coverage fractions (x-axis)
        risks: Corresponding error rates (y-axis)
        output_path: Output file path
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(coverage_levels, risks, 'o-', linewidth=2, markersize=6)
    ax.fill_between(coverage_levels, 0, risks, alpha=0.2)
    
    ax.set_xlabel('Coverage (fraction of samples retained)')
    ax.set_ylabel('Risk (error rate)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, max(risks) * 1.1])
    
    # Add annotation
    aurc = np.trapz(risks, coverage_levels)
    ax.text(0.05, 0.95, f'AURC = {aurc:.3f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_retrieval_vs_gallery_size(
    retrieval_by_size: Dict[int, Dict[str, float]],
    output_path: Path,
    title: str = "Retrieval vs Gallery Size"
):
    """
    Plot retrieval metrics vs gallery size.
    
    Args:
        retrieval_by_size: Dict mapping gallery_size -> {top1, top5, mean_rank, chance}
        output_path: Output file path
        title: Plot title
    """
    sizes = sorted(retrieval_by_size.keys())
    top1 = [retrieval_by_size[s]['top1_accuracy'] for s in sizes]
    top5 = [retrieval_by_size[s]['top5_accuracy'] for s in sizes]
    chance = [retrieval_by_size[s]['chance_top1'] for s in sizes]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.semilogx(sizes, top1, 'o-', label='Top-1', linewidth=2, markersize=6)
    ax.semilogx(sizes, top5, 's-', label='Top-5', linewidth=2, markersize=6)
    ax.semilogx(sizes, chance, 'k--', label='Chance (Top-1)', linewidth=1)
    
    ax.set_xlabel('Gallery Size (log scale)')
    ax.set_ylabel('Retrieval Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_expected_vs_best_of_n(
    n_samples: List[int],
    expected_metric: List[float],
    best_metric: List[float],
    metric_name: str,
    output_path: Path,
    lower_is_better: bool = False
):
    """
    Plot expected-of-N vs best-of-N curves.
    
    Args:
        n_samples: N values
        expected_metric: Expected metric values
        best_metric: Best metric values
        metric_name: Name of metric (e.g., 'PixCorr', 'LPIPS', 'CLIP Similarity')
        output_path: Output file path
        lower_is_better: If True, metric is loss-like (LPIPS)
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(n_samples, expected_metric, 'o-', label='Expected-of-N', linewidth=2, markersize=6)
    ax.plot(n_samples, best_metric, 's-', label='Best-of-N', linewidth=2, markersize=6)
    
    ax.set_xlabel('Number of Samples (N)')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name}: Expected vs Best-of-N')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_samples)
    
    if lower_is_better:
        ax.set_ylim([0, max(expected_metric) * 1.1])
    else:
        ax.set_ylim([min(min(expected_metric), min(best_metric)) * 0.9, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_full_report(
    embedding_metrics: Optional[Dict] = None,
    probabilistic_metrics: Optional[Dict] = None,
    reconstruction_metrics: Optional[Dict] = None,
    sampling_metrics: Optional[Dict] = None,
    output_dir: Path = Path("outputs/eval"),
    experiment_name: str = "Experiment"
):
    """
    Generate comprehensive evaluation report with all plots.
    
    Args:
        embedding_metrics: Embedding evaluation results
        probabilistic_metrics: Probabilistic evaluation results
        reconstruction_metrics: Reconstruction evaluation results
        sampling_metrics: Sampling evaluation results
        output_dir: Output directory
        experiment_name: Experiment name for title
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "full_evaluation_report.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# Full Evaluation Report: {experiment_name}\n\n")
        f.write(f"Generated evaluation report with all metrics and visualizations.\n\n")
        
        # Stage 1: Embeddings
        if embedding_metrics:
            f.write(f"## Stage 1: Embedding Evaluation\n\n")
            
            # Retrieval table
            retr = embedding_metrics.get('retrieval', {})
            f.write(f"### Retrieval Metrics\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            for k, v in retr.items():
                f.write(f"| {k} | {v:.3f} |\n")
            f.write("\n")
            
            # Gallery size plot
            if 'retrieval_by_gallery_size' in embedding_metrics:
                plot_path = plots_dir / "retrieval_vs_gallery_size.png"
                plot_retrieval_vs_gallery_size(
                    {int(k): v for k, v in embedding_metrics['retrieval_by_gallery_size'].items()},
                    plot_path
                )
                f.write(f"![Retrieval vs Gallery Size](plots/retrieval_vs_gallery_size.png)\n\n")
        
        # Stage 1: Probabilistic
        if probabilistic_metrics:
            f.write(f"## Stage 1: Probabilistic Evaluation (Novel)\n\n")
            
            # Proper scoring rules
            psr = probabilistic_metrics.get('proper_scoring_rules', {})
            f.write(f"### Proper Scoring Rules\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| NLL | {psr.get('nll_mean', 0):.2f} ± {psr.get('nll_std', 0):.2f} |\n")
            f.write(f"| Energy Score | {psr.get('energy_score_mean', 0):.3f} ± {psr.get('energy_score_std', 0):.3f} |\n\n")
            
            # Calibration
            if 'calibration' in probabilistic_metrics:
                calib = probabilistic_metrics['calibration']
                plot_path = plots_dir / "calibration_curve.png"
                plot_calibration_curve(
                    {float(k): float(v) for k, v in calib.get('coverage_by_nominal', {}).items()},
                    plot_path
                )
                f.write(f"### Calibration\n\n")
                f.write(f"![Calibration Curve](plots/calibration_curve.png)\n\n")
                f.write(f"**Calibration Error**: {calib.get('mean_absolute_error', 0):.3f}\n\n")
            
            # Risk-coverage
            if 'risk_coverage' in probabilistic_metrics:
                # Note: We need to extract coverage_levels and risks from somewhere
                # This would require storing them in the metrics dict
                f.write(f"### Risk-Coverage Analysis\n\n")
                f.write(f"**AURC**: {probabilistic_metrics['risk_coverage'].get('aurc', 0):.3f}\n\n")
        
        # Stage 2: Reconstructions
        if reconstruction_metrics:
            f.write(f"## Stage 2: Image Reconstruction\n\n")
            
            low_level = reconstruction_metrics.get('low_level', {})
            f.write(f"### Low-Level Metrics\n\n")
            f.write(f"| Metric | Mean | Std |\n")
            f.write(f"|--------|------|-----|\n")
            f.write(f"| PixCorr | {low_level.get('pixcorr_mean', 0):.3f} | {low_level.get('pixcorr_std', 0):.3f} |\n")
            f.write(f"| SSIM | {low_level.get('ssim_mean', 0):.3f} | {low_level.get('ssim_std', 0):.3f} |\n")
            f.write(f"| PSNR | {low_level.get('psnr_mean', 0):.1f} dB | {low_level.get('psnr_std', 0):.1f} |\n\n")
            
            perceptual = reconstruction_metrics.get('perceptual', {})
            f.write(f"### Perceptual Metrics\n\n")
            f.write(f"| Metric | Mean | Std |\n")
            f.write(f"|--------|------|-----|\n")
            f.write(f"| LPIPS | {perceptual.get('lpips_mean', 0):.3f} | {perceptual.get('lpips_std', 0):.3f} |\n\n")
            
            semantic = reconstruction_metrics.get('semantic', {})
            f.write(f"### Semantic Metrics\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| CLIP Sim | {semantic.get('clip_sim_mean', 0):.3f} ± {semantic.get('clip_sim_std', 0):.3f} |\n")
            f.write(f"| CLIP 2AFC | {semantic.get('clip_2afc_accuracy', 0):.1%} |\n\n")
        
        # Sampling
        if sampling_metrics:
            f.write(f"## End-to-End: Sampling Evaluation\n\n")
            
            n_samples = sampling_metrics.get('n_samples', [])
            
            # PixCorr plot
            if 'expected_of_n' in sampling_metrics and 'best_of_n' in sampling_metrics:
                exp = sampling_metrics['expected_of_n']
                best = sampling_metrics['best_of_n']
                
                if 'pixcorr' in exp and 'pixcorr' in best:
                    plot_path = plots_dir / "sampling_pixcorr.png"
                    plot_expected_vs_best_of_n(
                        n_samples, exp['pixcorr'], best['pixcorr'],
                        'PixCorr', plot_path
                    )
                    f.write(f"### PixCorr: Expected vs Best-of-N\n\n")
                    f.write(f"![PixCorr Sampling](plots/sampling_pixcorr.png)\n\n")
                
                if 'clip_sim' in exp and 'clip_sim' in best:
                    plot_path = plots_dir / "sampling_clip_sim.png"
                    plot_expected_vs_best_of_n(
                        n_samples, exp['clip_sim'], best['clip_sim'],
                        'CLIP Similarity', plot_path
                    )
                    f.write(f"### CLIP Similarity: Expected vs Best-of-N\n\n")
                    f.write(f"![CLIP Sim Sampling](plots/sampling_clip_sim.png)\n\n")
        
        f.write(f"---\n\n")
        f.write(f"Report generated by research-grade evaluation suite.\n")
    
    print(f"✓ Full report saved to {report_path}")


def create_comparison_table(
    experiments: List[Dict[str, any]],
    output_path: Path
):
    """
    Create comparison table across multiple experiments.
    
    Args:
        experiments: List of dicts with keys: 'name', 'embedding_metrics', 'probabilistic_metrics', etc.
        output_path: Output markdown file path
    """
    with open(output_path, 'w') as f:
        f.write("# Experiment Comparison\n\n")
        
        f.write("## Stage 1: Embedding Retrieval\n\n")
        f.write("| Experiment | Top-1 | Top-5 | Top-10 | Mean Rank | 2AFC |\n")
        f.write("|------------|-------|-------|--------|-----------|------|\n")
        
        for exp in experiments:
            name = exp.get('name', 'Unknown')
            emb = exp.get('embedding_metrics', {}).get('retrieval', {})
            ident = exp.get('embedding_metrics', {}).get('identification', {})
            
            f.write(f"| {name} | "
                   f"{emb.get('top1_accuracy', 0):.3f} | "
                   f"{emb.get('top5_accuracy', 0):.3f} | "
                   f"{emb.get('top10_accuracy', 0):.3f} | "
                   f"{emb.get('mean_rank', 0):.1f} | "
                   f"{ident.get('twoafc_accuracy', 0):.3f} |\n")
        
        f.write("\n## Stage 1: Probabilistic (Novel)\n\n")
        f.write("| Experiment | Bayesian Top-1 | Cosine Top-1 | Improvement | Calib Error | AURC |\n")
        f.write("|------------|----------------|--------------|-------------|-------------|------|\n")
        
        for exp in experiments:
            name = exp.get('name', 'Unknown')
            prob = exp.get('probabilistic_metrics', {})
            bayes_retr = prob.get('bayesian_retrieval', {})
            calib = prob.get('calibration', {})
            risk = prob.get('risk_coverage', {})
            
            f.write(f"| {name} | "
                   f"{bayes_retr.get('top1_accuracy', 0):.3f} | "
                   f"{prob.get('cosine_top1', 0):.3f} | "
                   f"{bayes_retr.get('improvement_vs_cosine', 0):+.3f} | "
                   f"{calib.get('mean_absolute_error', 0):.3f} | "
                   f"{risk.get('aurc', 0):.3f} |\n")
    
    print(f"✓ Comparison table saved to {output_path}")
