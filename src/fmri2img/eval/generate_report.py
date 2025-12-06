#!/usr/bin/env python3
"""
Automated Results Reporting for fMRI Reconstruction
==================================================

Generates comprehensive reports from evaluation results:
- LaTeX tables for papers
- Markdown summaries for documentation
- Statistical significance tests
- Performance visualizations

Usage:
    # Generate report from evaluation results
    python scripts/generate_report.py \\
        --results-dir outputs/eval_comprehensive \\
        --output-dir outputs/reports \\
        --report-type full
    
    # Compare multiple runs
    python scripts/generate_report.py \\
        --results-dir outputs/ablations/infonce \\
        --output-dir outputs/reports/ablation_infonce \\
        --report-type ablation
    
    # Quick summary
    python scripts/generate_report.py \\
        --results-dir outputs/eval_comprehensive \\
        --output-dir outputs/reports \\
        --report-type summary
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load evaluation results from directory."""
    results = {}
    
    # Check for different result files
    if (results_dir / "retrieval_metrics.json").exists():
        with open(results_dir / "retrieval_metrics.json", "r") as f:
            results["retrieval"] = json.load(f)
    
    if (results_dir / "generation_metrics.json").exists():
        with open(results_dir / "generation_metrics.json", "r") as f:
            results["generation"] = json.load(f)
    
    if (results_dir / "brain_alignment.json").exists():
        with open(results_dir / "brain_alignment.json", "r") as f:
            results["brain_alignment"] = json.load(f)
    
    return results


def create_latex_table(
    data: pd.DataFrame,
    caption: str,
    label: str,
    output_path: Path,
    bold_best: bool = True
):
    """
    Create LaTeX table from DataFrame.
    
    Args:
        data: DataFrame with results
        caption: Table caption
        label: Table label for referencing
        output_path: Path to save .tex file
        bold_best: Bold the best value in each column
    """
    latex = r"\begin{table}[htbp]" + "\n"
    latex += r"\centering" + "\n"
    latex += r"\small" + "\n"
    
    # Column format
    n_cols = len(data.columns)
    latex += r"\begin{tabular}{l" + "c" * (n_cols - 1) + "}\n"
    latex += r"\toprule" + "\n"
    
    # Header
    header = " & ".join([col.replace("_", r"\_") for col in data.columns])
    latex += header + r" \\" + "\n"
    latex += r"\midrule" + "\n"
    
    # Find best values if requested
    if bold_best:
        best_indices = {}
        for col in data.columns[1:]:  # Skip first column (usually labels)
            if data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Higher is better for most metrics (R@K, cosine, etc.)
                best_indices[col] = data[col].idxmax()
    
    # Rows
    for idx, row in data.iterrows():
        row_str = []
        for i, (col, val) in enumerate(row.items()):
            if isinstance(val, float):
                val_str = f"{val:.4f}"
                # Bold if best
                if bold_best and col in best_indices and best_indices[col] == idx:
                    val_str = r"\textbf{" + val_str + "}"
            else:
                val_str = str(val).replace("_", r"\_")
            row_str.append(val_str)
        
        latex += " & ".join(row_str) + r" \\" + "\n"
    
    latex += r"\bottomrule" + "\n"
    latex += r"\end{tabular}" + "\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{tab:{label}}}\n"
    latex += r"\end{table}" + "\n"
    
    # Save
    with open(output_path, "w") as f:
        f.write(latex)
    
    logger.info(f"Saved LaTeX table to {output_path}")


def create_markdown_summary(
    results: Dict[str, Any],
    output_path: Path
):
    """Create Markdown summary of results."""
    md = "# Evaluation Results Summary\n\n"
    
    # Retrieval metrics
    if "retrieval" in results:
        md += "## Retrieval Performance\n\n"
        md += "| Metric | Value |\n"
        md += "|--------|-------|\n"
        
        retrieval = results["retrieval"]
        for k, v in retrieval.items():
            if isinstance(v, float):
                md += f"| {k} | {v:.4f} |\n"
            else:
                md += f"| {k} | {v} |\n"
        md += "\n"
    
    # Generation metrics
    if "generation" in results:
        md += "## Generation Quality\n\n"
        gen = results["generation"]
        
        if isinstance(gen, dict) and "strategies" in gen:
            # Multi-strategy comparison
            md += "| Strategy | CLIPScore | SSIM | LPIPS |\n"
            md += "|----------|-----------|------|-------|\n"
            
            for strategy, metrics in gen["strategies"].items():
                clip_score = metrics.get("CLIPScore", 0.0)
                ssim = metrics.get("SSIM", 0.0)
                lpips = metrics.get("LPIPS", 0.0)
                md += f"| {strategy} | {clip_score:.4f} | {ssim:.4f} | {lpips:.4f} |\n"
        md += "\n"
    
    # Brain alignment
    if "brain_alignment" in results:
        md += "## Brain Alignment\n\n"
        md += "| Metric | Value |\n"
        md += "|--------|-------|\n"
        
        ba = results["brain_alignment"]
        for k, v in ba.items():
            if isinstance(v, float):
                md += f"| {k} | {v:.4f} |\n"
        md += "\n"
    
    # Save
    with open(output_path, "w") as f:
        f.write(md)
    
    logger.info(f"Saved Markdown summary to {output_path}")


def create_performance_plot(
    data: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    output_path: Path,
    title: str = "Performance Comparison",
    xlabel: str = "Parameter",
    ylabel: str = "Score"
):
    """
    Create line plot showing performance across parameter values.
    
    Args:
        data: DataFrame with results
        x_col: Column to use for x-axis
        y_cols: Columns to plot
        output_path: Path to save figure
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    plt.figure(figsize=(10, 6))
    
    for y_col in y_cols:
        if y_col in data.columns:
            plt.plot(data[x_col], data[y_col], marker='o', label=y_col, linewidth=2)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved performance plot to {output_path}")


def statistical_comparison(
    results: Dict[str, List[float]],
    baseline: str
) -> pd.DataFrame:
    """
    Perform statistical tests comparing strategies to baseline.
    
    Args:
        results: Dict mapping strategy name to list of per-sample scores
        baseline: Name of baseline strategy
        
    Returns:
        comparison_df: DataFrame with test results
    """
    if baseline not in results:
        logger.warning(f"Baseline '{baseline}' not found in results")
        return pd.DataFrame()
    
    baseline_scores = results[baseline]
    comparisons = []
    
    for strategy, scores in results.items():
        if strategy == baseline:
            continue
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores, baseline_scores)
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(scores) - np.mean(baseline_scores)
        pooled_std = np.sqrt((np.std(scores)**2 + np.std(baseline_scores)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
        
        comparisons.append({
            "Strategy": strategy,
            "Mean": np.mean(scores),
            "Std": np.std(scores),
            "vs_Baseline": mean_diff,
            "t_stat": t_stat,
            "p_value": p_value,
            "Significant": "Yes" if p_value < 0.05 else "No",
            "Cohen_d": cohens_d
        })
    
    return pd.DataFrame(comparisons)


def generate_full_report(
    results_dir: Path,
    output_dir: Path
):
    """Generate comprehensive report."""
    logger.info("Generating full report...")
    
    # Load results
    results = load_results(results_dir)
    
    if not results:
        logger.error(f"No results found in {results_dir}")
        return
    
    # Create Markdown summary
    create_markdown_summary(results, output_dir / "summary.md")
    
    # Create LaTeX tables
    if "retrieval" in results:
        # Retrieval metrics table
        retrieval_data = pd.DataFrame([results["retrieval"]])
        create_latex_table(
            retrieval_data,
            caption="Retrieval performance on NSD test set",
            label="retrieval_results",
            output_path=output_dir / "retrieval_table.tex"
        )
    
    logger.info(f"Full report generated in {output_dir}")


def generate_ablation_report(
    results_dir: Path,
    output_dir: Path
):
    """Generate ablation study report."""
    logger.info("Generating ablation report...")
    
    # Look for results.csv
    results_path = results_dir / "results.csv"
    if not results_path.exists():
        logger.error(f"No results.csv found in {results_dir}")
        return
    
    # Load results
    results_df = pd.read_csv(results_path)
    
    # Create LaTeX table
    create_latex_table(
        results_df,
        caption="Ablation study results",
        label="ablation_results",
        output_path=output_dir / "ablation_table.tex"
    )
    
    # Create performance plot
    if len(results_df.columns) > 1:
        x_col = results_df.columns[0]
        y_cols = [col for col in results_df.columns[1:] 
                  if results_df[col].dtype in [np.float64, np.float32]]
        
        if y_cols:
            create_performance_plot(
                results_df,
                x_col=x_col,
                y_cols=y_cols[:3],  # Plot up to 3 metrics
                output_path=output_dir / "ablation_plot.png",
                title="Ablation Study: Performance vs Parameter",
                xlabel=x_col,
                ylabel="Score"
            )
    
    logger.info(f"Ablation report generated in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate automated reports from evaluation results"
    )
    
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing results")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for reports")
    parser.add_argument("--report-type", type=str, default="full",
                        choices=["full", "ablation", "summary"],
                        help="Type of report to generate")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Automated Report Generation")
    logger.info("=" * 80)
    logger.info(f"Results: {results_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Type: {args.report_type}")
    
    if args.report_type == "full":
        generate_full_report(results_dir, output_dir)
    elif args.report_type == "ablation":
        generate_ablation_report(results_dir, output_dir)
    elif args.report_type == "summary":
        results = load_results(results_dir)
        create_markdown_summary(results, output_dir / "summary.md")
    
    logger.info("\n" + "=" * 80)
    logger.info("Report generation complete!")
    logger.info(f"Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
