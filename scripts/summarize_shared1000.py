#!/usr/bin/env python3
"""
Aggregate Shared1000 Results Across Subjects and Strategies
==========================================================

Creates publication-ready tables and figures from evaluation outputs.

Input Structure:
    outputs/eval_shared1000/
        subj01/
            shared1000_avg_metrics_single.json
            shared1000_avg_metrics_best_of_8.json
            shared1000_avg_metrics_boi_lite.json
        subj02/
            ...

Output:
    outputs/eval_shared1000/
        SUMMARY.csv                    # All results in table
        SUMMARY.tex                    # LaTeX table
        SUMMARY.md                     # Markdown table
        figures/
            r1_comparison.png          # R@1 bar chart
            clipscore_comparison.png   # CLIPScore bar chart
            brain_alignment.png        # Brain alignment chart
            
Usage:
    python scripts/summarize_shared1000.py \\
        --eval-dir outputs/eval_shared1000 \\
        --output-dir outputs/eval_shared1000 \\
        --subjects subj01 subj02 subj03 \\
        --strategies single best_of_8 boi_lite \\
        --rep-mode avg
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fmri2img.stats.inference import bootstrap_ci, paired_permutation_test, holm_bonferroni_correction


def load_all_metrics(
    eval_dir: Path,
    subjects: List[str],
    strategies: List[str],
    rep_mode: str
) -> Dict[str, Dict[str, Any]]:
    """
    Load all metrics files.
    
    Returns:
        {(subject, strategy): metrics_dict}
    """
    all_metrics = {}
    
    for subject in subjects:
        for strategy in strategies:
            # Try multiple possible filenames
            possible_paths = [
                eval_dir / subject / f"shared1000_{rep_mode}_metrics_{strategy}.json",
                eval_dir / subject / f"metrics_{strategy}.json",
                eval_dir / subject / strategy / "metrics.json"
            ]
            
            for path in possible_paths:
                if path.exists():
                    logger.info(f"Loading {path}")
                    with open(path) as f:
                        data = json.load(f)
                        all_metrics[(subject, strategy)] = data.get("metrics", data)
                    break
            else:
                logger.warning(f"No metrics found for {subject}/{strategy}")
    
    return all_metrics


def create_summary_table(
    all_metrics: Dict,
    metric_keys: List[str]
) -> pd.DataFrame:
    """
    Create summary table with all metrics.
    
    Args:
        all_metrics: {(subject, strategy): metrics}
        metric_keys: List of metric paths like "retrieval.R@1"
    
    Returns:
        DataFrame with subjects x strategies
    """
    records = []
    
    for (subject, strategy), metrics in all_metrics.items():
        record = {
            "subject": subject,
            "strategy": strategy
        }
        
        for key_path in metric_keys:
            # Navigate nested dict
            value = metrics
            for key in key_path.split('.'):
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break
            
            record[key_path] = value
        
        records.append(record)
    
    return pd.DataFrame(records)


def create_latex_table(df: pd.DataFrame, output_path: Path):
    """Create LaTeX table with formatting."""
    # Format numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    latex = df.to_latex(
        index=False,
        float_format="%.3f",
        column_format="l" + "c" * (len(df.columns) - 1),
        escape=False
    )
    
    with open(output_path, "w") as f:
        f.write(latex)
    
    logger.info(f"Wrote LaTeX table to {output_path}")


def create_markdown_table(df: pd.DataFrame, output_path: Path):
    """Create Markdown table."""
    with open(output_path, "w") as f:
        f.write("# Shared1000 Evaluation Summary\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")
    
    logger.info(f"Wrote Markdown table to {output_path}")


def plot_metric_comparison(
    all_metrics: Dict,
    metric_path: str,
    output_path: Path,
    ylabel: str,
    title: str,
    figsize=(12, 6)
):
    """
    Create bar chart comparing metric across subjects and strategies.
    
    Args:
        all_metrics: {(subject, strategy): metrics}
        metric_path: Dot-notation path (e.g., "retrieval.R@1")
        output_path: Where to save figure
        ylabel: Y-axis label
        title: Plot title
    """
    # Extract data
    subjects = sorted(set(subj for subj, _ in all_metrics.keys()))
    strategies = sorted(set(strat for _, strat in all_metrics.keys()))
    
    # Build matrix: subjects x strategies
    data = np.zeros((len(subjects), len(strategies)))
    
    for i, subject in enumerate(subjects):
        for j, strategy in enumerate(strategies):
            metrics = all_metrics.get((subject, strategy))
            if metrics:
                value = metrics
                for key in metric_path.split('.'):
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        value = np.nan
                        break
                data[i, j] = value if value is not None else np.nan
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(subjects))
    width = 0.8 / len(strategies)
    
    colors = sns.color_palette("husl", len(strategies))
    
    for j, strategy in enumerate(strategies):
        offset = (j - len(strategies)/2 + 0.5) * width
        bars = ax.bar(x + offset, data[:, j], width, label=strategy, color=colors[j])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
    
    ax.set_xlabel('Subject', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    ax.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved plot to {output_path}")


def statistical_tests(
    all_metrics: Dict,
    subjects: List[str],
    strategies: List[str],
    metric_path: str,
    output_path: Path
):
    """
    Perform pairwise statistical tests between strategies.
    
    Computes:
    - Paired permutation test (per subject, same test set)
    - Holm-Bonferroni correction for multiple comparisons
    - Cohen's d effect sizes
    
    Args:
        all_metrics: {(subject, strategy): metrics}
        subjects: List of subjects
        strategies: List of strategies
        metric_path: Metric to test (e.g., "retrieval.R@1")
        output_path: Where to save results JSON
    """
    from fmri2img.stats.inference import cohens_d_paired
    
    logger.info(f"Statistical tests for {metric_path}")
    
    # Extract per-subject scores for each strategy
    strategy_scores = {strat: [] for strat in strategies}
    
    for subject in subjects:
        for strategy in strategies:
            metrics = all_metrics.get((subject, strategy))
            if metrics:
                value = metrics
                for key in metric_path.split('.'):
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        value = None
                        break
                if value is not None:
                    strategy_scores[strategy].append(value)
    
    # Pairwise tests
    tests = {}
    p_values = {}
    
    for i, strat1 in enumerate(strategies):
        for j, strat2 in enumerate(strategies):
            if i < j:
                key = f"{strat1}_vs_{strat2}"
                
                scores1 = np.array(strategy_scores[strat1])
                scores2 = np.array(strategy_scores[strat2])
                
                # Paired permutation test
                p_val = paired_permutation_test(scores1, scores2, n_perm=10000, seed=42)
                
                # Effect size
                d = cohens_d_paired(scores1, scores2)
                
                tests[key] = {
                    "p_value": float(p_val),
                    "cohens_d": float(d),
                    "mean_diff": float(scores1.mean() - scores2.mean()),
                    "strategy1_mean": float(scores1.mean()),
                    "strategy2_mean": float(scores2.mean())
                }
                p_values[key] = p_val
    
    # Multiple comparison correction
    corrected = holm_bonferroni_correction(p_values, alpha=0.05)
    
    for key, (adj_p, significant) in corrected.items():
        tests[key]["adjusted_p"] = float(adj_p)
        tests[key]["significant"] = bool(significant)
    
    # Save results
    output = {
        "metric": metric_path,
        "n_subjects": len(subjects),
        "strategies": strategies,
        "pairwise_tests": tests
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Saved statistical tests to {output_path}")
    
    # Log significant differences
    for key, result in tests.items():
        if result["significant"]:
            logger.info(
                f"✓ {key}: p={result['adjusted_p']:.4f}, d={result['cohens_d']:.3f} "
                f"(SIGNIFICANT)"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Summarize Shared1000 evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--eval-dir", type=str, required=True,
                        help="Directory containing subject evaluation folders")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for summary")
    parser.add_argument("--subjects", nargs="+", required=True,
                        help="Subjects to include")
    parser.add_argument("--strategies", nargs="+", required=True,
                        help="Strategies to include")
    parser.add_argument("--rep-mode", type=str, default="avg",
                        help="Repetition mode")
    
    args = parser.parse_args()
    
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Shared1000 Summary Report")
    logger.info("=" * 80)
    logger.info(f"Eval dir: {eval_dir}")
    logger.info(f"Subjects: {args.subjects}")
    logger.info(f"Strategies: {args.strategies}")
    
    # Load all metrics
    logger.info("\nLoading metrics...")
    all_metrics = load_all_metrics(
        eval_dir,
        args.subjects,
        args.strategies,
        args.rep_mode
    )
    logger.info(f"Loaded {len(all_metrics)} result files")
    
    # Create summary table
    logger.info("\nCreating summary tables...")
    metric_keys = [
        "retrieval.R@1",
        "retrieval.R@5",
        "retrieval.R@10",
        "retrieval.mean_rank",
        "perceptual.CLIPScore",
        "brain_alignment.voxelwise_corr_mean"
    ]
    
    df = create_summary_table(all_metrics, metric_keys)
    
    # Save tables
    df.to_csv(output_dir / "SUMMARY.csv", index=False)
    logger.info(f"Saved CSV to {output_dir / 'SUMMARY.csv'}")
    
    create_latex_table(df, output_dir / "SUMMARY.tex")
    create_markdown_table(df, output_dir / "SUMMARY.md")
    
    # Create plots
    logger.info("\nCreating comparison plots...")
    
    plot_metric_comparison(
        all_metrics,
        "retrieval.R@1",
        figures_dir / "r1_comparison.png",
        ylabel="R@1 (%)",
        title="Retrieval@1 Comparison Across Subjects and Strategies"
    )
    
    plot_metric_comparison(
        all_metrics,
        "retrieval.R@5",
        figures_dir / "r5_comparison.png",
        ylabel="R@5 (%)",
        title="Retrieval@5 Comparison Across Subjects and Strategies"
    )
    
    # Statistical tests
    logger.info("\nPerforming statistical tests...")
    statistical_tests(
        all_metrics,
        args.subjects,
        args.strategies,
        "retrieval.R@1",
        output_dir / "stats_r1.json"
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("Summary Complete!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - SUMMARY.csv")
    logger.info(f"  - SUMMARY.tex")
    logger.info(f"  - SUMMARY.md")
    logger.info(f"  - figures/*.png")
    logger.info(f"  - stats_*.json")


if __name__ == "__main__":
    main()
