"""
Shared1000 Evaluation I/O Utilities
==================================

Standardized output formats for NSD Shared 1000 evaluation results.
Ensures consistent schema across different evaluation runs and strategies.

Output Structure:
    outputs/eval_shared1000/subj01/
        manifest.json                              # Run metadata
        shared1000_avg_metrics_single.json        # Aggregate metrics
        shared1000_avg_per_sample_single.csv      # Per-sample results
        shared1000_avg_grid_single.png            # Visualization
        shared1000_avg_summary.md                 # Human-readable report

JSON Schema (metrics):
    {
        "subject": "subj01",
        "strategy": "single",
        "rep_mode": "avg",
        "n_samples": 1000,
        "clip_dim": 512,
        "retrieval": {"R@1": 0.45, "R@5": 0.72, ...},
        "perceptual": {"clipscore_mean": 0.68, "ssim_mean": 0.45, ...},
        "brain_alignment": {"voxelwise_corr_mean": 0.32, ...},
        "repeat_consistency": {"mean": 0.85, ...},  # If rep_mode=all
        "ceiling_normalized": {...}  # If ceiling available
    }

CSV Schema (per-sample):
    nsdId, clip_dim, rank, r@1, r@5, r@10, clipscore, ssim, lpips, ...
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def write_metrics_json(
    output_path: Union[str, Path],
    metrics: Dict[str, Any],
    subject: str,
    strategy: str,
    rep_mode: str,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Write aggregate metrics to JSON file.
    
    Args:
        output_path: Path to output JSON file
        metrics: Dictionary with metric categories:
            - retrieval: {R@1, R@5, R@10, mean_rank, ...}
            - perceptual: {clipscore_mean, ssim_mean, lpips_mean, ...}
            - brain_alignment: {voxelwise_corr_mean, ...}
            - repeat_consistency: {...} (if applicable)
            - ceiling_normalized: {...} (if applicable)
        subject: Subject ID (e.g., "subj01")
        strategy: Generation strategy (e.g., "single", "best_of_8")
        rep_mode: Repetition mode (e.g., "avg", "rep1", "all")
        additional_info: Additional metadata
    
    Example:
        >>> metrics = {
        ...     "retrieval": {"R@1": 0.45, "R@5": 0.72, "R@10": 0.84},
        ...     "perceptual": {"clipscore_mean": 0.68, "ssim_mean": 0.45}
        ... }
        >>> write_metrics_json(
        ...     "outputs/eval_shared1000/subj01/metrics.json",
        ...     metrics,
        ...     subject="subj01",
        ...     strategy="single",
        ...     rep_mode="avg"
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        "evaluation_type": "shared1000",
        "created": datetime.now().isoformat(),
        "subject": subject,
        "strategy": strategy,
        "rep_mode": rep_mode,
        "metrics": metrics
    }
    
    if additional_info:
        output["additional_info"] = additional_info
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    logger.info(f"Wrote metrics to {output_path}")


def write_per_sample_csv(
    output_path: Union[str, Path],
    per_sample_data: pd.DataFrame,
    required_columns: Optional[List[str]] = None
) -> None:
    """
    Write per-sample evaluation results to CSV.
    
    Args:
        output_path: Path to output CSV file
        per_sample_data: DataFrame with per-sample results
                        Must include: nsdId
                        Optional: clip_dim, rank, r@1, r@5, r@10, 
                                 clipscore, ssim, lpips, etc.
        required_columns: List of required column names to validate
    
    Example:
        >>> per_sample = pd.DataFrame({
        ...     "nsdId": [0, 1, 2],
        ...     "clip_dim": [512, 512, 512],
        ...     "rank": [1, 5, 3],
        ...     "r@1": [1, 0, 0],
        ...     "r@5": [1, 1, 1],
        ...     "clipscore": [0.68, 0.72, 0.65]
        ... })
        >>> write_per_sample_csv("outputs/per_sample.csv", per_sample)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate required columns
    if required_columns is None:
        required_columns = ["nsdId"]
    
    missing = set(required_columns) - set(per_sample_data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Write CSV
    per_sample_data.to_csv(output_path, index=False)
    
    logger.info(
        f"Wrote per-sample results to {output_path} "
        f"({len(per_sample_data)} samples, {len(per_sample_data.columns)} columns)"
    )


def write_summary_markdown(
    output_path: Union[str, Path],
    metrics: Dict[str, Any],
    subject: str,
    strategy: str,
    rep_mode: str,
    per_sample_csv_path: Optional[Path] = None,
    grid_image_path: Optional[Path] = None
) -> None:
    """
    Write human-readable Markdown summary of evaluation.
    
    Args:
        output_path: Path to output .md file
        metrics: Metrics dictionary (same as write_metrics_json)
        subject: Subject ID
        strategy: Generation strategy
        rep_mode: Repetition mode
        per_sample_csv_path: Path to per-sample CSV (for linking)
        grid_image_path: Path to visualization grid (for embedding)
    
    Example:
        >>> write_summary_markdown(
        ...     "outputs/summary.md",
        ...     metrics=metrics,
        ...     subject="subj01",
        ...     strategy="single",
        ...     rep_mode="avg"
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        # Header
        f.write(f"# Shared1000 Evaluation Summary\n\n")
        f.write(f"**Subject:** {subject}  \n")
        f.write(f"**Strategy:** {strategy}  \n")
        f.write(f"**Repetition Mode:** {rep_mode}  \n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
        
        # Retrieval metrics
        if "retrieval" in metrics:
            f.write("## Retrieval Metrics\n\n")
            ret = metrics["retrieval"]
            f.write(f"- **R@1:** {ret.get('R@1', 'N/A'):.4f}\n")
            f.write(f"- **R@5:** {ret.get('R@5', 'N/A'):.4f}\n")
            f.write(f"- **R@10:** {ret.get('R@10', 'N/A'):.4f}\n")
            f.write(f"- **Mean Rank:** {ret.get('mean_rank', 'N/A'):.2f}\n")
            f.write(f"- **Median Rank:** {ret.get('median_rank', 'N/A'):.2f}\n\n")
        
        # Perceptual metrics
        if "perceptual" in metrics:
            f.write("## Perceptual Metrics\n\n")
            perc = metrics["perceptual"]
            if "clipscore_mean" in perc:
                f.write(f"- **CLIPScore:** {perc['clipscore_mean']:.4f} ± {perc.get('clipscore_std', 0):.4f}\n")
            if "ssim_mean" in perc:
                f.write(f"- **SSIM:** {perc['ssim_mean']:.4f} ± {perc.get('ssim_std', 0):.4f}\n")
            if "lpips_mean" in perc:
                f.write(f"- **LPIPS:** {perc['lpips_mean']:.4f} ± {perc.get('lpips_std', 0):.4f}\n")
            f.write("\n")
        
        # Brain alignment
        if "brain_alignment" in metrics:
            f.write("## Brain Alignment\n\n")
            brain = metrics["brain_alignment"]
            f.write(f"- **Voxel-wise Correlation:** {brain.get('voxelwise_corr_mean', 'N/A'):.4f}\n")
            f.write(f"- **Subject-level Correlation:** {brain.get('subject_level_corr', 'N/A'):.4f}\n")
            
            if "voxelwise_corr_mean_normalized" in brain and brain["voxelwise_corr_mean_normalized"] is not None:
                f.write(f"- **Ceiling-Normalized:** {brain['voxelwise_corr_mean_normalized']:.4f}\n")
                f.write(f"- **ROI Ceiling:** {brain.get('roi_ceiling', 'N/A'):.4f}\n")
            f.write("\n")
        
        # Repeat consistency
        if "repeat_consistency" in metrics:
            f.write("## Repeat Consistency\n\n")
            rep = metrics["repeat_consistency"]
            f.write(f"- **Mean Consistency:** {rep.get('mean', 'N/A'):.4f} ± {rep.get('std', 0):.4f}\n")
            f.write(f"- **Number of Repetitions:** {rep.get('n_reps', 'N/A')}\n\n")
        
        # Links to detailed results
        f.write("## Detailed Results\n\n")
        if per_sample_csv_path:
            rel_path = per_sample_csv_path.name
            f.write(f"- [Per-sample results]({rel_path})\n")
        if grid_image_path:
            rel_path = grid_image_path.name
            f.write(f"- [Visualization grid]({rel_path})\n\n")
            # Embed image if path provided
            f.write(f"![Reconstruction Grid]({rel_path})\n")
    
    logger.info(f"Wrote summary to {output_path}")


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path],
    metric_key: str = "retrieval.R@1",
    title: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot comparison of a metric across strategies.
    
    Args:
        metrics_dict: Dictionary mapping strategy names to metrics
                     E.g., {"single": {...}, "best_of_8": {...}}
        output_path: Path to save figure
        metric_key: Dot-separated key path (e.g., "retrieval.R@1")
        title: Plot title (default: auto-generated)
        figsize: Figure size
    
    Example:
        >>> metrics = {
        ...     "single": {"retrieval": {"R@1": 0.45}},
        ...     "best_of_8": {"retrieval": {"R@1": 0.52}},
        ...     "boi_lite": {"retrieval": {"R@1": 0.58}}
        ... }
        >>> plot_metrics_comparison(
        ...     metrics,
        ...     "outputs/r1_comparison.png",
        ...     metric_key="retrieval.R@1"
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract metric values
    strategies = []
    values = []
    
    for strategy, metrics in metrics_dict.items():
        # Navigate nested dict using dot notation
        value = metrics
        for key in metric_key.split('.'):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                value = None
                break
        
        if value is not None:
            strategies.append(strategy)
            values.append(value)
    
    if not strategies:
        logger.warning(f"No data found for metric {metric_key}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(strategies))
    bars = ax.bar(x_pos, values, color=sns.color_palette("husl", len(strategies)))
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom'
        )
    
    ax.set_xlabel('Strategy', fontsize=12)
    ax.set_ylabel(metric_key.split('.')[-1], fontsize=12)
    ax.set_title(title or f'Comparison: {metric_key}', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison plot to {output_path}")


def load_metrics_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load metrics from JSON file.
    
    Args:
        json_path: Path to metrics JSON
    
    Returns:
        Metrics dictionary
    """
    with open(json_path) as f:
        data = json.load(f)
    
    return data.get("metrics", data)


def aggregate_metrics_across_subjects(
    metrics_files: List[Path],
    output_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Aggregate metrics across multiple subjects.
    
    Args:
        metrics_files: List of paths to metrics JSON files
        output_path: Path to save aggregate CSV
    
    Returns:
        DataFrame with aggregated metrics
    
    Example:
        >>> files = [
        ...     Path("outputs/eval_shared1000/subj01/metrics_single.json"),
        ...     Path("outputs/eval_shared1000/subj02/metrics_single.json"),
        ... ]
        >>> df = aggregate_metrics_across_subjects(files, "outputs/aggregate.csv")
        >>> print(df[["subject", "R@1", "clipscore_mean"]])
    """
    records = []
    
    for file in metrics_files:
        try:
            with open(file) as f:
                data = json.load(f)
            
            record = {
                "subject": data.get("subject"),
                "strategy": data.get("strategy"),
                "rep_mode": data.get("rep_mode")
            }
            
            metrics = data.get("metrics", {})
            
            # Flatten retrieval metrics
            if "retrieval" in metrics:
                for key, val in metrics["retrieval"].items():
                    record[f"retrieval_{key}"] = val
            
            # Flatten perceptual metrics
            if "perceptual" in metrics:
                for key, val in metrics["perceptual"].items():
                    record[f"perceptual_{key}"] = val
            
            # Flatten brain alignment
            if "brain_alignment" in metrics:
                for key, val in metrics["brain_alignment"].items():
                    record[f"brain_{key}"] = val
            
            # Repeat consistency
            if "repeat_consistency" in metrics:
                for key, val in metrics["repeat_consistency"].items():
                    record[f"repeat_{key}"] = val
            
            records.append(record)
            
        except Exception as e:
            logger.warning(f"Failed to load {file}: {e}")
    
    df = pd.DataFrame(records)
    
    if len(df) > 0:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Aggregated {len(df)} results to {output_path}")
    
    return df
