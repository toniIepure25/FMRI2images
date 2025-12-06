#!/usr/bin/env python3
"""
Compare multiple reconstruction evaluations with bootstrap confidence intervals.

Aggregates evaluation JSONs, computes bootstrap 95% CIs, and generates:
- CSV with all metrics and CIs
- LaTeX table for thesis
- Markdown comparison summary
- Bar plots with error bars

Usage:
    python scripts/compare_evals.py \\
        --report-dir outputs/reports/subj01 \\
        --out-csv outputs/reports/subj01/recon_compare.csv \\
        --out-tex outputs/reports/subj01/recon_compare.tex \\
        --out-md outputs/reports/subj01/recon_compare.md \\
        --out-fig outputs/reports/subj01/recon_compare.png
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import utilities
from fmri2img.eval._report_utils import (
    load_eval_json,
    guess_run_name,
    bootstrap_ci,
    format_mean_ci,
    format_mean_ci_range
)


def discover_eval_jsons(
    report_dir: Path,
    pattern: str = "recon_eval*.json"
) -> List[Path]:
    """
    Recursively discover evaluation JSON files.
    
    Args:
        report_dir: Root directory to search
        pattern: Glob pattern for JSON files
        
    Returns:
        List of paths to JSON files
    """
    if not report_dir.exists():
        return []
    
    # Recursively glob
    json_files = list(report_dir.rglob(pattern))
    
    # Sort for reproducibility
    json_files.sort()
    
    return json_files


def flatten_dict(
    d: Dict,
    parent_key: str = "",
    sep: str = "_"
) -> Dict:
    """
    Flatten nested dictionary one level deep.
    
    Converts nested dicts like {"a": {"b": 1, "c": 2}} to {"a_b": 1, "a_c": 2}.
    Handles only depth-1 nesting to avoid issues with DataFrame creation.
    
    Args:
        d: Dictionary to flatten
        parent_key: Prefix for nested keys
        sep: Separator between parent and child keys
        
    Returns:
        Flattened dictionary with all scalar values
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            # Flatten one level
            for nested_k, nested_v in v.items():
                nested_key = f"{new_key}{sep}{nested_k}"
                items.append((nested_key, nested_v))
        else:
            items.append((new_key, v))
    
    return dict(items)


def load_per_sample_csv(json_path: Path) -> Optional[pd.DataFrame]:
    """
    Load per-sample CSV if available.
    
    Looks for CSV path in JSON metadata or infers from JSON path.
    
    Args:
        json_path: Path to evaluation JSON
        
    Returns:
        DataFrame with per-sample metrics, or None if not found
    """
    # Try to load JSON to get CSV path
    try:
        data = load_eval_json(json_path)
        
        # Check if CSV path is in metadata
        # (eval_reconstruction.py doesn't store this, so we'll infer)
    except:
        pass
    
    # Infer CSV path from JSON path
    csv_path = json_path.parent / json_path.name.replace(".json", ".csv")
    
    if not csv_path.exists():
        # Try alternative naming
        csv_path = json_path.parent / "recon_eval.csv"
    
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Warning: Failed to load CSV {csv_path}: {e}")
        return None


def compute_run_metrics(
    json_path: Path,
    boots: int = 1000
) -> Dict:
    """
    Compute metrics with bootstrap CIs for a single run.
    
    Args:
        json_path: Path to evaluation JSON
        boots: Number of bootstrap resamples
        
    Returns:
        Dictionary with run metadata and metrics (point + CI)
    """
    # Load JSON
    data = load_eval_json(json_path)
    
    # Flatten nested dicts to avoid unhashable type errors in DataFrame
    data = flatten_dict(data)
    
    # Extract metadata (using flattened keys)
    run_name = guess_run_name(json_path)
    clip_space = data.get("clip_space", "unknown")
    clip_dim = data.get("clip_dim", 512)
    use_adapter = data.get("use_adapter", False)
    model_id = data.get("model_id", "N/A")
    n_samples = data.get("n_samples", 0)
    encoder = data.get("encoder", "unknown")
    steps = data.get("steps", None)
    
    # Extract aggregate metrics from flattened JSON
    # Note: nested keys are now flattened with underscores
    # e.g., "clipscore": {"mean": 0.5} -> "clipscore_mean": 0.5
    clipscore_mean = data.get("clipscore_mean", np.nan)
    clipscore_std = data.get("clipscore_std", np.nan)
    r1 = data.get("retrieval_R@1", np.nan)
    r5 = data.get("retrieval_R@5", np.nan)
    r10 = data.get("retrieval_R@10", np.nan)
    mean_rank = data.get("ranking_mean_rank", np.nan)
    mrr = data.get("ranking_mrr", np.nan)
    
    # Try to load per-sample data for bootstrap
    csv_df = load_per_sample_csv(json_path)
    
    result = {
        "run_name": run_name,
        "json_path": str(json_path),
        "encoder": encoder,
        "use_adapter": use_adapter,
        "clip_space": clip_space,
        "clip_dim": clip_dim,
        "model_id": model_id,
        "n_samples": n_samples,
        "steps": steps if steps else "N/A",
        
        # Point estimates
        "clipscore_mean": clipscore_mean,
        "clipscore_std": clipscore_std,
        "r1": r1,
        "r5": r5,
        "r10": r10,
        "mean_rank": mean_rank,
        "mrr": mrr,
    }
    
    # Add flattened gallery metadata if present
    # e.g., retrieval_gallery_type, retrieval_gallery_size, etc.
    for key in data:
        if key.startswith("retrieval_gallery_") or key.startswith("adapter_ablation_"):
            result[key] = data[key]
    
    # Bootstrap CIs if per-sample data available
    if csv_df is not None and len(csv_df) > 0:
        print(f"  Bootstrapping {run_name} with n={len(csv_df)}...")
        
        # CLIPScore CI
        if "clipscore" in csv_df.columns:
            cs_values = csv_df["clipscore"].values
            cs_low, cs_high = bootstrap_ci(cs_values, boots=boots)
            result["clipscore_ci_low"] = cs_low
            result["clipscore_ci_high"] = cs_high
        else:
            result["clipscore_ci_low"] = np.nan
            result["clipscore_ci_high"] = np.nan
        
        # R@1 CI (per-sample binary success)
        if "r@1" in csv_df.columns:
            r1_values = csv_df["r@1"].values
            r1_low, r1_high = bootstrap_ci(r1_values, boots=boots)
            result["r1_ci_low"] = r1_low
            result["r1_ci_high"] = r1_high
        else:
            result["r1_ci_low"] = np.nan
            result["r1_ci_high"] = np.nan
        
        # R@5 CI
        if "r@5" in csv_df.columns:
            r5_values = csv_df["r@5"].values
            r5_low, r5_high = bootstrap_ci(r5_values, boots=boots)
            result["r5_ci_low"] = r5_low
            result["r5_ci_high"] = r5_high
        else:
            result["r5_ci_low"] = np.nan
            result["r5_ci_high"] = np.nan
        
        # R@10 CI
        if "r@10" in csv_df.columns:
            r10_values = csv_df["r@10"].values
            r10_low, r10_high = bootstrap_ci(r10_values, boots=boots)
            result["r10_ci_low"] = r10_low
            result["r10_ci_high"] = r10_high
        else:
            result["r10_ci_low"] = np.nan
            result["r10_ci_high"] = np.nan
        
        # MRR CI (compute from ranks if available)
        if "rank" in csv_df.columns:
            ranks = csv_df["rank"].values
            mrr_values = 1.0 / ranks
            mrr_low, mrr_high = bootstrap_ci(mrr_values, boots=boots)
            result["mrr_ci_low"] = mrr_low
            result["mrr_ci_high"] = mrr_high
        else:
            result["mrr_ci_low"] = np.nan
            result["mrr_ci_high"] = np.nan
        
    else:
        # No per-sample data - use std as proxy (not bootstrap)
        print(f"  No per-sample CSV for {run_name}, using point estimates only")
        
        # Use ±std as rough CI (not bootstrap)
        result["clipscore_ci_low"] = clipscore_mean - clipscore_std
        result["clipscore_ci_high"] = clipscore_mean + clipscore_std
        
        # No CIs for other metrics without per-sample data
        result["r1_ci_low"] = np.nan
        result["r1_ci_high"] = np.nan
        result["r5_ci_low"] = np.nan
        result["r5_ci_high"] = np.nan
        result["r10_ci_low"] = np.nan
        result["r10_ci_high"] = np.nan
        result["mrr_ci_low"] = np.nan
        result["mrr_ci_high"] = np.nan
    
    return result


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize DataFrame to prevent unhashable type errors.
    
    Converts dict/list columns to stable string representations.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Sanitized DataFrame with all hashable values
    """
    df = df.copy()
    
    for col in df.columns:
        # Check if column contains dicts or lists
        sample_val = df[col].iloc[0] if len(df) > 0 else None
        
        if isinstance(sample_val, (dict, list)):
            # Convert to stable JSON string
            df[col] = df[col].apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, (dict, list)) else x)
            print(f"  Sanitized column '{col}' (dict/list → JSON string)")
    
    return df


def create_comparison_dataframe(
    run_metrics: List[Dict]
) -> pd.DataFrame:
    """
    Create tidy DataFrame from run metrics.
    
    Args:
        run_metrics: List of metric dictionaries
        
    Returns:
        Pandas DataFrame with one row per run
    """
    df = pd.DataFrame(run_metrics)
    
    # Sanitize to prevent unhashable type errors
    df = sanitize_dataframe(df)
    
    # Sort by: adapter (desc), clip_dim (desc), R@1 (desc)
    # Check if sort columns exist
    sort_cols = []
    sort_orders = []
    
    if "use_adapter" in df.columns:
        sort_cols.append("use_adapter")
        sort_orders.append(False)
    
    if "clip_dim" in df.columns:
        sort_cols.append("clip_dim")
        sort_orders.append(False)
    
    if "r1" in df.columns:
        sort_cols.append("r1")
        sort_orders.append(False)
    
    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=sort_orders)
    else:
        print("  Warning: No sort columns found, keeping original order")
    
    return df


def write_csv(df: pd.DataFrame, out_path: Path) -> None:
    """Write comparison DataFrame to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✓ CSV written: {out_path}")


def write_latex_table(df: pd.DataFrame, out_path: Path) -> None:
    """
    Write LaTeX table for thesis.
    
    Columns: Run, CLIP Space, n, CLIPScore, R@1, R@5, R@10, MRR
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        # Table header
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Reconstruction Evaluation Comparison with 95\\% Bootstrap Confidence Intervals}\n")
        f.write("\\label{tab:recon_comparison}\n")
        f.write("\\begin{tabular}{lcccccccc}\n")
        f.write("\\hline\n")
        f.write("Run & CLIP Space & n & CLIPScore & R@1 & R@5 & R@10 & MRR \\\\\n")
        f.write("\\hline\n")
        
        # Table rows
        for _, row in df.iterrows():
            run_name = row["run_name"].replace("_", "\\_")
            clip_space = row["clip_space"].replace("-D", "D")
            n = int(row["n_samples"])
            
            # Format metrics with CIs
            cs = format_mean_ci(
                row["clipscore_mean"],
                row["clipscore_ci_low"],
                row["clipscore_ci_high"]
            )
            
            r1 = format_mean_ci(
                row["r1"],
                row["r1_ci_low"],
                row["r1_ci_high"]
            )
            
            r5 = format_mean_ci(
                row["r5"],
                row["r5_ci_low"],
                row["r5_ci_high"]
            )
            
            r10 = format_mean_ci(
                row["r10"],
                row["r10_ci_low"],
                row["r10_ci_high"]
            )
            
            mrr = format_mean_ci(
                row["mrr"],
                row["mrr_ci_low"],
                row["mrr_ci_high"]
            )
            
            f.write(f"{run_name} & {clip_space} & {n} & {cs} & {r1} & {r5} & {r10} & {mrr} \\\\\n")
        
        # Table footer
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✓ LaTeX table written: {out_path}")


def write_markdown_summary(df: pd.DataFrame, out_path: Path) -> None:
    """
    Write Markdown comparison summary for thesis.
    
    Includes:
    - Bullet list of runs
    - Metrics table
    - Interpretation paragraph
    - Space consistency footnote
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        f.write("# Reconstruction Evaluation Comparison\n\n")
        
        # Run list
        f.write("## Evaluated Runs\n\n")
        for _, row in df.iterrows():
            adapter_status = "with adapter" if row["use_adapter"] else "no adapter"
            f.write(f"- **{row['run_name']}**: {row['clip_space']}, {adapter_status}, ")
            f.write(f"encoder={row['encoder']}, n={int(row['n_samples'])}")
            if row['steps'] != "N/A":
                f.write(f", steps={row['steps']}")
            f.write("\n")
        
        f.write("\n---\n\n")
        
        # Metrics table
        f.write("## Metrics with 95% Bootstrap Confidence Intervals\n\n")
        f.write("| Run | CLIP Space | n | CLIPScore | R@1 | R@5 | R@10 | MRR |\n")
        f.write("|-----|------------|---|-----------|-----|-----|------|-----|\n")
        
        for _, row in df.iterrows():
            cs = format_mean_ci(
                row["clipscore_mean"],
                row["clipscore_ci_low"],
                row["clipscore_ci_high"]
            )
            r1 = format_mean_ci(row["r1"], row["r1_ci_low"], row["r1_ci_high"])
            r5 = format_mean_ci(row["r5"], row["r5_ci_low"], row["r5_ci_high"])
            r10 = format_mean_ci(row["r10"], row["r10_ci_low"], row["r10_ci_high"])
            mrr = format_mean_ci(row["mrr"], row["mrr_ci_low"], row["mrr_ci_high"])
            
            f.write(f"| {row['run_name']} | {row['clip_space']} | {int(row['n_samples'])} | ")
            f.write(f"{cs} | {r1} | {r5} | {r10} | {mrr} |\n")
        
        f.write("\n---\n\n")
        
        # Interpretation
        f.write("## Interpretation\n\n")
        
        # Find best runs
        best_r1_idx = df["r1"].idxmax()
        best_cs_idx = df["clipscore_mean"].idxmax()
        
        best_r1_run = df.loc[best_r1_idx]
        best_cs_run = df.loc[best_cs_idx]
        
        f.write(f"**Best R@1:** {best_r1_run['run_name']} ({best_r1_run['r1']:.3f}) ")
        f.write(f"— {best_r1_run['clip_space']}, ")
        f.write(f"{'with adapter' if best_r1_run['use_adapter'] else 'no adapter'}. ")
        
        f.write(f"**Best CLIPScore:** {best_cs_run['run_name']} ({best_cs_run['clipscore_mean']:.3f}) ")
        f.write(f"— {best_cs_run['clip_space']}, ")
        f.write(f"{'with adapter' if best_cs_run['use_adapter'] else 'no adapter'}. ")
        
        # Adapter analysis
        adapter_runs = df[df["use_adapter"] == True]
        no_adapter_runs = df[df["use_adapter"] == False]
        
        if len(adapter_runs) > 0 and len(no_adapter_runs) > 0:
            adapter_mean_r1 = adapter_runs["r1"].mean()
            no_adapter_mean_r1 = no_adapter_runs["r1"].mean()
            
            if adapter_mean_r1 > no_adapter_mean_r1:
                improvement = ((adapter_mean_r1 - no_adapter_mean_r1) / no_adapter_mean_r1) * 100
                f.write(f"Using the CLIP adapter in target space improved average R@1 by {improvement:.1f}% ")
                f.write(f"({no_adapter_mean_r1:.3f} → {adapter_mean_r1:.3f}). ")
            else:
                f.write("The adapter did not improve average R@1 compared to the 512-D baseline. ")
        
        f.write("\n\n")
        
        # Footnote
        f.write("---\n\n")
        f.write("**Note:** Evaluation CLIP space matches generation space where adapter was used. ")
        f.write("Comparisons across different CLIP dimensions should be interpreted cautiously, ")
        f.write("as they represent different semantic spaces.\n\n")
        
        f.write("**Confidence Intervals:** 95% bootstrap CIs computed from per-sample metrics ")
        f.write("using 1000 resamples with replacement.\n")
    
    print(f"✓ Markdown summary written: {out_path}")


def create_comparison_plots(df: pd.DataFrame, out_path: Path) -> None:
    """
    Create bar plots comparing runs.
    
    Two panels (stacked vertically):
    - Panel A: CLIPScore with error bars
    - Panel B: R@1 with error bars
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure with 2 subplots (vertical stack)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Prepare data
    run_names = df["run_name"].values
    x_pos = np.arange(len(run_names))
    
    # Panel A: CLIPScore
    ax = axes[0]
    cs_means = df["clipscore_mean"].values
    cs_lows = df["clipscore_ci_low"].values
    cs_highs = df["clipscore_ci_high"].values
    
    # Compute error bar values (distance from mean)
    cs_err_low = np.maximum(0, cs_means - cs_lows)  # Ensure non-negative
    cs_err_high = np.maximum(0, cs_highs - cs_means)
    cs_errors = np.array([cs_err_low, cs_err_high])
    
    ax.bar(x_pos, cs_means, alpha=0.7)
    ax.errorbar(x_pos, cs_means, yerr=cs_errors, fmt='none', 
                ecolor='black', capsize=5, capthick=2)
    ax.set_ylabel("CLIPScore", fontsize=12)
    ax.set_title("A. CLIPScore Comparison", fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(run_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Panel B: R@1
    ax = axes[1]
    r1_means = df["r1"].values
    r1_lows = df["r1_ci_low"].values
    r1_highs = df["r1_ci_high"].values
    
    # Compute error bar values
    r1_err_low = np.maximum(0, r1_means - r1_lows)  # Ensure non-negative
    r1_err_high = np.maximum(0, r1_highs - r1_means)
    r1_errors = np.array([r1_err_low, r1_err_high])
    
    ax.bar(x_pos, r1_means, alpha=0.7)
    ax.errorbar(x_pos, r1_means, yerr=r1_errors, fmt='none',
                ecolor='black', capsize=5, capthick=2)
    ax.set_ylabel("R@1 (Proportion)", fontsize=12)
    ax.set_title("B. Retrieval@1 Comparison", fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(run_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0, top=1.0)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plots saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--report-dir", type=Path, required=True,
                        help="Root directory to scan for evaluation JSONs")
    parser.add_argument("--pattern", type=str, default="recon_eval*.json",
                        help="Glob pattern for JSON files (default: recon_eval*.json)")
    parser.add_argument("--out-csv", type=Path, required=True,
                        help="Output CSV path")
    parser.add_argument("--out-tex", type=Path, required=True,
                        help="Output LaTeX table path")
    parser.add_argument("--out-md", type=Path, required=True,
                        help="Output Markdown summary path")
    parser.add_argument("--out-fig", type=Path, required=True,
                        help="Output figure path (PNG)")
    parser.add_argument("--metrics", type=str,
                        default="clipscore_mean,R@1,R@5,R@10",
                        help="Comma-separated list of metrics to include")
    parser.add_argument("--boots", type=int, default=1000,
                        help="Number of bootstrap resamples (default: 1000)")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("  Comparing Reconstruction Evaluations")
    print("="*80 + "\n")
    
    # Discover JSONs
    print(f"Scanning: {args.report_dir}")
    print(f"Pattern: {args.pattern}")
    
    json_files = discover_eval_jsons(args.report_dir, args.pattern)
    
    if len(json_files) == 0:
        print(f"\nERROR: No JSON files found matching '{args.pattern}' in {args.report_dir}")
        return 1
    
    print(f"Found {len(json_files)} evaluation JSON(s):\n")
    for json_file in json_files:
        print(f"  - {json_file}")
    print()
    
    # Compute metrics for each run
    print("Computing metrics with bootstrap CIs...\n")
    run_metrics = []
    
    for json_file in json_files:
        try:
            metrics = compute_run_metrics(json_file, boots=args.boots)
            run_metrics.append(metrics)
        except Exception as e:
            print(f"ERROR processing {json_file}: {e}")
            continue
    
    if len(run_metrics) == 0:
        print("\nERROR: No valid runs found")
        return 1
    
    print()
    
    # Create comparison DataFrame
    df = create_comparison_dataframe(run_metrics)
    
    print(f"Aggregated {len(df)} run(s)\n")
    
    # Write outputs
    write_csv(df, args.out_csv)
    write_latex_table(df, args.out_tex)
    write_markdown_summary(df, args.out_md)
    create_comparison_plots(df, args.out_fig)
    
    print("\n" + "="*80)
    print("  Comparison Complete!")
    print("="*80 + "\n")
    print(f"📄 CSV:      {args.out_csv}")
    print(f"📄 LaTeX:    {args.out_tex}")
    print(f"📄 Markdown: {args.out_md}")
    print(f"📊 Figure:   {args.out_fig}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
