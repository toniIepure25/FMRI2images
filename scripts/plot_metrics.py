#!/usr/bin/env python3
"""
Plot Metrics
============

Generate publication-quality figures from evaluation reports.

Creates:
- CLIPScore distributions (per subject, aggregated)
- Rank distributions (per subject, aggregated)
- R@K comparison bars
- Gallery size comparison

Usage:
    python scripts/plot_metrics.py \\
        --reports-dir outputs/reports \\
        --output-dir outputs/reports/figures \\
        --subjects subj01 subj02 subj03
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Publication-ready style
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')


def load_csv_data(reports_dir: Path, subject: str, gallery_type: str) -> pd.DataFrame:
    """
    Load per-sample CSV data.
    
    Args:
        reports_dir: Reports directory
        subject: Subject ID
        gallery_type: Gallery type (matched/test/all)
    
    Returns:
        DataFrame with per-sample metrics
    """
    csv_path = reports_dir / subject / f"eval_{gallery_type}.csv"
    
    if not csv_path.exists():
        logger.warning(f"CSV not found: {csv_path}")
        return pd.DataFrame()
    
    return pd.read_csv(csv_path)


def plot_clipscore_distributions(
    reports_dir: Path,
    subjects: List[str],
    output_dir: Path
) -> None:
    """
    Plot CLIPScore distributions for all subjects.
    
    Args:
        reports_dir: Reports directory
        subjects: List of subject IDs
        output_dir: Output directory for figures
    """
    fig, axes = plt.subplots(1, len(subjects), figsize=(5 * len(subjects), 4))
    
    if len(subjects) == 1:
        axes = [axes]
    
    all_scores = []
    
    for idx, subject in enumerate(subjects):
        ax = axes[idx]
        
        # Load matched gallery data (primary evaluation)
        df = load_csv_data(reports_dir, subject, "matched")
        
        if len(df) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(subject)
            continue
        
        scores = df['clipscore'].values
        all_scores.extend(scores)
        
        # Histogram
        ax.hist(scores, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.3f}')
        ax.set_xlabel('CLIPScore', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{subject}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('CLIPScore Distribution by Subject (Matched Gallery)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / "clipscore_by_subject.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved: {output_path}")
    
    # Aggregated plot
    if len(all_scores) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(all_scores, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.mean(all_scores), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(all_scores):.3f}')
        ax.set_xlabel('CLIPScore', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('CLIPScore Distribution (All Subjects Combined)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = output_dir / "clipscore_combined.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved: {output_path}")


def plot_rank_distributions(
    reports_dir: Path,
    subjects: List[str],
    gallery_types: List[str],
    output_dir: Path
) -> None:
    """
    Plot rank distributions for different gallery types.
    
    Args:
        reports_dir: Reports directory
        subjects: List of subject IDs
        gallery_types: List of gallery types
        output_dir: Output directory for figures
    """
    fig, axes = plt.subplots(len(gallery_types), len(subjects), 
                             figsize=(5 * len(subjects), 4 * len(gallery_types)))
    
    if len(subjects) == 1 and len(gallery_types) == 1:
        axes = np.array([[axes]])
    elif len(subjects) == 1:
        axes = axes.reshape(-1, 1)
    elif len(gallery_types) == 1:
        axes = axes.reshape(1, -1)
    
    for gal_idx, gallery_type in enumerate(gallery_types):
        for subj_idx, subject in enumerate(subjects):
            ax = axes[gal_idx, subj_idx]
            
            df = load_csv_data(reports_dir, subject, gallery_type)
            
            if len(df) == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{subject}\n{gallery_type}")
                continue
            
            ranks = df['rank'].values
            valid_ranks = ranks[ranks > 0]  # Exclude invalid ranks
            
            if len(valid_ranks) == 0:
                ax.text(0.5, 0.5, "No valid ranks", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{subject}\n{gallery_type}")
                continue
            
            # Histogram with log scale for better visibility
            bins = np.logspace(0, np.log10(max(valid_ranks) + 1), 30)
            ax.hist(valid_ranks, bins=bins, edgecolor='black', alpha=0.7, color='coral')
            ax.axvline(np.median(valid_ranks), color='blue', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(valid_ranks):.0f}')
            ax.set_xlabel('Rank (log scale)', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(f'{subject} | {gallery_type}', fontsize=11, fontweight='bold')
            ax.set_xscale('log')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, which='both')
    
    plt.suptitle('Retrieval Rank Distribution', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_path = output_dir / "rank_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved: {output_path}")


def plot_retrieval_comparison(
    reports_dir: Path,
    subjects: List[str],
    gallery_types: List[str],
    output_dir: Path
) -> None:
    """
    Plot R@K comparison across subjects and gallery types.
    
    Args:
        reports_dir: Reports directory
        subjects: List of subject IDs
        gallery_types: List of gallery types
        output_dir: Output directory for figures
    """
    # Collect data
    data = []
    
    for subject in subjects:
        for gallery_type in gallery_types:
            json_path = reports_dir / subject / f"eval_{gallery_type}.json"
            
            if not json_path.exists():
                continue
            
            with open(json_path) as f:
                report = json.load(f)
            
            retrieval = report.get("retrieval", {})
            data.append({
                "subject": subject,
                "gallery": gallery_type,
                "R@1": retrieval.get("R@1", 0),
                "R@5": retrieval.get("R@5", 0),
                "R@10": retrieval.get("R@10", 0),
            })
    
    if len(data) == 0:
        logger.warning("No data for retrieval comparison")
        return
    
    df = pd.DataFrame(data)
    
    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(subjects))
    width = 0.25
    
    for idx, gallery_type in enumerate(gallery_types):
        subset = df[df['gallery'] == gallery_type]
        offset = (idx - len(gallery_types) / 2 + 0.5) * width
        
        r1_vals = [subset[subset['subject'] == s]['R@1'].values[0] if len(subset[subset['subject'] == s]) > 0 else 0 
                   for s in subjects]
        
        ax.bar(x + offset, r1_vals, width, label=f'{gallery_type}', alpha=0.8)
    
    ax.set_xlabel('Subject', fontsize=12)
    ax.set_ylabel('R@1', fontsize=12)
    ax.set_title('Retrieval@1 by Subject and Gallery Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.legend(title='Gallery', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    output_path = output_dir / "retrieval_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved: {output_path}")


def plot_adapter_ablation(
    reports_dir: Path,
    subjects: List[str],
    output_dir: Path
) -> None:
    """
    Plot adapter ablation comparison.
    
    Args:
        reports_dir: Reports directory
        subjects: List of subject IDs
        output_dir: Output directory for figures
    """
    # Collect data
    data = []
    
    for subject in subjects:
        json_path = reports_dir / subject / "eval_matched.json"
        
        if not json_path.exists():
            continue
        
        with open(json_path) as f:
            report = json.load(f)
        
        ablations = report.get("ablations", {})
        if not ablations:
            continue
        
        with_adapter = ablations.get("with_adapter", {})
        without_adapter = ablations.get("without_adapter", {})
        
        data.append({
            "subject": subject,
            "CLIPScore (with)": with_adapter.get("clipscore", {}).get("mean", 0),
            "CLIPScore (without)": without_adapter.get("clipscore", {}).get("mean", 0),
            "R@1 (with)": with_adapter.get("retrieval", {}).get("R@1", 0),
            "R@1 (without)": without_adapter.get("retrieval", {}).get("R@1", 0),
        })
    
    if len(data) == 0:
        logger.warning("No adapter ablation data available")
        return
    
    df = pd.DataFrame(data)
    
    # Plot side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(subjects))
    width = 0.35
    
    # CLIPScore
    ax1.bar(x - width/2, df['CLIPScore (with)'], width, label='With Adapter', alpha=0.8, color='steelblue')
    ax1.bar(x + width/2, df['CLIPScore (without)'], width, label='Without Adapter', alpha=0.8, color='coral')
    ax1.set_xlabel('Subject', fontsize=12)
    ax1.set_ylabel('CLIPScore', fontsize=12)
    ax1.set_title('Adapter Ablation: CLIPScore', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(subjects)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # R@1
    ax2.bar(x - width/2, df['R@1 (with)'], width, label='With Adapter', alpha=0.8, color='steelblue')
    ax2.bar(x + width/2, df['R@1 (without)'], width, label='Without Adapter', alpha=0.8, color='coral')
    ax2.set_xlabel('Subject', fontsize=12)
    ax2.set_ylabel('R@1', fontsize=12)
    ax2.set_title('Adapter Ablation: R@1', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(subjects)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    output_path = output_dir / "adapter_ablation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation metrics")
    parser.add_argument("--reports-dir", default="outputs/reports",
                       help="Directory containing evaluation reports")
    parser.add_argument("--output-dir", default="outputs/reports/figures",
                       help="Output directory for figures")
    parser.add_argument("--subjects", nargs='+', default=["subj01", "subj02", "subj03"],
                       help="List of subjects to plot")
    args = parser.parse_args()
    
    reports_dir = Path(args.reports_dir)
    output_dir = Path(args.output_dir)
    
    if not reports_dir.exists():
        logger.error(f"Reports directory not found: {reports_dir}")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("GENERATING PUBLICATION FIGURES")
    logger.info("=" * 80)
    logger.info(f"Reports: {reports_dir}")
    logger.info(f"Output:  {output_dir}")
    logger.info(f"Subjects: {args.subjects}")
    
    gallery_types = ["matched", "test", "all"]
    
    # Generate figures
    try:
        plot_clipscore_distributions(reports_dir, args.subjects, output_dir)
        plot_rank_distributions(reports_dir, args.subjects, gallery_types, output_dir)
        plot_retrieval_comparison(reports_dir, args.subjects, gallery_types, output_dir)
        plot_adapter_ablation(reports_dir, args.subjects, output_dir)
    except Exception as e:
        logger.error(f"Failed to generate figures: {e}", exc_info=True)
        return 1
    
    logger.info("=" * 80)
    logger.info("✅ All figures generated successfully!")
    logger.info(f"📈 Output directory: {output_dir}")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
