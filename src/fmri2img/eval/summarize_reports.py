#!/usr/bin/env python3
"""
Summarize Evaluation Reports
=============================

Aggregates multiple evaluation JSON files into a summary CSV and Markdown report.

Usage:
    python scripts/summarize_reports.py \\
        --reports-dir outputs/reports \\
        --output-csv outputs/reports/summary_by_subject.csv \\
        --output-md outputs/reports/SUMMARY.md
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_report(json_path: Path) -> Dict[str, Any]:
    """
    Load evaluation JSON report.
    
    Args:
        json_path: Path to JSON file
    
    Returns:
        Dictionary with report data
    """
    with open(json_path) as f:
        return json.load(f)


def extract_metrics(report: Dict[str, Any], subject: str, gallery_type: str) -> Dict[str, Any]:
    """
    Extract key metrics from report.
    
    Args:
        report: Report dictionary
        subject: Subject ID
        gallery_type: Gallery type (matched/test/all)
    
    Returns:
        Dictionary with extracted metrics
    """
    metrics = {
        "subject": subject,
        "gallery_type": gallery_type,
        "n_samples": report.get("n_samples", 0),
        "gallery_size": report.get("gallery_size", 0),
        "retrieval_eligible": report.get("retrieval_eligible", 0),
    }
    
    # CLIPScore
    clipscore = report.get("clipscore", {})
    metrics["clipscore_mean"] = clipscore.get("mean", np.nan)
    metrics["clipscore_std"] = clipscore.get("std", np.nan)
    
    # Retrieval
    retrieval = report.get("retrieval", {})
    metrics["R@1"] = retrieval.get("R@1", np.nan)
    metrics["R@5"] = retrieval.get("R@5", np.nan)
    metrics["R@10"] = retrieval.get("R@10", np.nan)
    
    # Ranking
    ranking = report.get("ranking", {})
    metrics["mean_rank"] = ranking.get("mean_rank", np.nan)
    metrics["median_rank"] = ranking.get("median_rank", np.nan)
    metrics["mrr"] = ranking.get("mrr", np.nan)
    
    # Top-1 similarity
    metrics["top1_mean_sim"] = report.get("top1_mean_sim", np.nan)
    
    # Adapter ablation (if available)
    ablations = report.get("ablations", {})
    if ablations:
        with_adapter = ablations.get("with_adapter", {})
        without_adapter = ablations.get("without_adapter", {})
        
        metrics["ablation_clipscore_with"] = with_adapter.get("clipscore", {}).get("mean", np.nan)
        metrics["ablation_clipscore_without"] = without_adapter.get("clipscore", {}).get("mean", np.nan)
        metrics["ablation_r1_with"] = with_adapter.get("retrieval", {}).get("R@1", np.nan)
        metrics["ablation_r1_without"] = without_adapter.get("retrieval", {}).get("R@1", np.nan)
    
    return metrics


def create_markdown_table(df: pd.DataFrame) -> str:
    """
    Create Markdown table from DataFrame.
    
    Args:
        df: DataFrame with summary metrics
    
    Returns:
        Markdown table string
    """
    md = "# Evaluation Summary\n\n"
    md += "## Per-Subject Results\n\n"
    
    # Main metrics table
    md += "| Subject | Gallery | N | Gallery Size | CLIPScore | R@1 | R@5 | R@10 | Mean Rank | MRR |\n"
    md += "|---------|---------|---|--------------|-----------|-----|-----|------|-----------|-----|\n"
    
    for _, row in df.iterrows():
        md += f"| {row['subject']} | {row['gallery_type']} | {row['n_samples']:.0f} | "
        md += f"{row['gallery_size']:.0f} | {row['clipscore_mean']:.3f} | "
        md += f"{row['R@1']:.3f} | {row['R@5']:.3f} | {row['R@10']:.3f} | "
        md += f"{row['mean_rank']:.1f} | {row['mrr']:.3f} |\n"
    
    # Macro averages
    md += "\n## Macro Averages (Across Subjects)\n\n"
    
    for gallery_type in df['gallery_type'].unique():
        subset = df[df['gallery_type'] == gallery_type]
        md += f"\n### Gallery: {gallery_type}\n\n"
        md += f"- **CLIPScore**: {subset['clipscore_mean'].mean():.3f} ± {subset['clipscore_mean'].std():.3f}\n"
        md += f"- **R@1**: {subset['R@1'].mean():.3f} ± {subset['R@1'].std():.3f}\n"
        md += f"- **R@5**: {subset['R@5'].mean():.3f} ± {subset['R@5'].std():.3f}\n"
        md += f"- **R@10**: {subset['R@10'].mean():.3f} ± {subset['R@10'].std():.3f}\n"
        md += f"- **Mean Rank**: {subset['mean_rank'].mean():.1f} ± {subset['mean_rank'].std():.1f}\n"
        md += f"- **MRR**: {subset['mrr'].mean():.3f} ± {subset['mrr'].std():.3f}\n"
    
    # Adapter ablation (if available)
    if 'ablation_clipscore_with' in df.columns:
        ablation_subset = df[df['ablation_clipscore_with'].notna()]
        if len(ablation_subset) > 0:
            md += "\n## Adapter Ablation\n\n"
            md += "| Subject | CLIPScore (with) | CLIPScore (without) | Δ | R@1 (with) | R@1 (without) | Δ |\n"
            md += "|---------|------------------|---------------------|---|------------|---------------|---|\n"
            
            for _, row in ablation_subset.iterrows():
                cs_delta = row['ablation_clipscore_with'] - row['ablation_clipscore_without']
                r1_delta = row['ablation_r1_with'] - row['ablation_r1_without']
                md += f"| {row['subject']} | {row['ablation_clipscore_with']:.3f} | "
                md += f"{row['ablation_clipscore_without']:.3f} | {cs_delta:+.3f} | "
                md += f"{row['ablation_r1_with']:.3f} | {row['ablation_r1_without']:.3f} | {r1_delta:+.3f} |\n"
    
    return md


def main():
    parser = argparse.ArgumentParser(description="Summarize evaluation reports")
    parser.add_argument("--reports-dir", default="outputs/reports",
                       help="Directory containing evaluation reports")
    parser.add_argument("--output-csv", default="outputs/reports/summary_by_subject.csv",
                       help="Output CSV path")
    parser.add_argument("--output-md", default="outputs/reports/SUMMARY.md",
                       help="Output Markdown path")
    args = parser.parse_args()
    
    reports_dir = Path(args.reports_dir)
    
    if not reports_dir.exists():
        logger.error(f"Reports directory not found: {reports_dir}")
        return 1
    
    logger.info("=" * 80)
    logger.info("EVALUATION REPORT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Reports directory: {reports_dir}")
    
    # Find all evaluation JSON files
    json_files = list(reports_dir.glob("*/eval_*.json"))
    
    if len(json_files) == 0:
        logger.warning(f"No evaluation reports found in {reports_dir}")
        return 1
    
    logger.info(f"Found {len(json_files)} evaluation reports")
    
    # Extract metrics from all reports
    all_metrics = []
    
    for json_path in sorted(json_files):
        # Parse subject and gallery type from path
        subject = json_path.parent.name
        filename = json_path.stem
        
        # Extract gallery type from filename (eval_matched, eval_test, eval_all)
        if "matched" in filename:
            gallery_type = "matched"
        elif "test" in filename:
            gallery_type = "test"
        elif "all" in filename:
            gallery_type = "all"
        else:
            gallery_type = "unknown"
        
        try:
            report = load_report(json_path)
            metrics = extract_metrics(report, subject, gallery_type)
            all_metrics.append(metrics)
            logger.info(f"✅ Loaded: {subject}/{gallery_type}")
        except Exception as e:
            logger.error(f"❌ Failed to load {json_path}: {e}")
            continue
    
    if len(all_metrics) == 0:
        logger.error("No metrics extracted")
        return 1
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Sort by subject and gallery type
    df = df.sort_values(["subject", "gallery_type"])
    
    # Save CSV
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"✅ CSV saved: {output_csv}")
    
    # Create and save Markdown
    md_content = create_markdown_table(df)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, 'w') as f:
        f.write(md_content)
    logger.info(f"✅ Markdown saved: {output_md}")
    
    # Print summary to console
    logger.info("=" * 80)
    logger.info("MACRO AVERAGES (Across All Subjects)")
    logger.info("=" * 80)
    
    for gallery_type in sorted(df['gallery_type'].unique()):
        subset = df[df['gallery_type'] == gallery_type]
        logger.info(f"\nGallery: {gallery_type}")
        logger.info(f"  CLIPScore: {subset['clipscore_mean'].mean():.3f} ± {subset['clipscore_mean'].std():.3f}")
        logger.info(f"  R@1:       {subset['R@1'].mean():.3f} ± {subset['R@1'].std():.3f}")
        logger.info(f"  R@5:       {subset['R@5'].mean():.3f} ± {subset['R@5'].std():.3f}")
        logger.info(f"  R@10:      {subset['R@10'].mean():.3f} ± {subset['R@10'].std():.3f}")
        logger.info(f"  Mean Rank: {subset['mean_rank'].mean():.1f} ± {subset['mean_rank'].std():.1f}")
        logger.info(f"  MRR:       {subset['mrr'].mean():.3f} ± {subset['mrr'].std():.3f}")
    
    logger.info("=" * 80)
    logger.info("✅ Summary complete!")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
