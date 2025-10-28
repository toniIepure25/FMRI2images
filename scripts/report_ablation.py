#!/usr/bin/env python3
"""
Ablation Study Reporter
=======================

Generates summary report and visualizations from mixed-model ablation results
(Ridge vs MLP across reliability × PCA dimensionality grid).

Outputs:
- Markdown summary with top-10 table and best settings per model
- Line plots: test_cosine vs k_eff for each reliability threshold (per model)
- Heatmaps: test_cosine across reliability × k_eff grid (per model)

Usage:
    # Generate report for subj01
    python scripts/report_ablation.py --subject subj01
    
    # Custom paths
    python scripts/report_ablation.py \\
        --csv outputs/reports/subj01/ablation_ridge.csv \\
        --out outputs/reports/subj01/ablation_summary.md \\
        --fig-dir outputs/reports/subj01/figs
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_ablation_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load ablation CSV with backward compatibility for model column.
    
    If 'model' column is missing, treat all rows as "Ridge" for backward compatibility.
    
    Args:
        csv_path: Path to ablation CSV
    
    Returns:
        DataFrame with guaranteed 'model' column
    """
    df = pd.read_csv(csv_path)
    
    # Backward compatibility: add model column if missing
    if "model" not in df.columns:
        logger.warning("No 'model' column found; treating all rows as 'Ridge'")
        df["model"] = "Ridge"
    
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    logger.info(f"Models found: {df['model'].unique()}")
    
    return df


def generate_summary_md(df: pd.DataFrame, out_path: Path) -> None:
    """
    Generate markdown summary with top-10 table and best settings per model.
    
    Args:
        df: Ablation results DataFrame
        out_path: Output markdown file path
    """
    logger.info("Generating markdown summary...")
    
    # Sort by test_cosine (descending)
    df_sorted = df.sort_values("test_cosine", ascending=False).reset_index(drop=True)
    
    # Top 10 rows
    top10 = df_sorted.head(10)
    
    # Best settings per model
    best_per_model = {}
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        if len(model_df) > 0 and not model_df["test_cosine"].isna().all():
            best_idx = model_df["test_cosine"].idxmax()
            best_per_model[model] = df.loc[best_idx]
    
    # Write markdown
    with open(out_path, "w") as f:
        f.write("# Ablation Study Summary\n\n")
        f.write(f"**Total experiments**: {len(df)}\n\n")
        f.write(f"**Models**: {', '.join(df['model'].unique())}\n\n")
        
        # Top 10 table
        f.write("## Top 10 Settings by Test Cosine\n\n")
        f.write("| Rank | Model | Rel | k_eff | Best α | Val Cos | Test Cos | R@1 | R@5 | R@10 | Train/Val/Test |\n")
        f.write("|------|-------|-----|-------|--------|---------|----------|-----|-----|------|----------------|\n")
        
        for i, row in top10.iterrows():
            alpha_str = f"{row['best_alpha']:.1f}" if pd.notna(row['best_alpha']) else "N/A"
            f.write(
                f"| {i+1} | {row['model']} | {row['rel_threshold']:.2f} | "
                f"{int(row['k_eff']) if pd.notna(row['k_eff']) else 'N/A'} | {alpha_str} | "
                f"{row['val_cosine']:.4f} | {row['test_cosine']:.4f} | "
                f"{row['R@1']:.3f} | {row['R@5']:.3f} | {row['R@10']:.3f} | "
                f"{int(row['n_train'])}/{int(row['n_val'])}/{int(row['n_test'])} |\n"
            )
        
        f.write("\n")
        
        # Best settings per model
        f.write("## Best Settings Per Model\n\n")
        for model, row in best_per_model.items():
            f.write(f"### {model}\n\n")
            f.write(f"- **Reliability threshold**: {row['rel_threshold']:.3f}\n")
            f.write(f"- **PCA components (k_eff)**: {int(row['k_eff']) if pd.notna(row['k_eff']) else 'N/A'}\n")
            if model == "Ridge" and pd.notna(row['best_alpha']):
                f.write(f"- **Best α**: {row['best_alpha']:.1f}\n")
            f.write(f"- **Val cosine**: {row['val_cosine']:.4f}\n")
            f.write(f"- **Test cosine**: {row['test_cosine']:.4f}\n")
            f.write(f"- **Test R@1**: {row['R@1']:.3f}\n")
            f.write(f"- **Test R@5**: {row['R@5']:.3f}\n")
            f.write(f"- **Test R@10**: {row['R@10']:.3f}\n")
            f.write(f"- **Checkpoint**: `{row['checkpoint']}`\n\n")
        
        # Tie notes
        f.write("## Notes\n\n")
        
        # Check for ties in top performers
        if len(top10) >= 2:
            top_cosine = top10.iloc[0]["test_cosine"]
            ties = top10[np.isclose(top10["test_cosine"], top_cosine, atol=1e-4)]
            if len(ties) > 1:
                f.write(f"- **Top performers tied**: {len(ties)} settings within 0.0001 cosine of best.\n")
        
        # Comparison between models
        if len(best_per_model) > 1:
            cosines = {m: r["test_cosine"] for m, r in best_per_model.items()}
            best_model = max(cosines, key=cosines.get)
            f.write(f"- **Best model overall**: {best_model} (test_cosine={cosines[best_model]:.4f})\n")
    
    logger.info(f"✅ Summary saved to {out_path}")


def plot_test_cosine_vs_k(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Generate line plots: test_cosine vs k_eff for each reliability threshold.
    
    Creates one figure per model.
    
    Args:
        df: Ablation results DataFrame
        fig_dir: Output directory for figures
    """
    logger.info("Generating test_cosine vs k_eff line plots...")
    
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    for model in df["model"].unique():
        model_df = df[df["model"] == model].copy()
        
        if len(model_df) == 0:
            continue
        
        # Filter out NaN test_cosine
        model_df = model_df[model_df["test_cosine"].notna()]
        
        if len(model_df) == 0:
            logger.warning(f"No valid data for {model}, skipping plot")
            continue
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot one line per reliability threshold
        for rel in sorted(model_df["rel_threshold"].unique()):
            rel_df = model_df[model_df["rel_threshold"] == rel].sort_values("k_eff")
            if len(rel_df) > 0:
                plt.plot(
                    rel_df["k_eff"],
                    rel_df["test_cosine"],
                    marker="o",
                    label=f"rel={rel:.2f}",
                    linewidth=2
                )
        
        plt.xlabel("PCA Components (k_eff)", fontsize=12)
        plt.ylabel("Test Cosine Similarity", fontsize=12)
        plt.title(f"{model}: Test Cosine vs PCA Dimensionality", fontsize=14, fontweight="bold")
        plt.legend(title="Reliability", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        fig_path = fig_dir / f"test_cosine_vs_k_by_rel_{model}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info(f"✅ Saved {fig_path}")


def plot_heatmap(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Generate heatmaps: test_cosine across reliability × k_eff grid.
    
    Creates one heatmap per model.
    
    Args:
        df: Ablation results DataFrame
        fig_dir: Output directory for figures
    """
    logger.info("Generating test_cosine heatmaps...")
    
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    for model in df["model"].unique():
        model_df = df[df["model"] == model].copy()
        
        if len(model_df) == 0:
            continue
        
        # Filter out NaN test_cosine
        model_df = model_df[model_df["test_cosine"].notna()]
        
        if len(model_df) == 0:
            logger.warning(f"No valid data for {model}, skipping heatmap")
            continue
        
        # Pivot table: rel_threshold × k_eff
        pivot = model_df.pivot_table(
            values="test_cosine",
            index="rel_threshold",
            columns="k_eff",
            aggfunc="mean"  # Average if multiple runs per setting
        )
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            cbar_kws={"label": "Test Cosine"},
            linewidths=0.5,
            linecolor="gray"
        )
        plt.xlabel("PCA Components (k_eff)", fontsize=12)
        plt.ylabel("Reliability Threshold", fontsize=12)
        plt.title(f"{model}: Test Cosine Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        # Save figure
        fig_path = fig_dir / f"heatmap_test_cosine_{model}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info(f"✅ Saved {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ablation study report and visualizations"
    )
    
    parser.add_argument("--subject", default="subj01", help="Subject ID")
    parser.add_argument(
        "--csv",
        help="Path to ablation CSV (default: outputs/reports/{subject}/ablation_ridge.csv)"
    )
    parser.add_argument(
        "--out",
        help="Output markdown path (default: outputs/reports/{subject}/ablation_summary.md)"
    )
    parser.add_argument(
        "--fig-dir",
        help="Output directory for figures (default: outputs/reports/{subject}/figs)"
    )
    
    args = parser.parse_args()
    
    # Default paths
    csv_path = Path(args.csv) if args.csv else Path(f"outputs/reports/{args.subject}/ablation_ridge.csv")
    out_path = Path(args.out) if args.out else Path(f"outputs/reports/{args.subject}/ablation_summary.md")
    fig_dir = Path(args.fig_dir) if args.fig_dir else Path(f"outputs/reports/{args.subject}/figs")
    
    try:
        # Check CSV exists
        if not csv_path.exists():
            logger.error(f"Ablation CSV not found: {csv_path}")
            return 1
        
        logger.info("=" * 80)
        logger.info("ABLATION STUDY REPORTER")
        logger.info("=" * 80)
        logger.info(f"Input CSV: {csv_path}")
        logger.info(f"Output summary: {out_path}")
        logger.info(f"Figure directory: {fig_dir}")
        
        # Load data
        df = load_ablation_csv(csv_path)
        
        # Generate summary
        out_path.parent.mkdir(parents=True, exist_ok=True)
        generate_summary_md(df, out_path)
        
        # Generate plots
        plot_test_cosine_vs_k(df, fig_dir)
        plot_heatmap(df, fig_dir)
        
        logger.info("\n" + "=" * 80)
        logger.info("REPORT GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Summary: {out_path}")
        logger.info(f"Figures: {fig_dir}/")
        
        return 0
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
