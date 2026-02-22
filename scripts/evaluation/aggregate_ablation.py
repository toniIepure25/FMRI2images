#!/usr/bin/env python3
"""
Aggregate Ablation Results Across Experiments and Subjects
==========================================================

Reads metrics/summary.json from each experiment x subject run and produces:
1. ablation_summary.csv  -- full table
2. ablation_summary.tex  -- LaTeX table for paper
3. ablation_comparison.png -- bar chart with error bars

Usage:
    python scripts/evaluation/aggregate_ablation.py
    python scripts/evaluation/aggregate_ablation.py \
        --results-dir experimental_results \
        --experiments B0_deterministic B1_gaussian N1_vmf_nce N2_roi_transformer N3_roi_dcf N4_full_system \
        --subjects subj01 subj02 subj05 subj07
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENTS = [
    "B0_deterministic",
    "B1_gaussian",
    "N1_vmf_nce",
    "N2_roi_transformer",
    "N3_roi_dcf",
    "N4_full_system",
]

SUBJECTS = ["subj01", "subj02", "subj05", "subj07"]


def load_summary(results_dir: Path, exp: str, subj: str) -> Optional[Dict]:
    """Load metrics/summary.json for one experiment x subject."""
    path = results_dir / exp / subj / "metrics" / "summary.json"
    if not path.exists():
        logger.warning("Missing: %s", path)
        return None
    with open(path) as f:
        return json.load(f)


def collect_results(
    results_dir: Path,
    experiments: List[str],
    subjects: List[str],
) -> pd.DataFrame:
    """Collect best_val_loss and wall_time across all runs."""
    rows = []
    for exp in experiments:
        for subj in subjects:
            summary = load_summary(results_dir, exp, subj)
            if summary is None:
                continue
            rows.append({
                "experiment": exp,
                "subject": subj,
                "best_val_loss": summary.get("best_val_loss"),
                "best_epoch": summary.get("best_epoch"),
                "total_epochs": summary.get("total_epochs"),
                "wall_time_min": round(summary.get("wall_time_seconds", 0) / 60, 1),
                "final_train_loss": summary.get("final_train_loss"),
                "final_val_loss": summary.get("final_val_loss"),
            })
    return pd.DataFrame(rows)


def aggregate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to experiment x metric with mean +/- std across subjects."""
    if df.empty:
        return df
    grouped = df.groupby("experiment").agg(
        val_loss_mean=("best_val_loss", "mean"),
        val_loss_std=("best_val_loss", "std"),
        epoch_mean=("best_epoch", "mean"),
        wall_min_mean=("wall_time_min", "mean"),
        n_subjects=("subject", "count"),
    ).reindex(df["experiment"].unique())
    return grouped.round(4)


def write_latex(agg: pd.DataFrame, path: Path) -> None:
    """Write a LaTeX table suitable for a paper."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Ablation study: validation loss across experiments (mean $\pm$ std over subjects).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Experiment & Val Loss & Best Epoch & Wall Time (min) \\",
        r"\midrule",
    ]
    for exp, row in agg.iterrows():
        name = exp.replace("_", r"\_")
        loss = f"${row['val_loss_mean']:.4f} \\pm {row['val_loss_std']:.4f}$"
        epoch = f"{row['epoch_mean']:.0f}"
        wtime = f"{row['wall_min_mean']:.0f}"
        lines.append(f"{name} & {loss} & {epoch} & {wtime} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines))
    logger.info("Wrote LaTeX table to %s", path)


def write_bar_chart(agg: pd.DataFrame, path: Path) -> None:
    """Write a bar chart comparing val loss across experiments."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping bar chart")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(agg))
    ax.bar(
        x,
        agg["val_loss_mean"],
        yerr=agg["val_loss_std"],
        capsize=4,
        color=["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974", "#64b5cd"][: len(agg)],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg.index, rotation=30, ha="right")
    ax.set_ylabel("Best Validation Loss")
    ax.set_title("Ablation Ladder: B0 → N4")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote bar chart to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ablation results")
    parser.add_argument("--results-dir", type=Path, default=Path("experimental_results"))
    parser.add_argument("--experiments", nargs="+", default=EXPERIMENTS)
    parser.add_argument("--subjects", nargs="+", default=SUBJECTS)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    out = args.output_dir or args.results_dir
    out.mkdir(parents=True, exist_ok=True)

    df = collect_results(args.results_dir, args.experiments, args.subjects)
    if df.empty:
        logger.error("No results found in %s", args.results_dir)
        sys.exit(1)

    df.to_csv(out / "ablation_summary.csv", index=False)
    logger.info("Wrote %d rows to ablation_summary.csv", len(df))

    agg = aggregate_table(df)
    print("\n" + agg.to_string())

    write_latex(agg, out / "ablation_summary.tex")
    write_bar_chart(agg, out / "ablation_comparison.png")


if __name__ == "__main__":
    main()
