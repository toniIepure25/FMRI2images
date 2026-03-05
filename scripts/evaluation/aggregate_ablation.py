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
    "B0v2_deterministic",
    "B1v2_gaussian",
    "N1v2_vmf_nce",
    "N2v2_roi_transformer",
    "N3v2_roi_dcf",
    "N4v2_full_system",
    "B0v3_deterministic",
    "B1v3_gaussian",
    "N1v3_vmf_nce",
    "N2v3_roi_transformer",
    "N3v3_roi_dcf",
    "N4v3_full_system",
    "B0v4_deterministic",
    "B1v4_gaussian",
    "N1v4_vmf_nce",
    "N2v4_roi_transformer",
    "N3v4_roi_dcf",
    "N4v4_full_system",
    "N1v5_vmf_nce",
    "N2v5_roi_transformer",
    "N3v5_roi_dcf",
    "N4v5_full_system",
    "N1v6_vmf_nce",
    "N2v6_roi_transformer",
    "N3v6_roi_dcf",
    "N4v6_full_system",
    "N1v7_vmf_nce",
    "N2v7_roi_transformer",
    "N3v7_roi_dcf",
    "N4v7_full_system",
    "N3v8_roi_dcf",
    "N4v8_full_system",
    "N1v9_vmf_nce",
    "N2v9_roi_transformer",
    "N3v9_roi_dcf",
    "N4v9_full_system",
    "N1v10_vmf_nce",
    "N2v10_roi_transformer",
    "N3v10_roi_dcf",
    "N4v10_full_system",
    "N1v11_vmf_nce",
    "N2v11_roi_transformer",
    "N3v11_roi_dcf",
    "N4v11_full_system",
    "N1v12_vmf_nce",
    "N2v12_roi_transformer",
    "N3v12_roi_dcf",
    "N4v12_full_system",
    "N1v13_vmf_nce",
    "N2v13_roi_transformer",
    "N3v13_roi_dcf",
    "N4v13_full_system",
    "N1v14_vmf_nce",
    "N2v14_roi_transformer",
    "N3v14_roi_dcf",
    "N4v14_full_system",
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
                "best_r@1": summary.get("best_r@1", None),
                "best_epoch": summary.get("best_epoch"),
                "total_epochs": summary.get("total_epochs"),
                "wall_time_min": round(summary.get("wall_time_seconds", 0) / 60, 1),
                "final_train_loss": summary.get("final_train_loss"),
                "final_val_loss": summary.get("final_val_loss"),
                "final_r@1": summary.get("final_r@1", None),
            })
    return pd.DataFrame(rows)


def aggregate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to experiment x metric with mean +/- std across subjects."""
    if df.empty:
        return df
    agg_spec = {
        "val_loss_mean": ("best_val_loss", "mean"),
        "val_loss_std": ("best_val_loss", "std"),
        "epoch_mean": ("best_epoch", "mean"),
        "wall_min_mean": ("wall_time_min", "mean"),
        "n_subjects": ("subject", "count"),
    }
    if "best_r@1" in df.columns and df["best_r@1"].notna().any():
        agg_spec["r@1_mean"] = ("best_r@1", "mean")
        agg_spec["r@1_std"] = ("best_r@1", "std")
    grouped = df.groupby("experiment").agg(**agg_spec).reindex(df["experiment"].unique())
    return grouped.round(4)


def write_latex(agg: pd.DataFrame, path: Path) -> None:
    """Write a LaTeX table suitable for a paper."""
    has_r1 = "r@1_mean" in agg.columns
    ncols = "lcccc" if has_r1 else "lccc"
    header = r"Experiment & Val Loss & R@1 & Best Epoch & Wall (min) \\" if has_r1 else \
             r"Experiment & Val Loss & Best Epoch & Wall Time (min) \\"
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Ablation study across experiments (mean $\pm$ std over subjects).}",
        r"\label{tab:ablation}",
        f"\\begin{{tabular}}{{{ncols}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for exp, row in agg.iterrows():
        name = exp.replace("_", r"\_")
        loss = f"${row['val_loss_mean']:.4f} \\pm {row['val_loss_std']:.4f}$"
        epoch = f"{row['epoch_mean']:.0f}"
        wtime = f"{row['wall_min_mean']:.0f}"
        if has_r1 and pd.notna(row.get("r@1_mean")):
            r1 = f"${row['r@1_mean']:.4f} \\pm {row['r@1_std']:.4f}$"
            lines.append(f"{name} & {loss} & {r1} & {epoch} & {wtime} \\\\")
        elif has_r1:
            lines.append(f"{name} & {loss} & --- & {epoch} & {wtime} \\\\")
        else:
            lines.append(f"{name} & {loss} & {epoch} & {wtime} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines))
    logger.info("Wrote LaTeX table to %s", path)


def write_bar_chart(agg: pd.DataFrame, path: Path) -> None:
    """Write a bar chart comparing R@1 (or val loss) across experiments."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping bar chart")
        return

    has_r1 = "r@1_mean" in agg.columns and agg["r@1_mean"].notna().any()
    palette = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974", "#64b5cd",
               "#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974", "#64b5cd"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(agg))

    if has_r1:
        vals = agg["r@1_mean"].fillna(0)
        errs = agg["r@1_std"].fillna(0)
        ylabel = "Retrieval R@1 (higher is better)"
        title = "Ablation Ladder: Retrieval Accuracy (R@1)"
    else:
        vals = agg["val_loss_mean"]
        errs = agg["val_loss_std"]
        ylabel = "Best Validation Loss"
        title = "Ablation Ladder: Validation Loss"

    ax.bar(x, vals, yerr=errs, capsize=4,
           color=palette[: len(agg)], edgecolor="black", linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg.index, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
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
