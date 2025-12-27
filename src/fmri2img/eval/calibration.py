import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_calibration_assets(trial_csv: Union[str, Path], run_dir: Union[str, Path]) -> None:
    """Create calibration/reliability artifacts from trial_results.csv.

    Outputs:
        paper_assets/calibration_reliability.png
        paper_assets/calibration_binned.csv
        paper_assets/calibration_summary.json
    """
    trial_csv = Path(trial_csv)
    run_dir = Path(run_dir)
    paper_dir = run_dir / "paper_assets"
    paper_dir.mkdir(parents=True, exist_ok=True)

    if not trial_csv.exists():
        raise FileNotFoundError(f"trial_results CSV not found: {trial_csv}")

    df = pd.read_csv(trial_csv)
    if "uncertainty" not in df.columns or "best_score" not in df.columns:
        raise ValueError("trial_results.csv must contain 'uncertainty' and 'best_score' columns")

    # Clean
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["uncertainty", "best_score"])

    # Bin by uncertainty quantiles
    quantiles = np.linspace(0, 1, 6)
    bins = np.unique(df["uncertainty"].quantile(quantiles).values)
    if len(bins) < 2:
        bins = np.array([df["uncertainty"].min(), df["uncertainty"].max()])
    df["uncertainty_bin"] = pd.cut(df["uncertainty"], bins=bins, include_lowest=True, duplicates="drop")

    grouped = df.groupby("uncertainty_bin").agg(
        n=("uncertainty", "count"),
        uncertainty_mean=("uncertainty", "mean"),
        score_mean=("best_score", "mean"),
        score_std=("best_score", "std"),
    ).reset_index()

    binned_csv = paper_dir / "calibration_binned.csv"
    grouped.to_csv(binned_csv, index=False)

    summary = {
        "n_trials": int(len(df)),
        "uncertainty_mean": float(df["uncertainty"].mean()),
        "score_mean": float(df["best_score"].mean()),
        "bins": len(grouped),
    }

    summary_path = paper_dir / "calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Plot reliability curve (uncertainty vs best_score)
    plt.figure(figsize=(6, 4))
    plt.plot(grouped["uncertainty_mean"], grouped["score_mean"], marker="o")
    plt.xlabel("Uncertainty (mean per bin)")
    plt.ylabel("Best score (mean)")
    plt.title("Calibration / Reliability Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    png_path = paper_dir / "calibration_reliability.png"
    plt.savefig(png_path, dpi=200)
    plt.close()

    return None
