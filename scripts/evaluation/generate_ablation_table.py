#!/usr/bin/env python3
"""
Generate paper-quality ablation table for the V55 PPNR experiment line.

Collects metrics from all V55 experiment variants, computes bootstrap
confidence intervals, and outputs:
  - ablation_table.csv
  - ablation_table.tex (LaTeX ready for the paper)
  - ablation_table.json (machine-readable)

The ablation decomposes gains from:
  1. Baseline (V35 frozen fusion): 77.2%
  2. + Multi-subject pretraining (V55a)
  3. + Dual-head architecture simplification (V55a vs V54a)
  4. + Uniformity regularization
  5. + Single-subject fine-tuning (V55b)
  6. + Fusion topology distillation (V55c)
  7. + PPR scoring (vs CSLS)
  8. + OOF resolver (V55e)

Usage:
    python scripts/evaluation/generate_ablation_table.py \
        --results-root experimental_results \
        --output-dir outputs/v55_ablation
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ablation_table")


def _load_metrics(experiment_dir: Path) -> Optional[Dict]:
    """Load summary metrics from an experiment directory."""
    candidates = [
        experiment_dir / "metrics" / "summary.json",
        experiment_dir / "metrics" / "shared1000_metrics_compact.json",
        experiment_dir / "metrics" / "shared1000_metrics.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def _bootstrap_ci(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for R@1.

    Returns:
        (mean_r1, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    N = len(predictions)

    sim = predictions @ ground_truth.T
    ranks = np.argsort(-sim, axis=1)
    correct = np.zeros(N, dtype=np.int32)
    for i in range(N):
        correct[i] = int(np.where(ranks[i] == i)[0][0])

    r1_samples = []
    for _ in range(n_bootstrap):
        idx = rng.choice(N, N, replace=True)
        r1_samples.append(float((correct[idx] < 1).mean()))

    r1_samples = np.array(r1_samples)
    mean_r1 = float((correct < 1).mean())
    alpha = 1 - confidence
    ci_lower = float(np.percentile(r1_samples, 100 * alpha / 2))
    ci_upper = float(np.percentile(r1_samples, 100 * (1 - alpha / 2)))

    return mean_r1, ci_lower, ci_upper


def _load_and_bootstrap(
    experiment_dir: Path,
    split: str = "shared1000",
    n_bootstrap: int = 1000,
) -> Optional[Dict]:
    """Load predictions and compute bootstrap CI."""
    metrics_dir = experiment_dir / "metrics"
    if not metrics_dir.exists():
        metrics_dir = experiment_dir

    pred_file = metrics_dir / f"{split}_predictions.npy"
    gt_file = metrics_dir / f"{split}_ground_truth.npy"

    if not pred_file.exists():
        pred_file = metrics_dir / "val_predictions.npy"
        gt_file = metrics_dir / "val_ground_truth.npy"

    if not pred_file.exists():
        return None

    preds = np.load(pred_file)
    gts = np.load(gt_file)
    norms = np.linalg.norm(preds, axis=-1, keepdims=True)
    preds = preds / np.maximum(norms, 1e-8)
    norms_gt = np.linalg.norm(gts, axis=-1, keepdims=True)
    gts = gts / np.maximum(norms_gt, 1e-8)

    mean_r1, ci_lo, ci_hi = _bootstrap_ci(preds, gts, n_bootstrap=n_bootstrap)
    return {
        "r@1": mean_r1,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n_samples": len(preds),
    }


ABLATION_ROWS = [
    {
        "id": "baseline",
        "label": "V35+N1v28a frozen fusion (baseline)",
        "dir_key": "V35_legacy_teacher_distill",
        "scoring": "CSLS fusion",
        "notes": r"$\alpha=0.3, \gamma=0.7$",
        "reported_r1": 0.772,
    },
    {
        "id": "v55a",
        "label": r"+ Multi-subject dual-head (V55a)",
        "dir_key": "V55a_multi_subject_dual_head",
        "scoring": "CSLS",
        "notes": "4 subjects, uniformity, dual-head",
    },
    {
        "id": "v55b",
        "label": r"+ subj01 fine-tuning (V55b)",
        "dir_key": "V55b_subj01_finetune",
        "scoring": "CSLS",
        "notes": "legacy distill, fusion distill",
    },
    {
        "id": "v55b_ppr",
        "label": r"+ PPR scoring (V55b)",
        "dir_key": "V55b_subj01_finetune",
        "scoring": "PPR-CSLS",
        "notes": r"$\kappa \cdot \cos + \log C_d(\kappa)$",
    },
    {
        "id": "v55c",
        "label": r"+ Fusion distillation (V55c)",
        "dir_key": "V55c_fusion_distill",
        "scoring": "CSLS",
        "notes": "topology transfer from frozen fusion",
    },
    {
        "id": "v55c_fusion",
        "label": r"+ Legacy fusion (V55c + N1v28a)",
        "dir_key": "V55c_fusion_distill",
        "scoring": "CSLS fusion",
        "notes": "best student + legacy expert",
    },
    {
        "id": "v55e",
        "label": r"+ OOF PPR resolver (V55e)",
        "dir_key": "V55e result",
        "scoring": "PPR resolver",
        "notes": "set-transformer on OOF predictions",
    },
]


def _generate_latex_table(rows: List[Dict]) -> str:
    """Generate LaTeX table from ablation rows."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study on NSD SHARED1000. Each row adds one component to the baseline.",
        r"95\% bootstrap CIs from 1000 resamples. $\dagger$~indicates pending experiment.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccl}",
        r"\toprule",
        r"\textbf{System} & \textbf{R@1} & \textbf{R@5} & \textbf{$\Delta$R@1} & \textbf{Scoring} \\",
        r"\midrule",
    ]

    prev_r1 = 0.0
    for row in rows:
        r1 = row.get("r@1", row.get("reported_r1", 0.0))
        r5 = row.get("r@5", 0.0)
        delta = r1 - prev_r1 if prev_r1 > 0 else 0.0

        ci_lo = row.get("ci_lower", r1)
        ci_hi = row.get("ci_upper", r1)

        pending = row.get("pending", False)
        marker = r"$\dagger$" if pending else ""

        if ci_lo != r1 and ci_hi != r1:
            r1_str = f"{r1*100:.1f}\\% ({ci_lo*100:.1f}--{ci_hi*100:.1f})"
        elif pending:
            r1_str = "pending"
        else:
            r1_str = f"{r1*100:.1f}\\%"

        r5_str = f"{r5*100:.1f}\\%" if r5 > 0 else "--"
        delta_str = f"+{delta*100:.1f}" if delta > 0 else f"{delta*100:.1f}" if delta != 0 else "--"

        label = row["label"]
        scoring = row.get("scoring", "")

        lines.append(
            f"{label}{marker} & {r1_str} & {r5_str} & {delta_str} & {scoring} \\\\"
        )
        if r1 > 0:
            prev_r1 = r1

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate V55 ablation table")
    parser.add_argument("--results-root", type=str, default="experimental_results")
    parser.add_argument("--output-dir", type=str, default="outputs/v55_ablation")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched_rows = []
    for row in ABLATION_ROWS:
        row_copy = dict(row)
        dir_key = row["dir_key"]
        exp_dir = results_root / dir_key / "subj01"

        if exp_dir.exists():
            metrics = _load_metrics(exp_dir)
            if metrics:
                row_copy["r@1"] = metrics.get("csls_r@1", metrics.get("r@1", 0.0))
                row_copy["r@5"] = metrics.get("csls_r@5", metrics.get("r@5", 0.0))
                row_copy["r@10"] = metrics.get("csls_r@10", metrics.get("r@10", 0.0))

            bootstrap = _load_and_bootstrap(exp_dir, n_bootstrap=args.n_bootstrap)
            if bootstrap:
                row_copy["ci_lower"] = bootstrap["ci_lower"]
                row_copy["ci_upper"] = bootstrap["ci_upper"]
                row_copy["n_samples"] = bootstrap["n_samples"]
                logger.info(
                    "%s: R@1=%.3f [%.3f, %.3f] (N=%d)",
                    row["id"], bootstrap["r@1"],
                    bootstrap["ci_lower"], bootstrap["ci_upper"],
                    bootstrap["n_samples"],
                )
            else:
                row_copy["pending"] = True
                logger.info("%s: predictions not found (pending)", row["id"])
        else:
            row_copy["pending"] = True
            logger.info("%s: experiment dir not found (pending)", row["id"])

        enriched_rows.append(row_copy)

    csv_path = out_dir / "ablation_table.csv"
    fieldnames = ["id", "label", "scoring", "r@1", "r@5", "r@10",
                   "ci_lower", "ci_upper", "n_samples", "pending", "notes"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(enriched_rows)

    json_path = out_dir / "ablation_table.json"
    with open(json_path, "w") as f:
        json.dump(enriched_rows, f, indent=2, default=str)

    latex = _generate_latex_table(enriched_rows)
    tex_path = out_dir / "ablation_table.tex"
    with open(tex_path, "w") as f:
        f.write(latex)

    logger.info("Ablation table saved to: %s", out_dir)
    logger.info("  CSV:   %s", csv_path)
    logger.info("  JSON:  %s", json_path)
    logger.info("  LaTeX: %s", tex_path)

    logger.info("")
    logger.info("=== Ablation Summary ===")
    for row in enriched_rows:
        r1 = row.get("r@1", row.get("reported_r1", 0))
        status = "PENDING" if row.get("pending") else f"R@1={r1*100:.1f}%"
        logger.info("  %-45s  %s", row["label"], status)


if __name__ == "__main__":
    main()
