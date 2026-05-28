#!/usr/bin/env python3
"""
NeuroBridge-OT Results Summarizer
====================================

Aggregate metrics from multiple experiments into tables and markdown reports.

Usage:
    python scripts/neurobridge_ot/summarize_neurobridge_results.py \
        --results-dir experimental_results/ \
        --output-dir experimental_results/neurobridge_ot_summary
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def find_metrics_files(results_dir: Path) -> list:
    """Find all metrics.json files under NeuroBridge-OT experiments."""
    metrics_files = []
    for p in results_dir.rglob("metrics.json"):
        if "neurobridge_ot" in str(p):
            metrics_files.append(p)
    return sorted(metrics_files)


def build_comparison_table(metrics_list: list) -> str:
    """Build markdown comparison table."""
    if not metrics_list:
        return "No results found yet.\n"

    lines = [
        "| Experiment | R@1 | R@5 | R@10 | MRR | Median Rank |",
        "|---|---|---|---|---|---|",
    ]

    for entry in metrics_list:
        name = entry.get("experiment", "unknown")
        r1 = entry.get("val_r@1", entry.get("r@1", entry.get("best_val_r@1", "-")))
        r5 = entry.get("val_r@5", entry.get("r@5", "-"))
        r10 = entry.get("val_r@10", entry.get("r@10", "-"))
        mrr = entry.get("val_mrr", entry.get("mrr", "-"))
        med_rank = entry.get("val_median_rank", entry.get("median_rank", "-"))

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

        lines.append(f"| {name} | {fmt(r1)} | {fmt(r5)} | {fmt(r10)} | {fmt(mrr)} | {fmt(med_rank)} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="NeuroBridge-OT Results Summarizer")
    parser.add_argument("--results-dir", type=str, default="experimental_results")
    parser.add_argument("--output-dir", type=str, default="experimental_results/neurobridge_ot_summary")
    parser.add_argument("--subjects", nargs="+",
                        default=["subj01", "subj02", "subj03", "subj04",
                                 "subj05", "subj06", "subj07", "subj08"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Scanning %s for NeuroBridge-OT results...", results_dir)
    metrics_files = find_metrics_files(results_dir)
    logger.info("Found %d metrics files", len(metrics_files))

    # Load all metrics
    all_metrics = []
    for mf in metrics_files:
        try:
            with open(mf) as f:
                data = json.load(f)
            # Infer experiment name from path
            rel = mf.relative_to(results_dir)
            data["experiment"] = str(rel.parent)
            data["metrics_path"] = str(mf)
            all_metrics.append(data)
        except Exception as e:
            logger.warning("Failed to load %s: %s", mf, e)

    # Build summary
    table = build_comparison_table(all_metrics)

    # Write markdown report
    report = f"""# NeuroBridge-OT Results Summary

Generated: {datetime.now(timezone.utc).isoformat()}

## Comparison Table

{table}

## Experiment Details

"""
    for entry in all_metrics:
        report += f"### {entry.get('experiment', 'unknown')}\n\n"
        report += f"- Path: `{entry.get('metrics_path', '')}`\n"
        for k, v in sorted(entry.items()):
            if k not in ("experiment", "metrics_path"):
                report += f"- {k}: {v}\n"
        report += "\n"

    with open(output_dir / "summary.md", "w") as f:
        f.write(report)

    # Save structured JSON
    summary_data = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_experiments": len(all_metrics),
        "experiments": all_metrics,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    logger.info("Summary written to %s", output_dir)
    logger.info("\n%s", table)


if __name__ == "__main__":
    main()
