#!/usr/bin/env python3
"""
NeuroBridge-OT Ablation Suite
================================

Run ablations with feature flags to test individual component contributions.

Usage:
    python scripts/neurobridge_ot/run_ablation_suite.py \
        --configs-dir configs/experiments/neurobridge_ot/ \
        --subjects subj01 subj02 subj05 subj07 \
        --output-dir experimental_results/neurobridge_ot_ablation
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ABLATION_CONFIGS = [
    "neurobridge_ot_full.yaml",
    "neurobridge_ot_no_ot.yaml",
    "neurobridge_ot_no_teacher.yaml",
    "neurobridge_ot_no_hyperadapter.yaml",
    "neurobridge_ot_no_adversarial.yaml",
    "neurobridge_ot_roi_summary_baseline.yaml",
]


def main():
    parser = argparse.ArgumentParser(description="NeuroBridge-OT Ablation Suite")
    parser.add_argument("--configs-dir", type=str,
                        default="configs/experiments/neurobridge_ot/")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific config filenames to run (default: all ablation configs)")
    parser.add_argument("--subjects", nargs="+", default=["subj01", "subj02", "subj05", "subj07"])
    parser.add_argument("--output-dir", type=str,
                        default="experimental_results/neurobridge_ot_ablation")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=None)
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = args.configs if args.configs else ABLATION_CONFIGS
    script_dir = Path(__file__).parent
    train_script = script_dir / "train_neurobridge_ot.py"

    logger.info("=" * 70)
    logger.info("NeuroBridge-OT Ablation Suite")
    logger.info("Configs: %s", configs)
    logger.info("Subjects: %s", args.subjects)
    logger.info("=" * 70)

    all_results = {}

    for config_name in configs:
        config_path = configs_dir / config_name
        if not config_path.exists():
            logger.warning("Config not found: %s", config_path)
            continue

        exp_name = config_name.replace(".yaml", "")
        logger.info("\n--- Ablation: %s ---", exp_name)

        exp_dir = output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(train_script),
            "--config", str(config_path),
            "--subjects", *args.subjects,
            "--gpu", str(args.gpu),
            "--seed", str(args.seed),
            "--output-dir", str(exp_dir),
        ]
        if args.limit_batches:
            cmd.extend(["--limit-batches", str(args.limit_batches)])
        if args.dry_run:
            cmd.append("--dry-run")

        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("Failed: %s\n%s", exp_name, result.stderr[-1000:])
            all_results[exp_name] = {"status": "failed", "error": result.stderr[-500:]}
        else:
            # Load metrics
            metrics_path = exp_dir / "metrics" / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    all_results[exp_name] = json.load(f)
                    all_results[exp_name]["status"] = "success"
            else:
                all_results[exp_name] = {"status": "completed_no_metrics"}

    # Summary table
    summary = {
        "ablation_suite": "neurobridge_ot",
        "configs": configs,
        "subjects": args.subjects,
        "results": all_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(output_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION RESULTS")
    logger.info("=" * 70)
    logger.info("%-40s | %8s | %8s | %8s", "Config", "R@1", "R@5", "MRR")
    logger.info("-" * 70)
    for name, metrics in all_results.items():
        if metrics.get("status") == "success":
            logger.info("%-40s | %8.4f | %8.4f | %8.4f",
                        name,
                        metrics.get("val_r@1", metrics.get("best_val_r@1", 0)),
                        metrics.get("val_r@5", 0),
                        metrics.get("val_mrr", 0))
        else:
            logger.info("%-40s | %8s", name, metrics.get("status", "unknown"))

    logger.info("\nResults saved to %s", output_dir)


if __name__ == "__main__":
    main()
