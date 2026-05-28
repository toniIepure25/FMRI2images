#!/usr/bin/env python3
"""
NeuroBridge-OT LOSO Script
=============================

Leave-One-Subject-Out training and evaluation.
For each target subject, trains on all other subjects and evaluates
on the held-out target. This is true zero-shot LOSO.

Usage:
    python scripts/neurobridge_ot/run_loso_neurobridge_ot.py \
        --config configs/experiments/neurobridge_ot/neurobridge_ot_loso.yaml \
        --all-subjects subj01 subj02 subj05 subj07 \
        --output-dir experimental_results/neurobridge_ot_loso
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="NeuroBridge-OT LOSO")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--all-subjects", nargs="+",
                        default=["subj01", "subj02", "subj05", "subj07"])
    parser.add_argument("--output-dir", type=str, default="experimental_results/neurobridge_ot_loso")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csls-k", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_subjects = args.all_subjects
    loso_results = {}

    logger.info("=" * 70)
    logger.info("NeuroBridge-OT LOSO: Leave-One-Subject-Out")
    logger.info("All subjects: %s", all_subjects)
    logger.info("=" * 70)

    script_dir = Path(__file__).parent
    train_script = script_dir / "train_neurobridge_ot.py"
    eval_script = script_dir / "eval_neurobridge_ot.py"

    for target_subj in all_subjects:
        source_subjects = [s for s in all_subjects if s != target_subj]
        fold_dir = output_dir / f"target_{target_subj}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        logger.info("\n" + "=" * 60)
        logger.info("LOSO fold: target=%s, sources=%s", target_subj, source_subjects)
        logger.info("=" * 60)

        # Train on source subjects
        train_cmd = [
            sys.executable, str(train_script),
            "--config", args.config,
            "--subjects", *source_subjects,
            "--gpu", str(args.gpu),
            "--seed", str(args.seed),
            "--output-dir", str(fold_dir / "train"),
        ]
        if args.limit_batches:
            train_cmd.extend(["--limit-batches", str(args.limit_batches)])

        if args.dry_run:
            train_cmd.append("--dry-run")
            logger.info("DRY RUN: %s", " ".join(train_cmd))
        else:
            logger.info("Running: %s", " ".join(train_cmd))
            result = subprocess.run(train_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Training failed for fold target=%s:\n%s", target_subj, result.stderr[-2000:])
                loso_results[target_subj] = {"error": "training_failed"}
                continue

        # Evaluate on target subject
        ckpt_path = fold_dir / "train" / "checkpoints" / "best.pt"
        if not ckpt_path.exists() and not args.dry_run:
            ckpt_path = fold_dir / "train" / "checkpoints" / "last.pt"

        eval_cmd = [
            sys.executable, str(eval_script),
            "--checkpoint", str(ckpt_path),
            "--config", args.config,
            "--subjects", target_subj,
            "--split", "shared1000",
            "--csls-k", str(args.csls_k),
            "--output-dir", str(fold_dir / "eval"),
            "--gpu", str(args.gpu),
        ]
        if args.limit_batches:
            eval_cmd.extend(["--limit-batches", str(args.limit_batches)])

        if args.dry_run:
            logger.info("DRY RUN eval: %s", " ".join(eval_cmd))
            loso_results[target_subj] = {"dry_run": True}
        else:
            logger.info("Running eval: %s", " ".join(eval_cmd))
            result = subprocess.run(eval_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Eval failed for target=%s:\n%s", target_subj, result.stderr[-2000:])
                loso_results[target_subj] = {"error": "eval_failed"}
                continue

            # Load metrics
            metrics_path = fold_dir / "eval" / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    loso_results[target_subj] = json.load(f)
                logger.info("target=%s R@1=%.4f", target_subj,
                            loso_results[target_subj].get("r@1", 0))

    # Summary
    summary = {
        "protocol": "loso_generalization",
        "taxonomy_label": "loso_generalization",
        "all_subjects": all_subjects,
        "per_subject": loso_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    valid_results = [v for v in loso_results.values() if "r@1" in v]
    if valid_results:
        summary["mean_r@1"] = sum(v["r@1"] for v in valid_results) / len(valid_results)
        summary["mean_r@5"] = sum(v["r@5"] for v in valid_results) / len(valid_results)
        summary["mean_mrr"] = sum(v["mrr"] for v in valid_results) / len(valid_results)
        logger.info("\nLOSO Summary: mean R@1=%.4f across %d folds",
                    summary["mean_r@1"], len(valid_results))

    with open(output_dir / "loso_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
