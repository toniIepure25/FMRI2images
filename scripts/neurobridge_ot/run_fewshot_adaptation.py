#!/usr/bin/env python3
"""
NeuroBridge-OT Few-Shot Adaptation Script
============================================

Pretrain on source subjects, then adapt to a target subject with variable
sample counts. Compares against training from scratch.

Usage:
    python scripts/neurobridge_ot/run_fewshot_adaptation.py \
        --config configs/experiments/neurobridge_ot/neurobridge_ot_fewshot.yaml \
        --source-subjects subj01 subj02 subj05 \
        --target-subject subj07 \
        --few-shot-sizes 10 25 50 100 250 500 1000 \
        --output-dir experimental_results/neurobridge_ot_fewshot
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

DEFAULT_FEW_SHOT_SIZES = [10, 25, 50, 100, 250, 500, 1000]
ALL_SUBJECTS = ["subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"]


def main():
    parser = argparse.ArgumentParser(description="NeuroBridge-OT Few-Shot Adaptation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--source-subjects", nargs="+", default=None,
                        help="Source subjects for pretraining (default: all except target)")
    parser.add_argument("--target-subject", type=str, default=None,
                        help="Single target subject (use --all-target-subjects for all 8)")
    parser.add_argument("--all-target-subjects", action="store_true",
                        help="Run adaptation for all 8 subjects as targets")
    parser.add_argument("--few-shot-sizes", nargs="+", type=int, default=DEFAULT_FEW_SHOT_SIZES)
    parser.add_argument("--output-dir", type=str, default="experimental_results/neurobridge_ot_8subj_fewshot")
    parser.add_argument("--pretrained-checkpoint", type=str, default=None,
                        help="Skip pretraining, use this checkpoint")
    parser.add_argument("--adaptation-mode", type=str, default="adapter_only",
                        choices=["adapter_only", "head_only", "full_finetune"],
                        help="What parameters to adapt")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csls-k", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=None)
    args = parser.parse_args()

    # Determine target subjects
    if args.all_target_subjects:
        target_subjects = ALL_SUBJECTS
    elif args.target_subject:
        target_subjects = [args.target_subject]
    else:
        logger.error("Must specify --target-subject or --all-target-subjects")
        sys.exit(1)

    for target_subj in target_subjects:
        # Source = all subjects except target
        source_subjects = args.source_subjects or [s for s in ALL_SUBJECTS if s != target_subj]
        _run_fewshot_for_target(args, target_subj, source_subjects)


def _run_fewshot_for_target(args, target_subj: str, source_subjects: list):
    """Run few-shot adaptation for a single target subject."""
    output_dir = Path(args.output_dir) / target_subj
    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).parent
    train_script = script_dir / "train_neurobridge_ot.py"
    eval_script = script_dir / "eval_neurobridge_ot.py"

    logger.info("=" * 70)
    logger.info("NeuroBridge-OT Few-Shot Adaptation")
    logger.info("Source: %s, Target: %s", source_subjects, target_subj)
    logger.info("Few-shot sizes: %s", args.few_shot_sizes)
    logger.info("=" * 70)

    # Step 1: Pretrain on source subjects (if no checkpoint provided)
    pretrain_ckpt = args.pretrained_checkpoint
    if pretrain_ckpt is None:
        pretrain_dir = output_dir / "pretrain"
        pretrain_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable, str(train_script),
            "--config", args.config,
            "--subjects", *source_subjects,
            "--gpu", str(args.gpu),
            "--seed", str(args.seed),
            "--output-dir", str(pretrain_dir),
        ]
        if args.limit_batches:
            train_cmd.extend(["--limit-batches", str(args.limit_batches)])
        if args.dry_run:
            train_cmd.append("--dry-run")

        logger.info("Pretraining on source subjects...")
        result = subprocess.run(train_cmd, capture_output=True, text=True)
        if result.returncode != 0 and not args.dry_run:
            logger.error("Pretraining failed:\n%s", result.stderr[-2000:])
            sys.exit(1)

        pretrain_ckpt = str(pretrain_dir / "checkpoints" / "best.pt")

    # Step 2: Few-shot adaptation at each size
    results = {}
    for n_shots in args.few_shot_sizes:
        logger.info("\n--- Few-shot N=%d ---", n_shots)
        adapt_dir = output_dir / f"adapt_n{n_shots}"
        adapt_dir.mkdir(parents=True, exist_ok=True)

        # Adapted training (pretrain + fine-tune on target)
        adapt_cmd = [
            sys.executable, str(train_script),
            "--config", args.config,
            "--subjects", target_subj,
            "--target-subject", target_subj,
            "--checkpoint", pretrain_ckpt,
            "--gpu", str(args.gpu),
            "--seed", str(args.seed),
            "--output-dir", str(adapt_dir / "adapted"),
        ]
        if args.limit_batches:
            adapt_cmd.extend(["--limit-batches", str(args.limit_batches)])
        if args.dry_run:
            adapt_cmd.append("--dry-run")

        logger.info("Adapting with N=%d samples...", n_shots)
        subprocess.run(adapt_cmd, capture_output=True, text=True)

        # Scratch baseline (train target-only from random init)
        scratch_cmd = [
            sys.executable, str(train_script),
            "--config", args.config,
            "--subjects", target_subj,
            "--gpu", str(args.gpu),
            "--seed", str(args.seed),
            "--output-dir", str(adapt_dir / "scratch"),
        ]
        if args.limit_batches:
            scratch_cmd.extend(["--limit-batches", str(args.limit_batches)])
        if args.dry_run:
            scratch_cmd.append("--dry-run")

        logger.info("Training from scratch with N=%d samples...", n_shots)
        subprocess.run(scratch_cmd, capture_output=True, text=True)

        # Evaluate both
        for mode in ["adapted", "scratch"]:
            ckpt = adapt_dir / mode / "checkpoints" / "best.pt"
            eval_dir = adapt_dir / mode / "eval"

            if ckpt.exists() or args.dry_run:
                eval_cmd = [
                    sys.executable, str(eval_script),
                    "--checkpoint", str(ckpt),
                    "--subjects", target_subj,
                    "--split", "shared1000",
                    "--csls-k", str(args.csls_k),
                    "--output-dir", str(eval_dir),
                    "--gpu", str(args.gpu),
                ]
                if args.limit_batches:
                    eval_cmd.extend(["--limit-batches", str(args.limit_batches)])

                if not args.dry_run:
                    subprocess.run(eval_cmd, capture_output=True, text=True)

                metrics_path = eval_dir / "metrics.json"
                if metrics_path.exists():
                    with open(metrics_path) as f:
                        results[f"n{n_shots}_{mode}"] = json.load(f)

    # Summary
    summary = {
        "protocol": "supervised_few_shot_adaptation",
        "taxonomy_label": "supervised_few_shot_adaptation",
        "source_subjects": source_subjects,
        "target_subject": target_subj,
        "adaptation_mode": args.adaptation_mode,
        "few_shot_sizes": args.few_shot_sizes,
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(output_dir / "fewshot_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print data efficiency curves
    logger.info("\n" + "=" * 60)
    logger.info("Few-Shot Data Efficiency Curves")
    logger.info("=" * 60)
    for n in args.few_shot_sizes:
        adapted = results.get(f"n{n}_adapted", {}).get("r@1", "N/A")
        scratch = results.get(f"n{n}_scratch", {}).get("r@1", "N/A")
        logger.info("N=%4d | adapted=%.4f | scratch=%.4f",
                    n, adapted if isinstance(adapted, float) else 0,
                    scratch if isinstance(scratch, float) else 0)

    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
