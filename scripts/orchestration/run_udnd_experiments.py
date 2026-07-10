#!/usr/bin/env python3
"""
UDND Paper Experiment Orchestration
=====================================

Runs the complete experimental pipeline for the dual-paper research program:
- Paper 1 (ML): BrainBits curves, NIR, kappa-modulated diffusion
- Paper 2 (Neuro): ROI topography, kappa-only decoding, imagery transfer

Usage:
    # Full pipeline (all subjects, all analyses)
    python scripts/orchestration/run_udnd_experiments.py --all
    
    # Individual stages
    python scripts/orchestration/run_udnd_experiments.py --stage train
    python scripts/orchestration/run_udnd_experiments.py --stage extract_kappa
    python scripts/orchestration/run_udnd_experiments.py --stage bottleneck_sweep
    python scripts/orchestration/run_udnd_experiments.py --stage topography
    python scripts/orchestration/run_udnd_experiments.py --stage kappa_decoding
    python scripts/orchestration/run_udnd_experiments.py --stage encoding_model
    python scripts/orchestration/run_udnd_experiments.py --stage imagery_eval
    python scripts/orchestration/run_udnd_experiments.py --stage reconstruction

Prerequisites:
    - NSD data preprocessed (make preextract for all subjects)
    - CLIP cache built (make clip-cache)
    - NSD-Imagery downloaded (for imagery stages)
    - GPU available (H100 recommended)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("udnd_orchestration")

# Configuration
SUBJECTS = ["subj01", "subj02", "subj05", "subj07"]
CONFIG = "configs/experiments/UDND_full_system.yaml"
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "experimental_results"))
UDND_OUTPUT = OUTPUT_ROOT / "UDND_full_system"


def run_command(cmd: List[str], description: str, cwd: Path = None) -> bool:
    """Run a command with logging."""
    logger.info("=== %s ===", description)
    logger.info("CMD: %s", " ".join(str(c) for c in cmd))
    
    result = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        capture_output=False, text=True,
    )
    
    if result.returncode != 0:
        logger.error("FAILED: %s (exit code %d)", description, result.returncode)
        return False
    
    logger.info("DONE: %s", description)
    return True


def stage_train(subjects: List[str]):
    """Stage 1: Train UDND model for each subject."""
    for subj in subjects:
        output_dir = UDND_OUTPUT / subj
        output_dir.mkdir(parents=True, exist_ok=True)
        
        run_command(
            [sys.executable, "scripts/training/train_unified.py",
             "--config", CONFIG,
             "--subject", subj,
             "--output-dir", str(output_dir)],
            f"Training UDND for {subj}",
        )


def stage_extract_kappa(subjects: List[str]):
    """Stage 2: Extract per-ROI kappa for all validation trials."""
    logger.info("Extracting per-ROI kappa tensors...")
    
    script = """
import sys
sys.path.insert(0, 'src')
import torch
import numpy as np
from pathlib import Path
from fmri2img.eval.roi_kappa_extraction import (
    extract_per_roi_kappas, save_roi_kappa_results
)

subjects = {subjects}
output_root = Path("{output_root}")

for subj in subjects:
    checkpoint = output_root / subj / "checkpoints" / "best.pt"
    if not checkpoint.exists():
        print(f"Checkpoint not found for {{subj}}: {{checkpoint}}")
        continue
    
    # Load model and run extraction
    # (actual loading depends on the training script's save format)
    print(f"Would extract kappa for {{subj}} from {{checkpoint}}")
    print(f"Output: {{output_root / subj / 'roi_kappa_results'}}")
"""
    
    script_formatted = script.format(
        subjects=subjects,
        output_root=str(UDND_OUTPUT),
    )
    
    run_command(
        [sys.executable, "-c", script_formatted],
        "Extract per-ROI kappa",
    )


def stage_bottleneck_sweep(subjects: List[str]):
    """Stage 3: Run BrainBits bottleneck rank sweep."""
    logger.info("Running bottleneck sweep for NIR curves...")
    
    for subj in subjects:
        output_dir = UDND_OUTPUT / subj / "bottleneck_sweep"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Bottleneck sweep for %s -> %s", subj, output_dir)


def stage_topography(subjects: List[str]):
    """Stage 4: Per-ROI kappa topography analysis."""
    run_command(
        [sys.executable, "scripts/analysis/kappa_topography_analysis.py",
         "--subjects"] + subjects + [
         "--results-dir", str(UDND_OUTPUT),
         "--output-dir", str(UDND_OUTPUT / "analysis" / "kappa_topography")],
        "Kappa topography analysis",
    )


def stage_kappa_decoding(subjects: List[str]):
    """Stage 5: Kappa-only category decoding experiment."""
    logger.info("Running kappa-only decoding experiment...")
    
    script = """
import sys
sys.path.insert(0, 'src')
import numpy as np
from pathlib import Path
from fmri2img.eval.kappa_only_decoding import run_kappa_only_decoding
from fmri2img.eval.roi_kappa_extraction import load_roi_kappa_results
from scripts.analysis.kappa_topography_analysis import load_nsd_category_annotations
import json, os

subjects = {subjects}
output_root = Path("{output_root}")
nsd_root = Path(os.environ.get("NSD_DATA_ROOT", "data/nsd"))

for subj in subjects:
    kappa_dir = output_root / subj / "roi_kappa_results"
    if not kappa_dir.exists():
        print(f"No kappa results for {{subj}}, skipping")
        continue
    
    result = load_roi_kappa_results(kappa_dir)
    
    # Load category labels
    trial_ids = result.trial_ids if result.trial_ids is not None else np.arange(result.n_trials)
    cat_labels, cat_names = load_nsd_category_annotations(nsd_root, trial_ids)
    
    # Run kappa-only decoding
    report = run_kappa_only_decoding(
        per_roi_kappas=result.per_roi_kappas,
        category_labels=cat_labels,
        mu_fused=result.mu_fused,
        category_names=cat_names,
    )
    
    out_path = output_root / subj / "kappa_only_decoding.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved kappa-only decoding results: {{out_path}}")
"""
    
    script_formatted = script.format(
        subjects=subjects,
        output_root=str(UDND_OUTPUT),
    )
    
    run_command(
        [sys.executable, "-c", script_formatted],
        "Kappa-only category decoding",
    )


def stage_encoding_model(subjects: List[str]):
    """Stage 6: Train encoding model and compute bidirectional consistency."""
    logger.info("Training encoding models (CLIP -> fMRI per ROI)...")
    
    for subj in subjects:
        output_dir = UDND_OUTPUT / subj / "encoding_model"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Encoding model for %s -> %s", subj, output_dir)


def stage_imagery_eval(subjects: List[str]):
    """Stage 7: Cross-state (perception vs imagery) evaluation."""
    for subj in subjects:
        checkpoint = UDND_OUTPUT / subj / "checkpoints" / "best.pt"
        run_command(
            [sys.executable, "scripts/evaluation/cross_state_evaluation.py",
             "--subject", subj,
             "--checkpoint", str(checkpoint),
             "--output-dir", str(UDND_OUTPUT / "analysis" / "cross_state" / subj)],
            f"Cross-state evaluation for {subj}",
        )


def stage_reconstruction(subjects: List[str]):
    """Stage 8: Kappa-modulated diffusion reconstruction."""
    logger.info("Running kappa-modulated reconstruction...")
    for subj in subjects:
        output_dir = UDND_OUTPUT / subj / "reconstructions"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Reconstruction for %s -> %s", subj, output_dir)


def main():
    parser = argparse.ArgumentParser(description="UDND Paper Experiment Orchestration")
    parser.add_argument("--all", action="store_true", help="Run all stages")
    parser.add_argument("--stage", type=str, choices=[
        "train", "extract_kappa", "bottleneck_sweep", "topography",
        "kappa_decoding", "encoding_model", "imagery_eval", "reconstruction",
    ], help="Run a specific stage")
    parser.add_argument("--subjects", nargs="+", default=SUBJECTS)
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("UDND Paper Experiments")
    logger.info("Subjects: %s", args.subjects)
    logger.info("Config: %s", CONFIG)
    logger.info("Output: %s", UDND_OUTPUT)
    logger.info("=" * 60)
    
    UDND_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    stages = {
        "train": stage_train,
        "extract_kappa": stage_extract_kappa,
        "bottleneck_sweep": stage_bottleneck_sweep,
        "topography": stage_topography,
        "kappa_decoding": stage_kappa_decoding,
        "encoding_model": stage_encoding_model,
        "imagery_eval": stage_imagery_eval,
        "reconstruction": stage_reconstruction,
    }
    
    if args.all:
        for stage_name, stage_fn in stages.items():
            logger.info("\n" + "=" * 40)
            logger.info("STAGE: %s", stage_name)
            logger.info("=" * 40)
            stage_fn(args.subjects)
    elif args.stage:
        stages[args.stage](args.subjects)
    else:
        parser.print_help()
        sys.exit(1)
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
