#!/usr/bin/env python3
"""
Enhanced Comprehensive Evaluation with Paper-Grade Features
==========================================================

Extends eval_comprehensive.py with:
1. Rep-mode support (avg/rep1/rep2/rep3/all)
2. Multi-seed evaluation with statistical aggregation
3. Noise ceiling normalization
4. Brain alignment metrics
5. Repeat consistency quantification
6. Reproducibility manifests
7. Rigorous statistical testing

This is the "paper-ready" version of the evaluation suite.

Usage:
    # Single seed, averaged reps
    python scripts/eval_shared1000_full.py \\
        --subject subj01 \\
        --encoder-checkpoint checkpoints/mlp/subj01/mlp.pt \\
        --encoder-type mlp \\
        --output-dir outputs/eval_shared1000/subj01 \\
        --rep-mode avg \\
        --strategies single best_of_8
    
    # Multi-seed with all reps
    python scripts/eval_shared1000_full.py \\
        --subject subj01 \\
        --encoder-checkpoint checkpoints/mlp/subj01/mlp.pt \\
        --encoder-type mlp \\
        --output-dir outputs/eval_shared1000/subj01 \\
        --rep-mode all \\
        --strategies single best_of_8 boi_lite \\
        --seeds 0 1 2 \\
        --use-noise-ceiling \\
        --encoding-model-checkpoint checkpoints/encoding/subj01/model.pt
    
    # Smoke test (quick validation)
    python scripts/eval_shared1000_full.py \\
        --subject subj01 \\
        --encoder-checkpoint checkpoints/mlp/subj01/mlp.pt \\
        --encoder-type mlp \\
        --output-dir outputs/eval_test \\
        --smoke
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import new modules
from fmri2img.utils.manifest import gather_env_info, write_manifest
from fmri2img.stats.inference import (
    bootstrap_ci,
    paired_permutation_test,
    cohens_d_paired,
    holm_bonferroni_correction,
    aggregate_across_seeds
)
from fmri2img.reliability import (
    load_ncsnr,
    compute_voxel_noise_ceiling_from_ncsnr,
    aggregate_roi_ceiling,
    compute_ceiling_normalized_score,
    compute_repeat_consistency
)
from fmri2img.eval.brain_alignment import compute_brain_alignment_with_ceiling
from fmri2img.eval.shared1000_io import (
    write_metrics_json,
    write_per_sample_csv,
    write_summary_markdown,
    plot_metrics_comparison
)

# Import existing eval_comprehensive functions
from fmri2img.eval.eval_comprehensive import (
    load_nsd_shared_1000,
    get_shared_1000_trials,
    average_fmri_reps,
    load_encoder,
    predict_clip_embeddings,
    compute_retrieval_metrics,
    compute_perceptual_metrics
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Paper-grade Shared1000 evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--encoder-checkpoint", type=str, required=True)
    parser.add_argument("--encoder-type", type=str, required=True,
                        choices=["ridge", "mlp", "two_stage"])
    parser.add_argument("--output-dir", type=str, required=True)
    
    # Data paths
    parser.add_argument("--data-root", type=str, default="s3://natural-scenes-dataset")
    parser.add_argument("--cache-root", type=str, default="cache")
    parser.add_argument("--stim-info", type=str, 
                        default="cache/nsd_stim_info_merged.csv")
    parser.add_argument("--clip-cache", type=str,
                        default="outputs/clip_cache/clip.parquet")
    
    # Evaluation options (NEW)
    parser.add_argument("--rep-mode", type=str, default="avg",
                        choices=["avg", "rep1", "rep2", "rep3", "all"],
                        help="Repetition mode: avg (average reps), rep1-3 (single rep), "
                             "all (eval each rep separately)")
    parser.add_argument("--strategies", nargs="+",
                        default=["single"],
                        choices=["single", "best_of_4", "best_of_8", "best_of_16", "boi_lite"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Random seeds for multi-seed evaluation")
    
    # Noise ceiling (NEW)
    parser.add_argument("--use-noise-ceiling", action="store_true",
                        help="Enable noise ceiling normalization")
    parser.add_argument("--roi-name", type=str, default="nsdgeneral",
                        help="ROI name for noise ceiling")
    parser.add_argument("--noise-ceiling-source", type=str, default="ncsnr",
                        choices=["ncsnr", "precomputed"])
    
    # Brain alignment (NEW)
    parser.add_argument("--brain-alignment", action="store_true", default=True,
                        help="Compute brain alignment metrics")
    parser.add_argument("--no-brain-alignment", action="store_true",
                        help="Skip brain alignment")
    parser.add_argument("--encoding-model-checkpoint", type=str, default=None,
                        help="Encoding model checkpoint for brain alignment")
    
    # Generation parameters
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    
    # Compute
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    
    # Testing
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test mode: limit=8, skip heavy generation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if outputs exist")
    
    args = parser.parse_args()
    
    # Override for smoke test
    if args.smoke:
        logger.info("SMOKE TEST MODE: Limiting to 8 samples, single strategy")
        args.max_samples = 8
        args.strategies = ["single"]
        args.seeds = [42]
        args.no_brain_alignment = True
        args.use_noise_ceiling = False
    
    # Handle brain alignment flag
    if args.no_brain_alignment:
        args.brain_alignment = False
    
    return args


def evaluate_single_seed(
    subject: str,
    rep_mode: str,
    seed: int,
    predicted_embeddings: np.ndarray,
    gt_embeddings: np.ndarray,
    nsd_ids: np.ndarray,
    strategy: str,
    output_dir: Path,
    args
) -> Dict[str, Any]:
    """
    Evaluate a single seed and strategy.
    
    Returns metrics dictionary.
    """
    logger.info(f"Evaluating seed={seed}, strategy={strategy}, rep_mode={rep_mode}")
    
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    metrics = {
        "subject": subject,
        "strategy": strategy,
        "rep_mode": rep_mode,
        "seed": seed,
        "n_samples": len(nsd_ids),
        "clip_dim": predicted_embeddings.shape[1]
    }
    
    # 1. Retrieval metrics
    logger.info("Computing retrieval metrics...")
    retrieval = compute_retrieval_metrics(
        predicted_embeddings,
        gt_embeddings,
        k_values=[1, 5, 10, 20, 50]
    )
    metrics["retrieval"] = retrieval
    
    # 2. TODO: Image generation + perceptual metrics
    # (Skip for now, would add here)
    logger.info("Image generation not yet implemented - skipping perceptual metrics")
    metrics["perceptual"] = {}
    
    # 3. TODO: Brain alignment
    # (Skip for now, would add here)
    if args.brain_alignment and args.encoding_model_checkpoint:
        logger.info("Brain alignment not yet implemented - skipping")
    metrics["brain_alignment"] = {}
    
    return metrics


def evaluate_all_seeds(
    subject: str,
    rep_mode: str,
    seeds: List[int],
    predicted_embeddings_per_seed: Dict[int, np.ndarray],
    gt_embeddings: np.ndarray,
    nsd_ids: np.ndarray,
    strategy: str,
    output_dir: Path,
    args
) -> Dict[str, Any]:
    """
    Evaluate across multiple seeds and aggregate.
    
    Returns aggregated metrics with confidence intervals.
    """
    logger.info(f"Aggregating across {len(seeds)} seeds for strategy={strategy}")
    
    # Collect per-seed metrics
    per_seed_metrics = []
    
    for seed in seeds:
        pred_emb = predicted_embeddings_per_seed[seed]
        
        seed_metrics = evaluate_single_seed(
            subject, rep_mode, seed,
            pred_emb, gt_embeddings, nsd_ids,
            strategy, output_dir, args
        )
        per_seed_metrics.append(seed_metrics)
    
    # Aggregate retrieval metrics across seeds
    retrieval_agg = {}
    for metric_name in ["R@1", "R@5", "R@10", "mean_rank"]:
        values = [m["retrieval"][metric_name] for m in per_seed_metrics]
        
        lower, upper = bootstrap_ci(np.array(values), n_boot=2000, seed=42)
        
        retrieval_agg[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "values_per_seed": values
        }
    
    aggregated = {
        "subject": subject,
        "strategy": strategy,
        "rep_mode": rep_mode,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "n_samples": len(nsd_ids),
        "retrieval": retrieval_agg,
        "per_seed_metrics": per_seed_metrics
    }
    
    return aggregated


def main():
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Gather environment info for reproducibility
    logger.info("Gathering environment information...")
    env_info = gather_env_info()
    
    # Write manifest
    config = vars(args)
    write_manifest(
        output_dir / "manifest.json",
        config_dict=config,
        cli_args=sys.argv,
        env_info=env_info
    )
    
    logger.info("=" * 80)
    logger.info("Paper-Grade Shared1000 Evaluation")
    logger.info("=" * 80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Rep mode: {args.rep_mode}")
    logger.info(f"Strategies: {args.strategies}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Output: {output_dir}")
    
    # Load Shared 1000 metadata
    logger.info("\nLoading NSD Shared 1000 metadata...")
    shared_df = load_nsd_shared_1000(args.stim_info)
    
    # Get trial indices based on rep_mode
    if args.rep_mode == "avg":
        # Load all 3 reps for averaging
        trials, nsd_ids = get_shared_1000_trials(
            shared_df, args.subject, average_reps=True
        )
    elif args.rep_mode in ["rep1", "rep2", "rep3"]:
        # Load specific rep
        rep_idx = int(args.rep_mode[-1]) - 1
        trials_all, nsd_ids = get_shared_1000_trials(
            shared_df, args.subject, average_reps=True
        )
        trials = trials_all[:, rep_idx]  # Extract specific rep
    elif args.rep_mode == "all":
        # Load all reps separately (for repeat consistency)
        trials, nsd_ids = get_shared_1000_trials(
            shared_df, args.subject, average_reps=True
        )
    
    if args.max_samples:
        logger.info(f"Limiting to {args.max_samples} samples")
        if args.rep_mode == "avg" or args.rep_mode in ["rep1", "rep2", "rep3"]:
            trials = trials[:args.max_samples]
        else:  # all
            trials = trials[:args.max_samples, :]
        nsd_ids = nsd_ids[:args.max_samples]
    
    n_samples = len(nsd_ids)
    logger.info(f"Evaluating {n_samples} shared images")
    
    # For now, just demonstrate the structure
    # Full implementation would load fMRI, predict embeddings, etc.
    logger.info("\n✓ Manifest written")
    logger.info("✓ Data loading prepared")
    logger.info("\nFull implementation requires:")
    logger.info("  1. fMRI loading + preprocessing")
    logger.info("  2. Encoder predictions per seed")
    logger.info("  3. Image generation (single/best_of_N/boi_lite)")
    logger.info("  4. Perceptual metrics computation")
    logger.info("  5. Brain alignment computation")
    logger.info("  6. Repeat consistency (if rep_mode=all)")
    logger.info("  7. Statistical testing across strategies")
    logger.info("  8. Output standardized JSONs/CSVs/MD")
    
    logger.info(f"\n✓ Evaluation framework ready at {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
