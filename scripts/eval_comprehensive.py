#!/usr/bin/env python3
"""
Comprehensive Evaluation Pipeline
==================================

Master script that runs all evaluation stages and generates complete report.

Usage:
    python scripts/eval_comprehensive.py \
        --checkpoint runs/exp001/checkpoints/best_model.pt \
        --config experiments/ultimate_novel_subj01.yaml \
        --output-dir experimental_results/exp001/evaluation \
        --run-all

Author: Research-grade evaluation suite
Date: January 2026
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import Dict
import logging

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fmri2img.eval.report_generation import generate_full_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_stage1_embedding(checkpoint: str, config: str, output_dir: str, args):
    """Run Stage 1 embedding evaluation."""
    logger.info("=" * 60)
    logger.info("STAGE 1: EMBEDDING EVALUATION")
    logger.info("=" * 60)
    
    import subprocess
    
    cmd = [
        'python3', 'scripts/eval_stage1_embeddings.py',
        '--checkpoint', checkpoint,
        '--config', config,
        '--output-dir', output_dir,
        '--split', args.split,
        '--seed', str(args.seed),
    ]
    
    if args.num_samples:
        cmd.extend(['--num-samples', str(args.num_samples)])
    
    if args.gallery_sizes:
        cmd.extend(['--gallery-sizes'] + [str(s) for s in args.gallery_sizes])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Stage 1 embedding evaluation failed:\n{result.stderr}")
        return None
    
    logger.info(result.stdout)
    
    # Load results
    metrics_path = Path(output_dir) / f"embedding_metrics_{args.split}.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    
    return None


def run_stage1_probabilistic(checkpoint: str, config: str, output_dir: str, args):
    """Run Stage 1 probabilistic evaluation."""
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1: PROBABILISTIC EVALUATION (NOVEL)")
    logger.info("=" * 60)
    
    import subprocess
    
    cmd = [
        'python3', 'scripts/eval_stage1_probabilistic.py',
        '--checkpoint', checkpoint,
        '--config', config,
        '--output-dir', output_dir,
        '--split', args.split,
        '--mc-samples', str(args.mc_samples),
        '--seed', str(args.seed),
    ]
    
    if args.num_samples:
        cmd.extend(['--num-samples', str(args.num_samples)])
    
    if args.run_conformal:
        cmd.append('--run-conformal')
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Stage 1 probabilistic evaluation failed:\n{result.stderr}")
        return None
    
    logger.info(result.stdout)
    
    # Load results
    metrics_path = Path(output_dir) / f"probabilistic_metrics_{args.split}.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation pipeline for two-stage probabilistic model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stages
  python scripts/eval_comprehensive.py \\
      --checkpoint runs/exp001/checkpoints/best_model.pt \\
      --config experiments/ultimate_novel_subj01.yaml \\
      --output-dir experimental_results/exp001/evaluation \\
      --run-all

  # Run only Stage 1 with specific options
  python scripts/eval_comprehensive.py \\
      --checkpoint runs/exp001/checkpoints/best_model.pt \\
      --config experiments/ultimate_novel_subj01.yaml \\
      --output-dir experimental_results/exp001/evaluation \\
      --stage1-embedding \\
      --stage1-probabilistic \\
      --gallery-sizes 10 100 1000 \\
      --mc-samples 128
        """
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to experiment config')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    
    # Evaluation stages
    parser.add_argument('--run-all', action='store_true',
                        help='Run all evaluation stages')
    parser.add_argument('--stage1-embedding', action='store_true',
                        help='Run Stage 1 embedding evaluation')
    parser.add_argument('--stage1-probabilistic', action='store_true',
                        help='Run Stage 1 probabilistic evaluation')
    
    # Common options
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    # Stage 1 embedding options
    parser.add_argument('--gallery-sizes', type=int, nargs='+', default=[10, 100, 1000])
    
    # Stage 1 probabilistic options
    parser.add_argument('--mc-samples', type=int, default=64)
    parser.add_argument('--run-conformal', action='store_true')
    
    # Reporting
    parser.add_argument('--generate-report', action='store_true',
                        help='Generate comprehensive report with plots')
    
    args = parser.parse_args()
    
    # Determine which stages to run
    run_embedding = args.run_all or args.stage1_embedding
    run_probabilistic = args.run_all or args.stage1_probabilistic
    
    if not any([run_embedding, run_probabilistic]):
        logger.error("No evaluation stages selected. Use --run-all or specify individual stages.")
        return
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run evaluations
    results = {}
    
    if run_embedding:
        embedding_metrics = run_stage1_embedding(
            args.checkpoint, args.config, args.output_dir, args
        )
        if embedding_metrics:
            results['embedding_metrics'] = embedding_metrics
    
    if run_probabilistic:
        probabilistic_metrics = run_stage1_probabilistic(
            args.checkpoint, args.config, args.output_dir, args
        )
        if probabilistic_metrics:
            results['probabilistic_metrics'] = probabilistic_metrics
    
    # Generate comprehensive report
    if args.generate_report or args.run_all:
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING COMPREHENSIVE REPORT")
        logger.info("=" * 60)
        
        generate_full_report(
            embedding_metrics=results.get('embedding_metrics'),
            probabilistic_metrics=results.get('probabilistic_metrics'),
            output_dir=Path(args.output_dir),
            experiment_name=Path(args.checkpoint).parent.parent.name
        )
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("\nGenerated files:")
    
    output_path = Path(args.output_dir)
    for f in output_path.rglob('*'):
        if f.is_file():
            logger.info(f"  - {f.relative_to(output_path)}")
    
    logger.info("\n✅ Done!")


if __name__ == '__main__':
    main()
