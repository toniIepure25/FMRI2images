#!/usr/bin/env python3
"""
Stage 1 Probabilistic Evaluation CLI
=====================================

Evaluates probabilistic fMRI → CLIP model using novel Bayesian metrics.

Usage:
    python scripts/eval_stage1_probabilistic.py \
        --checkpoint runs/exp001/checkpoints/best_model.pt \
        --config experiments/ultimate_novel_subj01.yaml \
        --output-dir experimental_results/exp001/evaluation \
        --mc-samples 64 \
        --run-conformal

Author: Research-grade probabilistic evaluation
Date: January 2026
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Tuple
import logging

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fmri2img.models.encoders import load_probabilistic_encoder
from fmri2img.data.torch_dataset import NSDIterableDataset
from fmri2img.data.preprocess import NSDPreprocessor
from fmri2img.eval.probabilistic_eval import evaluate_probabilistic, ProbabilisticEvalResults

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_probabilistic_outputs(
    model: torch.nn.Module,
    dataloader,
    device: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract μ, logvar, and ground truth embeddings.
    
    Returns:
        mu (N, D), logvar (N, D), z_gt (N, D)
    """
    logger.info("Extracting probabilistic outputs...")
    
    all_mu = []
    all_logvar = []
    all_gt = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            # Handle different batch formats
            if isinstance(batch, dict):
                # Dictionary format: {'fmri': ..., 'clip': ..., ...}
                fmri = batch['fmri']
                clip_gt = batch.get('clip', batch.get('clip_gt', batch.get('image_clip')))
            elif isinstance(batch, (list, tuple)):
                # Tuple/list format: (fmri, clip_gt, ...)
                if len(batch) >= 2:
                    fmri = batch[0]
                    clip_gt = batch[1]
                else:
                    raise ValueError(f"Batch tuple must have at least 2 elements, got {len(batch)}")
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")
            
            fmri = fmri.to(device)
            clip_gt = clip_gt.to(device)
            
            outputs, _ = model(fmri, sample=False, return_kl=False)
            
            # Get final output
            final_out = outputs['final']
            
            all_mu.append(final_out.mu.cpu().numpy())
            all_logvar.append(final_out.logvar.cpu().numpy())
            all_gt.append(clip_gt.cpu().numpy())
    
    mu = np.concatenate(all_mu, axis=0)
    logvar = np.concatenate(all_logvar, axis=0)
    z_gt = np.concatenate(all_gt, axis=0)
    
    logger.info(f"Extracted: μ {mu.shape}, logvar {logvar.shape}, GT {z_gt.shape}")
    
    return mu, logvar, z_gt


def save_probabilistic_results(
    results: ProbabilisticEvalResults,
    output_dir: Path,
    checkpoint_path: Path,
    split: str,
    args
):
    """Save probabilistic evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_path = output_dir / f"probabilistic_metrics_{split}.json"
    metrics_dict = results.to_dict()
    metrics_dict['metadata'] = {
        'checkpoint': str(checkpoint_path),
        'config': str(args.config),
        'split': split,
        'mc_samples': args.mc_samples,
        'run_conformal': args.run_conformal,
        'seed': args.seed,
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    logger.info(f"✓ Saved metrics to {metrics_path}")
    
    # Create markdown summary
    summary_path = output_dir / f"probabilistic_summary_{split}.md"
    
    with open(summary_path, 'w') as f:
        f.write(f"# Stage 1 Probabilistic Evaluation Summary\n\n")
        f.write(f"**Split**: {split}  \n")
        f.write(f"**MC Samples**: {args.mc_samples}  \n")
        f.write(f"**Checkpoint**: `{checkpoint_path.name}`  \n\n")
        
        f.write(f"## Proper Scoring Rules\n\n")
        f.write(f"| Metric | Value | Interpretation |\n")
        f.write(f"|--------|-------|----------------|\n")
        f.write(f"| NLL (mean) | {results.nll_mean:.2f} ± {results.nll_std:.2f} | Lower is better |\n")
        f.write(f"| Energy Score | {results.energy_score_mean:.3f} ± {results.energy_score_std:.3f} | Lower is better |\n\n")
        
        f.write(f"## Bayesian Retrieval (Novel)\n\n")
        f.write(f"| Metric | Bayesian | Cosine (baseline) | Improvement |\n")
        f.write(f"|--------|----------|-------------------|-------------|\n")
        f.write(f"| Top-1 | {results.bayesian_top1:.3f} | {results.cosine_top1:.3f} | {results.improvement_top1:+.3f} |\n")
        f.write(f"| Top-5 | {results.bayesian_top5:.3f} | - | - |\n")
        f.write(f"| Top-10 | {results.bayesian_top10:.3f} | - | - |\n")
        f.write(f"| Mean Rank | {results.bayesian_mean_rank:.1f} | - | - |\n")
        f.write(f"| MRR | {results.bayesian_mrr:.3f} | - | - |\n\n")
        
        if results.improvement_top1 > 0:
            f.write(f"✅ **Bayesian retrieval improves** over cosine baseline by {results.improvement_top1:.1%}.\n\n")
        else:
            f.write(f"⚠️ **Bayesian retrieval** does not improve over baseline. Consider tuning uncertainty estimation.\n\n")
        
        f.write(f"## Probabilistic 2AFC\n\n")
        f.write(f"| Metric | Value | Chance |\n")
        f.write(f"|--------|-------|--------|\n")
        f.write(f"| Mean P(correct) | {results.prob_2afc_mean:.3f} | 0.500 |\n")
        f.write(f"| Accuracy (P>0.5) | {results.prob_2afc_accuracy:.3f} | 0.500 |\n\n")
        
        f.write(f"## Calibration Analysis\n\n")
        f.write(f"| Nominal Coverage | Empirical Coverage | Deviation |\n")
        f.write(f"|------------------|--------------------|-----------|\n")
        for nom, emp in results.calibration_coverage.items():
            dev = abs(emp - nom)
            status = "✅" if dev < 0.05 else "⚠️" if dev < 0.10 else "❌"
            f.write(f"| {nom:.0%} | {emp:.1%} | {dev:.1%} {status} |\n")
        f.write(f"\n**Calibration Error**: {results.calibration_error:.3f}\n\n")
        
        if results.calibration_error < 0.05:
            f.write(f"✅ **Well-calibrated**: calibration error < 5%.\n\n")
        elif results.calibration_error < 0.10:
            f.write(f"⚠️ **Moderately calibrated**: calibration error < 10%.\n\n")
        else:
            f.write(f"❌ **Poorly calibrated**: calibration error > 10%. Model uncertainty is unreliable.\n\n")
        
        f.write(f"## Risk-Coverage Curve\n\n")
        f.write(f"| Coverage | Risk (Error) |\n")
        f.write(f"|----------|-------------|\n")
        for cov, risk in results.risk_at_coverage.items():
            f.write(f"| {cov:.0%} | {risk:.3f} |\n")
        f.write(f"\n**AURC**: {results.aurc:.3f} (lower is better)\n\n")
        
        if results.conformal_coverage is not None:
            f.write(f"## Conformal Prediction (Distribution-Free)\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Empirical Coverage | {results.conformal_coverage:.1%} |\n")
            f.write(f"| Target Coverage | 90% |\n")
            f.write(f"| Avg Set Size | {results.conformal_avg_set_size:.1f} |\n\n")
        
        f.write(f"## Key Takeaways\n\n")
        
        # Generate takeaways
        takeaways = []
        
        if results.bayesian_top1 > results.cosine_top1:
            takeaways.append(f"✅ Bayesian retrieval outperforms cosine baseline")
        
        if results.calibration_error < 0.05:
            takeaways.append(f"✅ Model uncertainties are well-calibrated")
        else:
            takeaways.append(f"⚠️ Calibration needs improvement")
        
        if results.prob_2afc_mean > 0.60:
            takeaways.append(f"✅ Probabilistic 2AFC is strong (>{results.prob_2afc_mean:.0%})")
        
        for takeaway in takeaways:
            f.write(f"- {takeaway}\n")
        f.write("\n")
    
    logger.info(f"✓ Saved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 1 probabilistic predictions with Bayesian metrics"
    )
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--mc-samples', type=int, default=64,
                        help='Number of MC samples for Energy Score')
    parser.add_argument('--run-conformal', action='store_true',
                        help='Run conformal prediction analysis (requires more samples)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model, meta = load_probabilistic_encoder(args.checkpoint, map_location=device)
    model = model.to(device)  # Explicitly move to device
    model.eval()
    
    # Prepare dataset (reuse logic from eval_stage1_embeddings.py)
    subject = config['data']['subject']
    preprocessor = None
    preprocessor_dir = Path(f"outputs/preproc/{subject}")
    if preprocessor_dir.exists():
        preprocessor = NSDPreprocessor(subject=subject, out_dir="outputs/preproc")
        if preprocessor.load_artifacts():
            logger.info("✓ Preprocessing loaded")
    
    index_path = config['data']['index_path']
    import pandas as pd
    if index_path.endswith('.parquet'):
        index_df = pd.read_parquet(index_path)
    else:
        index_df = pd.read_csv(index_path)
    
    if args.split == 'val':
        val_size = int(len(index_df) * 0.1)
        index_df = index_df.iloc[-val_size:]
    
    if args.num_samples and args.num_samples < len(index_df):
        index_df = index_df.iloc[:args.num_samples]
    
    logger.info(f"Dataset: {len(index_df)} samples")
    
    dataset = NSDIterableDataset(
        index_path_or_root=index_path,
        subject=subject,
        shuffle=False,
        limit=len(index_df),
        preprocessor=preprocessor,
        clip_cache=config['data']['clip_cache_path']
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        num_workers=4,
        pin_memory=(device == 'cuda')
    )
    
    # Extract probabilistic outputs
    mu, logvar, z_gt = extract_probabilistic_outputs(model, loader, device)
    
    # Run probabilistic evaluation
    logger.info("Running probabilistic evaluation...")
    results = evaluate_probabilistic(
        mu=mu,
        logvar=logvar,
        z_gt=z_gt,
        n_mc_samples=args.mc_samples,
        run_conformal=args.run_conformal,
        seed=args.seed
    )
    
    logger.info("Evaluation complete!")
    
    # Print summary
    print("\n" + "="*60)
    print("STAGE 1 PROBABILISTIC EVALUATION RESULTS")
    print("="*60)
    print(f"NLL:                {results.nll_mean:.2f} ± {results.nll_std:.2f}")
    print(f"Energy Score:       {results.energy_score_mean:.3f} ± {results.energy_score_std:.3f}")
    print(f"Bayesian Top-1:     {results.bayesian_top1:.1%}")
    print(f"Cosine Top-1:       {results.cosine_top1:.1%}")
    print(f"Improvement:        {results.improvement_top1:+.1%}")
    print(f"Prob 2AFC:          {results.prob_2afc_accuracy:.1%}")
    print(f"Calibration Error:  {results.calibration_error:.3f}")
    print(f"AURC:               {results.aurc:.3f}")
    print("="*60 + "\n")
    
    # Save results
    save_probabilistic_results(results, Path(args.output_dir), Path(args.checkpoint), args.split, args)
    
    logger.info("✓ Done! Results saved to " + args.output_dir)


if __name__ == '__main__':
    main()
