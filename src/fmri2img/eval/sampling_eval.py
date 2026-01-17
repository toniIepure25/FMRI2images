"""
End-to-End Two-Stage Sampling Evaluation
=========================================

Evaluates the benefit of sampling multiple embeddings from q(z|x) and
generating multiple reconstructions.

Key Question: Does uncertainty help? Can we improve reconstruction quality
by generating multiple samples and selecting the best one?

Metrics:
  - Expected-of-N: Average quality across N samples
  - Best-of-N: Best quality among N samples
  - Diversity: Variance among samples

Evaluated for N ∈ {1, 2, 4, 8, 16, ...}

Scientific Context:
- Bayesian decision theory: Use full posterior, not just MAP
- Risk-averse decoding: Select best sample post-hoc
- Similar to nucleus sampling in language generation (Holtzman et al. 2020)

References:
- Holtzman et al. (2020). "The Curious Case of Neural Text Degeneration"
- Kuleshov et al. (2018). "Accurate Uncertainties for Deep Learning Using Calibrated Regression"

Author: Research-grade end-to-end evaluation
Date: January 2026
"""

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SamplingEvalResults:
    """Container for sampling evaluation results."""
    
    # Quality vs N samples
    n_samples: List[int]
    
    # Expected-of-N: Average quality
    expected_pixcorr: List[float]
    expected_lpips: List[float]
    expected_clip_sim: List[float]
    
    # Best-of-N: Best quality among samples
    best_pixcorr: List[float]
    best_lpips: List[float]
    best_clip_sim: List[float]
    
    # Diversity
    diversity_pixcorr: List[float]  # Std among samples
    diversity_lpips: List[float]
    diversity_clip_sim: List[float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'n_samples': self.n_samples,
            'expected_of_n': {
                'pixcorr': [float(x) for x in self.expected_pixcorr],
                'lpips': [float(x) for x in self.expected_lpips],
                'clip_sim': [float(x) for x in self.expected_clip_sim],
            },
            'best_of_n': {
                'pixcorr': [float(x) for x in self.best_pixcorr],
                'lpips': [float(x) for x in self.best_lpips],
                'clip_sim': [float(x) for x in self.best_clip_sim],
            },
            'diversity': {
                'pixcorr_std': [float(x) for x in self.diversity_pixcorr],
                'lpips_std': [float(x) for x in self.diversity_lpips],
                'clip_sim_std': [float(x) for x in self.diversity_clip_sim],
            }
        }


def evaluate_sampling(
    reconstruction_quality_fn: Callable[[np.ndarray, np.ndarray], Dict[str, float]],
    imgs_gt: List[np.ndarray],
    samples_per_input: List[List[np.ndarray]],
    n_values: Optional[List[int]] = None
) -> SamplingEvalResults:
    """
    Evaluate expected-of-N vs best-of-N reconstruction quality.
    
    For each input:
      - We have S reconstructed images (from S embedding samples)
      - For N in [1, 2, 4, ...]:
          - Expected-of-N: average quality of random N samples
          - Best-of-N: best quality among random N samples
    
    Args:
        reconstruction_quality_fn: Function that takes (img_recon, img_gt) and
            returns dict with metrics {'pixcorr': ..., 'lpips': ..., 'clip_sim': ...}
        imgs_gt: Ground truth images (M,)
        samples_per_input: List of lists, samples_per_input[i] = [S reconstructions for input i]
        n_values: N values to test (default: [1, 2, 4, 8])
    
    Returns:
        SamplingEvalResults with quality vs N
    """
    if n_values is None:
        n_values = [1, 2, 4, 8]
    
    M = len(imgs_gt)
    S = len(samples_per_input[0])  # Number of samples per input
    
    logger.info(f"Evaluating sampling: M={M} inputs, S={S} samples per input")
    
    # Compute quality for all samples
    # qualities[i][s] = quality of sample s for input i
    qualities = []
    for i in range(M):
        input_qualities = []
        for s in range(S):
            q = reconstruction_quality_fn(samples_per_input[i][s], imgs_gt[i])
            input_qualities.append(q)
        qualities.append(input_qualities)
    
    # For each N
    results = {
        'expected_pixcorr': [],
        'expected_lpips': [],
        'expected_clip_sim': [],
        'best_pixcorr': [],
        'best_lpips': [],
        'best_clip_sim': [],
        'diversity_pixcorr': [],
        'diversity_lpips': [],
        'diversity_clip_sim': [],
    }
    
    rng = np.random.RandomState(42)
    
    for N in n_values:
        if N > S:
            logger.warning(f"N={N} > S={S}, skipping")
            continue
        
        # For each input, sample N reconstructions (repeat trials for robustness)
        n_trials = 100
        expected_pc = []
        expected_lp = []
        expected_cs = []
        best_pc = []
        best_lp = []
        best_cs = []
        div_pc = []
        div_lp = []
        div_cs = []
        
        for i in range(M):
            trial_expected_pc = []
            trial_expected_lp = []
            trial_expected_cs = []
            trial_best_pc = []
            trial_best_lp = []
            trial_best_cs = []
            trial_div_pc = []
            trial_div_lp = []
            trial_div_cs = []
            
            for _ in range(n_trials):
                # Sample N reconstructions
                indices = rng.choice(S, size=N, replace=False)
                sampled_qualities = [qualities[i][idx] for idx in indices]
                
                # Extract metrics
                pcs = [q['pixcorr'] for q in sampled_qualities]
                lps = [q['lpips'] for q in sampled_qualities]
                css = [q['clip_sim'] for q in sampled_qualities]
                
                # Expected-of-N: average
                trial_expected_pc.append(np.mean(pcs))
                trial_expected_lp.append(np.mean(lps))
                trial_expected_cs.append(np.mean(css))
                
                # Best-of-N: best quality
                trial_best_pc.append(np.max(pcs))
                trial_best_lp.append(np.min(lps))  # LPIPS: lower is better
                trial_best_cs.append(np.max(css))
                
                # Diversity: std among N samples
                trial_div_pc.append(np.std(pcs))
                trial_div_lp.append(np.std(lps))
                trial_div_cs.append(np.std(css))
            
            # Average across trials for this input
            expected_pc.append(np.mean(trial_expected_pc))
            expected_lp.append(np.mean(trial_expected_lp))
            expected_cs.append(np.mean(trial_expected_cs))
            best_pc.append(np.mean(trial_best_pc))
            best_lp.append(np.mean(trial_best_lp))
            best_cs.append(np.mean(trial_best_cs))
            div_pc.append(np.mean(trial_div_pc))
            div_lp.append(np.mean(trial_div_lp))
            div_cs.append(np.mean(trial_div_cs))
        
        # Average across all inputs
        results['expected_pixcorr'].append(np.mean(expected_pc))
        results['expected_lpips'].append(np.mean(expected_lp))
        results['expected_clip_sim'].append(np.mean(expected_cs))
        results['best_pixcorr'].append(np.mean(best_pc))
        results['best_lpips'].append(np.mean(best_lp))
        results['best_clip_sim'].append(np.mean(best_cs))
        results['diversity_pixcorr'].append(np.mean(div_pc))
        results['diversity_lpips'].append(np.mean(div_lp))
        results['diversity_clip_sim'].append(np.mean(div_cs))
    
    return SamplingEvalResults(
        n_samples=n_values[:len(results['expected_pixcorr'])],
        expected_pixcorr=results['expected_pixcorr'],
        expected_lpips=results['expected_lpips'],
        expected_clip_sim=results['expected_clip_sim'],
        best_pixcorr=results['best_pixcorr'],
        best_lpips=results['best_lpips'],
        best_clip_sim=results['best_clip_sim'],
        diversity_pixcorr=results['diversity_pixcorr'],
        diversity_lpips=results['diversity_lpips'],
        diversity_clip_sim=results['diversity_clip_sim'],
    )


def evaluate_sampling_from_disk(
    imgs_gt_dir: str,
    recon_dirs: List[str],
    n_values: Optional[List[int]] = None,
    reconstruction_quality_fn: Optional[Callable] = None
) -> SamplingEvalResults:
    """
    Load reconstructions from disk and evaluate sampling.
    
    Directory structure:
      imgs_gt_dir/
        000000.png
        000001.png
        ...
      recon_dirs[0]/  # Sample 0
        000000.png
        000001.png
        ...
      recon_dirs[1]/  # Sample 1
        ...
    
    Args:
        imgs_gt_dir: Directory with ground truth images
        recon_dirs: List of directories, one per sample
        n_values: N values to test
        reconstruction_quality_fn: Custom quality function (default: uses basic metrics)
    
    Returns:
        SamplingEvalResults
    """
    from pathlib import Path
    from PIL import Image
    
    # Load ground truth images
    gt_path = Path(imgs_gt_dir)
    gt_files = sorted(gt_path.glob('*.png'))
    imgs_gt = []
    for f in gt_files:
        img = Image.open(f).convert('RGB')
        imgs_gt.append(np.array(img) / 255.0)
    
    logger.info(f"Loaded {len(imgs_gt)} ground truth images from {imgs_gt_dir}")
    
    # Load reconstructions
    samples_per_input = [[] for _ in range(len(imgs_gt))]
    
    for sample_idx, recon_dir in enumerate(recon_dirs):
        recon_path = Path(recon_dir)
        recon_files = sorted(recon_path.glob('*.png'))
        
        if len(recon_files) != len(imgs_gt):
            raise ValueError(
                f"Mismatch: {len(recon_files)} recons in {recon_dir} vs {len(imgs_gt)} GTs"
            )
        
        for i, f in enumerate(recon_files):
            img = Image.open(f).convert('RGB')
            samples_per_input[i].append(np.array(img) / 255.0)
        
        logger.info(f"Loaded sample {sample_idx} from {recon_dir}")
    
    # Default quality function
    if reconstruction_quality_fn is None:
        from .recon_eval import compute_pixcorr, compute_ssim, LPIPSEvaluator
        
        lpips_eval = LPIPSEvaluator() if LPIPS_AVAILABLE else None
        
        def default_quality_fn(img_recon, img_gt):
            pc = compute_pixcorr(img_recon, img_gt)
            lp = lpips_eval.compute(img_recon, img_gt) if lpips_eval else 0.0
            # Placeholder for CLIP (would need CLIP model)
            cs = 0.5
            return {'pixcorr': pc, 'lpips': lp, 'clip_sim': cs}
        
        reconstruction_quality_fn = default_quality_fn
    
    return evaluate_sampling(
        reconstruction_quality_fn,
        imgs_gt,
        samples_per_input,
        n_values
    )


def compute_oracle_best_of_n(
    imgs_gt: List[np.ndarray],
    samples_per_input: List[List[np.ndarray]],
    metric: str = 'clip_sim',
    clip_evaluator = None
) -> Dict[str, float]:
    """
    Compute "oracle" best-of-N by selecting best sample per input.
    
    For each input, evaluate all S samples and pick the best one based on metric.
    This is an upper bound on what best-of-N can achieve.
    
    Args:
        imgs_gt: Ground truth images
        samples_per_input: Reconstructions
        metric: Metric to optimize ('clip_sim', 'pixcorr', 'lpips')
        clip_evaluator: CLIP evaluator (if needed for metric)
    
    Returns:
        Dictionary with oracle best metric
    """
    from .recon_eval import compute_pixcorr, LPIPSEvaluator
    
    M = len(imgs_gt)
    S = len(samples_per_input[0])
    
    best_values = []
    
    if metric == 'pixcorr':
        for i in range(M):
            scores = [compute_pixcorr(samples_per_input[i][s], imgs_gt[i]) for s in range(S)]
            best_values.append(max(scores))
    
    elif metric == 'lpips':
        lpips_eval = LPIPSEvaluator()
        for i in range(M):
            scores = [lpips_eval.compute(samples_per_input[i][s], imgs_gt[i]) for s in range(S)]
            best_values.append(min(scores))  # Lower is better
    
    elif metric == 'clip_sim':
        if clip_evaluator is None:
            raise ValueError("clip_evaluator required for clip_sim metric")
        
        for i in range(M):
            emb_recons = clip_evaluator.encode_batch(samples_per_input[i])
            emb_gt = clip_evaluator.encode_image(imgs_gt[i])
            
            scores = np.dot(emb_recons, emb_gt)
            best_values.append(max(scores))
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return {
        f'oracle_best_{metric}': float(np.mean(best_values)),
        f'oracle_best_{metric}_std': float(np.std(best_values)),
    }
