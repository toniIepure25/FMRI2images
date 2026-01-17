"""
Probabilistic Evaluation Metrics for Uncertainty Quantification

Implements proper scoring rules and calibration tests for probabilistic predictions:
- NLL per-dimension and ΔNLL vs baseline
- Energy Score (sampling-based proper scoring rule)
- Calibration: coverage vs nominal (chi-square thresholds)
- Expected Calibration Error (ECE)
- Risk-coverage curves and Area Under Risk-Coverage (AURC)
- Probabilistic 2AFC using likelihood ratios
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional
from scipy.stats import chi2
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProbabilisticMetrics:
    """Container for probabilistic metrics."""
    nll_per_dim: float
    delta_nll: float
    energy_score: float
    coverage_80: float
    coverage_95: float
    ece: float
    aurc: float
    prob_2afc_accuracy: float
    
    def to_dict(self) -> Dict:
        return {
            "nll_per_dim": self.nll_per_dim,
            "delta_nll": self.delta_nll,
            "energy_score": self.energy_score,
            "coverage_80": self.coverage_80,
            "coverage_95": self.coverage_95,
            "ece": self.ece,
            "aurc": self.aurc,
            "prob_2afc_accuracy": self.prob_2afc_accuracy,
        }


def compute_nll_per_dim(
    mu: np.ndarray,
    logvar: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Compute negative log-likelihood per dimension.
    
    Args:
        mu: (N, D) predicted mean
        logvar: (N, D) predicted log-variance
        target: (N, D) ground truth
        
    Returns:
        NLL per dimension (scalar)
    """
    err = target - mu
    nll = 0.5 * (logvar + (err ** 2) / np.exp(logvar))
    return nll.mean()


def compute_delta_nll(
    mu: np.ndarray,
    logvar: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Compute ΔNLL vs baseline (empirical variance).
    
    Args:
        mu: (N, D) predicted mean
        logvar: (N, D) predicted log-variance
        target: (N, D) ground truth
        
    Returns:
        ΔNLL (negative = better than baseline)
    """
    # Model NLL
    err = target - mu
    model_nll = 0.5 * (logvar + (err ** 2) / np.exp(logvar))
    model_nll = model_nll.mean()
    
    # Baseline: constant prediction with empirical variance
    baseline_mean = target.mean(axis=0)
    baseline_var = target.var(axis=0, ddof=1)
    baseline_logvar = np.log(baseline_var + 1e-8)
    
    baseline_err = target - baseline_mean
    baseline_nll = 0.5 * (baseline_logvar + (baseline_err ** 2) / np.exp(baseline_logvar))
    baseline_nll = baseline_nll.mean()
    
    return model_nll - baseline_nll


def compute_energy_score(
    samples: np.ndarray,
    target: np.ndarray,
    n_samples: int = 64,
) -> float:
    """
    Compute Energy Score (proper scoring rule for multivariate distributions).
    
    ES = E[||Y - X||] - 0.5 * E[||Y - Y'||]
    where Y, Y' are independent samples from predicted distribution, X is ground truth.
    
    Args:
        samples: (N, S, D) samples from predicted distribution
        target: (N, D) ground truth
        n_samples: Number of samples to use (if samples has more)
        
    Returns:
        Energy Score (lower is better)
    """
    N, S, D = samples.shape
    
    # Subsample if needed
    if S > n_samples:
        idx = np.random.choice(S, n_samples, replace=False)
        samples = samples[:, idx, :]
        S = n_samples
    
    # Expand target for broadcasting
    target_exp = target[:, np.newaxis, :]  # (N, 1, D)
    
    # E[||Y - X||]
    dist_to_target = np.linalg.norm(samples - target_exp, axis=2)  # (N, S)
    term1 = dist_to_target.mean()
    
    # E[||Y - Y'||] using pairwise distances
    # For efficiency, sample pairs
    n_pairs = min(S * (S - 1) // 2, 1000)
    pairwise_dists = []
    
    for _ in range(n_pairs):
        i, j = np.random.choice(S, 2, replace=False)
        dist = np.linalg.norm(samples[:, i, :] - samples[:, j, :], axis=1)
        pairwise_dists.append(dist.mean())
    
    term2 = 0.5 * np.mean(pairwise_dists)
    
    energy_score = term1 - term2
    
    return energy_score


def compute_calibration_coverage(
    mu: np.ndarray,
    logvar: np.ndarray,
    target: np.ndarray,
    nominal_levels: list = [0.80, 0.95],
) -> Dict[str, float]:
    """
    Compute calibration: empirical coverage vs nominal using chi-square thresholds.
    
    For diagonal Gaussian, Mahalanobis distance follows chi-square distribution:
    D^2 = sum((X - mu)^2 / var) ~ chi^2(D)
    
    Args:
        mu: (N, D) predicted mean
        logvar: (N, D) predicted log-variance
        target: (N, D) ground truth
        nominal_levels: List of nominal coverage levels (e.g., [0.80, 0.95])
        
    Returns:
        Dictionary mapping nominal level to empirical coverage
    """
    N, D = mu.shape
    
    # Compute Mahalanobis distance
    var = np.exp(logvar)
    mahal_sq = ((target - mu) ** 2 / var).sum(axis=1)  # (N,)
    
    # Chi-square thresholds
    coverages = {}
    for level in nominal_levels:
        threshold = chi2.ppf(level, df=D)
        empirical_coverage = (mahal_sq <= threshold).mean()
        coverages[f"coverage_{int(level*100)}"] = empirical_coverage
    
    return coverages


def compute_ece(
    mu: np.ndarray,
    logvar: np.ndarray,
    target: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Bins predictions by confidence (inverse variance) and compares
    predicted vs empirical accuracy in each bin.
    
    Args:
        mu: (N, D) predicted mean
        logvar: (N, D) predicted log-variance
        target: (N, D) ground truth
        n_bins: Number of bins for calibration
        
    Returns:
        ECE (lower is better)
    """
    N, D = mu.shape
    
    # Use average variance as confidence proxy
    var = np.exp(logvar)
    confidence = 1.0 / (var.mean(axis=1) + 1e-8)  # (N,)
    
    # Compute errors
    errors = np.linalg.norm(target - mu, axis=1)  # (N,)
    
    # Bin by confidence
    confidence_bins = np.linspace(confidence.min(), confidence.max(), n_bins + 1)
    
    ece = 0.0
    total_count = 0
    
    for i in range(n_bins):
        bin_mask = (confidence >= confidence_bins[i]) & (confidence < confidence_bins[i + 1])
        bin_count = bin_mask.sum()
        
        if bin_count == 0:
            continue
        
        bin_confidence = confidence[bin_mask].mean()
        bin_error = errors[bin_mask].mean()
        
        # Calibration error: |confidence - (1 - normalized_error)|
        # Normalize error to [0, 1] range (roughly)
        max_error = errors.max()
        normalized_error = bin_error / (max_error + 1e-8)
        bin_accuracy = 1.0 - normalized_error
        
        # Weight by bin size
        ece += (bin_count / N) * abs(bin_confidence - bin_accuracy)
        total_count += bin_count
    
    return ece


def compute_risk_coverage_curve(
    mu: np.ndarray,
    logvar: np.ndarray,
    target: np.ndarray,
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute risk-coverage curve for selective prediction.
    
    Rank samples by uncertainty (variance) and compute risk (error) vs coverage.
    Well-calibrated uncertainty should achieve lower risk at high confidence.
    
    Args:
        mu: (N, D) predicted mean
        logvar: (N, D) predicted log-variance
        target: (N, D) ground truth
        n_points: Number of points on curve
        
    Returns:
        (coverage, risk) arrays
    """
    N = len(mu)
    
    # Compute uncertainty (average variance)
    var = np.exp(logvar)
    uncertainty = var.mean(axis=1)  # (N,)
    
    # Compute errors
    errors = np.linalg.norm(target - mu, axis=1)  # (N,)
    
    # Sort by uncertainty (ascending = most confident first)
    sorted_indices = np.argsort(uncertainty)
    sorted_errors = errors[sorted_indices]
    
    # Compute cumulative risk at different coverage levels
    coverage_levels = np.linspace(0, 1, n_points)
    risks = []
    
    for coverage in coverage_levels:
        if coverage == 0:
            risks.append(0)
        else:
            n_covered = int(coverage * N)
            risk = sorted_errors[:n_covered].mean()
            risks.append(risk)
    
    return coverage_levels, np.array(risks)


def compute_aurc(
    mu: np.ndarray,
    logvar: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Compute Area Under Risk-Coverage curve (AURC).
    
    Lower AURC indicates better uncertainty calibration for selective prediction.
    
    Args:
        mu: (N, D) predicted mean
        logvar: (N, D) predicted log-variance
        target: (N, D) ground truth
        
    Returns:
        AURC (lower is better)
    """
    coverage, risk = compute_risk_coverage_curve(mu, logvar, target)
    
    # Compute area using trapezoidal rule
    aurc = np.trapz(risk, coverage)
    
    return aurc


def compute_probabilistic_2afc(
    mu_a: np.ndarray,
    logvar_a: np.ndarray,
    mu_b: np.ndarray,
    logvar_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    n_pairs: int = 5000,
    seed: int = 42,
) -> float:
    """
    Compute probabilistic 2AFC using likelihood ratios.
    
    For each pair of predictions, choose the one with higher likelihood
    under the predicted distribution.
    
    Args:
        mu_a: (N, D) predicted mean for set A
        logvar_a: (N, D) predicted log-variance for set A
        mu_b: (M, D) predicted mean for set B
        logvar_b: (M, D) predicted log-variance for set B
        target_a: (N, D) targets for set A
        target_b: (M, D) targets for set B
        labels_a: (N,) labels for set A
        labels_b: (M,) labels for set B
        n_pairs: Number of pairs to sample
        seed: Random seed
        
    Returns:
        Probabilistic 2AFC accuracy
    """
    rng = np.random.RandomState(seed)
    
    N = len(mu_a)
    M = len(mu_b)
    
    # Sample pairs
    idx_a = rng.randint(0, N, size=n_pairs)
    idx_b = rng.randint(0, M, size=n_pairs)
    
    # Compute log-likelihoods
    # log N(x; mu, var) = -0.5 * (log(2*pi) + logvar + (x - mu)^2 / var)
    # Omit constant for comparison
    
    def log_likelihood(x, mu, logvar):
        var = np.exp(logvar)
        return -0.5 * (logvar + (x - mu) ** 2 / var).sum(axis=1)
    
    # Match A: likelihood of target_a under prediction_a
    loglik_match_a = log_likelihood(target_a[idx_a], mu_a[idx_a], logvar_a[idx_a])
    
    # Match B: likelihood of target_b under prediction_b
    loglik_match_b = log_likelihood(target_b[idx_b], mu_b[idx_b], logvar_b[idx_b])
    
    # Cross: likelihood of target_b under prediction_a
    loglik_cross_ab = log_likelihood(target_b[idx_b], mu_a[idx_a], logvar_a[idx_a])
    
    # Cross: likelihood of target_a under prediction_b
    loglik_cross_ba = log_likelihood(target_a[idx_a], mu_b[idx_b], logvar_b[idx_b])
    
    # For each pair, check if labels match
    is_match = labels_a[idx_a] == labels_b[idx_b]
    
    # Probabilistic 2AFC: choose based on likelihood
    # If match: correct if loglik_match_a > loglik_cross_ab or loglik_match_b > loglik_cross_ba
    # If not match: correct if loglik_match_a > loglik_cross_ab or loglik_match_b > loglik_cross_ba
    
    # Simplified: use average of both directions
    score_match = (loglik_match_a + loglik_match_b) / 2
    score_cross = (loglik_cross_ab + loglik_cross_ba) / 2
    
    correct = (score_match > score_cross) == is_match
    
    return correct.mean()


class ProbabilisticEvaluator:
    """
    Unified evaluator for probabilistic metrics.
    
    Usage:
        evaluator = ProbabilisticEvaluator()
        results = evaluator.evaluate(
            mu=pred_mu,
            logvar=pred_logvar,
            target=gt_embeddings,
            samples=samples,  # Optional: (N, S, D) samples
        )
    """
    
    def __init__(
        self,
        n_samples_energy: int = 64,
        n_calibration_bins: int = 10,
        seed: int = 42,
    ):
        self.n_samples_energy = n_samples_energy
        self.n_calibration_bins = n_calibration_bins
        self.seed = seed
    
    def evaluate(
        self,
        mu: np.ndarray,
        logvar: np.ndarray,
        target: np.ndarray,
        samples: Optional[np.ndarray] = None,
        ids: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Run full probabilistic evaluation.
        
        Args:
            mu: (N, D) predicted mean
            logvar: (N, D) predicted log-variance
            target: (N, D) ground truth
            samples: Optional (N, S, D) samples from distribution
            ids: Optional (N,) sample IDs for 2AFC
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Running probabilistic evaluation...")
        
        results = {}
        
        # 1. NLL per-dim
        logger.info("  Computing NLL per-dim...")
        nll_per_dim = compute_nll_per_dim(mu, logvar, target)
        results["nll_per_dim"] = float(nll_per_dim)
        
        # 2. ΔNLL
        logger.info("  Computing ΔNLL vs baseline...")
        delta_nll = compute_delta_nll(mu, logvar, target)
        results["delta_nll"] = float(delta_nll)
        
        # 3. Energy Score (if samples provided)
        if samples is not None:
            logger.info("  Computing Energy Score...")
            energy_score = compute_energy_score(samples, target, self.n_samples_energy)
            results["energy_score"] = float(energy_score)
        else:
            logger.warning("  Samples not provided, skipping Energy Score")
            results["energy_score"] = None
        
        # 4. Calibration coverage
        logger.info("  Computing calibration coverage...")
        coverages = compute_calibration_coverage(mu, logvar, target)
        results.update(coverages)
        
        # 5. ECE
        logger.info("  Computing ECE...")
        ece = compute_ece(mu, logvar, target, n_bins=self.n_calibration_bins)
        results["ece"] = float(ece)
        
        # 6. AURC
        logger.info("  Computing AURC...")
        aurc = compute_aurc(mu, logvar, target)
        results["aurc"] = float(aurc)
        
        # 7. Probabilistic 2AFC (if IDs provided)
        if ids is not None:
            logger.info("  Computing probabilistic 2AFC...")
            prob_2afc = compute_probabilistic_2afc(
                mu, logvar, mu, logvar,
                target, target,
                ids, ids,
                seed=self.seed,
            )
            results["prob_2afc_accuracy"] = float(prob_2afc)
        else:
            results["prob_2afc_accuracy"] = None
        
        logger.info("Probabilistic evaluation complete.")
        
        return results
