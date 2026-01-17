"""
Probabilistic (Bayesian) Evaluation Metrics for Stage 1
========================================================

Novel evaluation metrics for probabilistic fMRI→CLIP models that predict
distributions q(z|x) instead of point estimates.

Key Contributions (Novel for neural decoding):
  1. Proper scoring rules (NLL, Energy Score)
  2. Distribution-aware ("Bayesian") retrieval
  3. Probabilistic 2AFC with uncertainty propagation
  4. Calibration analysis (reliability diagrams)
  5. Risk-coverage curves (selective prediction)
  6. Conformal prediction sets (distribution-free uncertainty)

Scientific Motivation:
- fMRI is noisy → predictions should quantify uncertainty
- Proper scoring rules incentivize well-calibrated probabilities (Gneiting & Raftery 2007)
- Bayesian retrieval uses full posterior, not just MAP estimate
- Calibration: Are 90% credible regions correct 90% of the time?
- Risk-coverage: Can we abstain on uncertain predictions?

Assumptions:
- Model outputs q(z|x) = Normal(μ, diag(σ²)) by default
- Easily extensible to low-rank or full covariance

References:
- Gneiting & Raftery (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation"
- Gneiting & Ranjan (2011). "Comparing Density Forecasts Using Threshold-Weighted Scoring Rules"
- Vovk et al. (2005). "Algorithmic Learning in a Random World" (conformal prediction)
- Geifman & El-Yaniv (2017). "Selective Prediction" (risk-coverage)
- Kendall & Gal (2017). "What Uncertainties Do We Need in Bayesian Deep Learning?"

Author: Research-grade probabilistic evaluation
Date: January 2026
"""

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import chi2
from scipy.special import logsumexp
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProbabilisticEvalResults:
    """Container for probabilistic evaluation results."""
    
    # Proper scoring rules
    nll_mean: float
    nll_std: float
    energy_score_mean: float
    energy_score_std: float
    
    # Bayesian retrieval (using log-likelihoods as scores)
    bayesian_top1: float
    bayesian_top5: float
    bayesian_top10: float
    bayesian_mean_rank: float
    bayesian_mrr: float
    
    # Compare with cosine baseline
    cosine_top1: float  # For comparison
    improvement_top1: float  # bayesian - cosine
    
    # Probabilistic 2AFC
    prob_2afc_mean: float  # Mean P(correct)
    prob_2afc_accuracy: float  # Thresholded accuracy (P > 0.5)
    
    # Calibration
    calibration_coverage: Dict[str, float]  # {50: 0.52, 80: 0.81, ...}
    calibration_error: float  # Mean absolute deviation
    
    # Risk-coverage
    aurc: float  # Area under risk-coverage curve
    risk_at_coverage: Dict[float, float]  # {0.9: error, 0.5: error}
    
    # Conformal sets (optional)
    conformal_coverage: Optional[float] = None
    conformal_avg_set_size: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'proper_scoring_rules': {
                'nll_mean': float(self.nll_mean),
                'nll_std': float(self.nll_std),
                'energy_score_mean': float(self.energy_score_mean),
                'energy_score_std': float(self.energy_score_std),
            },
            'bayesian_retrieval': {
                'top1_accuracy': float(self.bayesian_top1),
                'top5_accuracy': float(self.bayesian_top5),
                'top10_accuracy': float(self.bayesian_top10),
                'mean_rank': float(self.bayesian_mean_rank),
                'mrr': float(self.bayesian_mrr),
                'improvement_vs_cosine': float(self.improvement_top1),
            },
            'probabilistic_2afc': {
                'mean_p_correct': float(self.prob_2afc_mean),
                'accuracy': float(self.prob_2afc_accuracy),
            },
            'calibration': {
                'coverage_by_nominal': {str(k): float(v) for k, v in self.calibration_coverage.items()},
                'mean_absolute_error': float(self.calibration_error),
            },
            'risk_coverage': {
                'aurc': float(self.aurc),
                'risk_at_coverage': {str(k): float(v) for k, v in self.risk_at_coverage.items()},
            },
        }
        if self.conformal_coverage is not None:
            self.to_dict()['conformal'] = {
                'coverage': float(self.conformal_coverage),
                'avg_set_size': float(self.conformal_avg_set_size) if self.conformal_avg_set_size else None,
            }


def gaussian_nll(
    z_gt: np.ndarray,
    mu: np.ndarray,
    logvar: np.ndarray
) -> np.ndarray:
    """
    Compute negative log-likelihood of ground truth under diagonal Gaussian.
    
    NLL(z_gt | mu, σ²) = 0.5 * [D*log(2π) + Σ log(σ²) + Σ((z - μ)² / σ²)]
    
    Args:
        z_gt: Ground truth embeddings (N, D)
        mu: Predicted means (N, D)
        logvar: Predicted log-variances (N, D)
    
    Returns:
        NLL per sample (N,)
    """
    D = z_gt.shape[1]
    var = np.exp(logvar)
    
    # NLL = 0.5 * [D*log(2π) + sum(log(σ²)) + sum((z-μ)²/σ²)]
    nll = 0.5 * (
        D * np.log(2 * np.pi) +
        np.sum(logvar, axis=1) +
        np.sum((z_gt - mu) ** 2 / var, axis=1)
    )
    
    return nll


def energy_score(
    z_gt: np.ndarray,
    samples: np.ndarray
) -> np.ndarray:
    """
    Compute Energy Score (multivariate proper scoring rule).
    
    ES = E||z - z_gt|| - 0.5 * E||z - z'||
    where z, z' ~ q(·|x) are independent samples.
    
    Lower is better. Proper scoring rule that uses samples directly.
    
    Args:
        z_gt: Ground truth (N, D)
        samples: Samples from q(z|x), shape (N, S, D) where S is number of samples
    
    Returns:
        Energy score per sample (N,)
    """
    N, S, D = samples.shape
    
    # Term 1: E||z - z_gt||
    # Compute ||samples[i, s] - z_gt[i]|| for all i, s
    diffs = samples - z_gt[:, None, :]  # (N, S, D)
    norms_to_gt = np.linalg.norm(diffs, axis=2)  # (N, S)
    term1 = norms_to_gt.mean(axis=1)  # (N,)
    
    # Term 2: 0.5 * E||z - z'||
    # For each sample i, compute pairwise distances among S samples
    term2 = np.zeros(N)
    for i in range(N):
        # samples[i]: (S, D)
        # Pairwise distances
        diffs_pairwise = samples[i][:, None, :] - samples[i][None, :, :]  # (S, S, D)
        norms_pairwise = np.linalg.norm(diffs_pairwise, axis=2)  # (S, S)
        # Mean over all pairs (including diagonal which is 0)
        term2[i] = norms_pairwise.mean()
    
    term2 = 0.5 * term2
    
    es = term1 - term2
    
    return es


def bayesian_retrieval(
    mu: np.ndarray,
    logvar: np.ndarray,
    gallery: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10)
) -> Dict[str, float]:
    """
    Distribution-aware ("Bayesian") retrieval using log-likelihood scores.
    
    For each query distribution q(z|x_i) = N(μ_i, σ_i²):
      Score candidate c_k by log q(c_k | x_i)
      Rank candidates by score (higher = more likely)
    
    Novel contribution: Uses full posterior instead of just MAP (mean).
    
    Args:
        mu: Query means (N, D)
        logvar: Query log-variances (N, D)
        gallery: Candidate embeddings (N, D), assumed normalized
        ks: K values for Top-K
    
    Returns:
        Retrieval metrics using Bayesian scores
    """
    N, D = mu.shape
    var = np.exp(logvar)
    
    # For each query i, compute log q(gallery[j] | x_i) for all j
    log_scores = np.zeros((N, N))
    
    for i in range(N):
        # log q(c | x_i) = -0.5 * [D*log(2π) + sum(log σ²) + sum((c - μ)²/σ²)]
        for j in range(N):
            c = gallery[j]
            log_q = -0.5 * (
                D * np.log(2 * np.pi) +
                np.sum(logvar[i]) +
                np.sum((c - mu[i]) ** 2 / var[i])
            )
            log_scores[i, j] = log_q
    
    # Rank by log-likelihood (higher is better)
    ranks = np.argsort(-log_scores, axis=1)  # Descending
    
    # Find rank of correct match
    correct_ranks = np.zeros(N, dtype=np.int32)
    for i in range(N):
        correct_ranks[i] = np.where(ranks[i] == i)[0][0]
    
    # Compute metrics
    results = {}
    for k in ks:
        results[f'top{k}_accuracy'] = float((correct_ranks < k).mean())
    
    results['mean_rank'] = float(correct_ranks.mean() + 1)
    results['mrr'] = float((1.0 / (correct_ranks + 1.0)).mean())
    
    return results


def cosine_retrieval_from_mu(
    mu: np.ndarray,
    gallery: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10)
) -> Dict[str, float]:
    """
    Standard cosine retrieval using only means (for comparison).
    
    Args:
        mu: Query means (N, D)
        gallery: Gallery embeddings (N, D)
        ks: K values
    
    Returns:
        Retrieval metrics using cosine similarity
    """
    # Normalize
    mu_norm = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-8)
    gallery_norm = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity
    similarities = mu_norm @ gallery_norm.T
    
    ranks = np.argsort(-similarities, axis=1)
    
    N = len(mu)
    correct_ranks = np.zeros(N, dtype=np.int32)
    for i in range(N):
        correct_ranks[i] = np.where(ranks[i] == i)[0][0]
    
    results = {}
    for k in ks:
        results[f'top{k}_accuracy'] = float((correct_ranks < k).mean())
    
    results['mean_rank'] = float(correct_ranks.mean() + 1)
    results['mrr'] = float((1.0 / (correct_ranks + 1.0)).mean())
    
    return results


def probabilistic_2afc(
    mu: np.ndarray,
    logvar: np.ndarray,
    ground_truth: np.ndarray,
    n_trials: int = 1000,
    n_mc_samples: int = 64,
    seed: int = 42
) -> Dict[str, float]:
    """
    Probabilistic two-alternative forced choice.
    
    For each trial:
      1. Sample query index i and distractor j
      2. Draw S samples z_s ~ N(μ_i, σ_i²)
      3. Estimate P(correct) = P[sim(z, gt_i) > sim(z, gt_j)]
      4. Average over samples and trials
    
    Novel: Propagates uncertainty through 2AFC decision.
    
    Args:
        mu: Means (N, D)
        logvar: Log-variances (N, D)
        ground_truth: GT embeddings (N, D)
        n_trials: Number of 2AFC trials
        n_mc_samples: MC samples per trial for estimating P(correct)
        seed: Random seed
    
    Returns:
        mean_p_correct: Average P(correct) across trials
        accuracy: Fraction where P(correct) > 0.5
    """
    rng = np.random.RandomState(seed)
    N, D = mu.shape
    std = np.sqrt(np.exp(logvar))
    
    # Normalize GT
    gt_norm = ground_truth / (np.linalg.norm(ground_truth, axis=1, keepdims=True) + 1e-8)
    
    p_corrects = []
    
    for _ in range(n_trials):
        i = rng.randint(N)
        j = rng.randint(N - 1)
        if j >= i:
            j += 1
        
        # Sample from q(z | x_i)
        samples = rng.randn(n_mc_samples, D) * std[i] + mu[i]  # (S, D)
        
        # Normalize samples
        samples_norm = samples / (np.linalg.norm(samples, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarities
        sims_correct = samples_norm @ gt_norm[i]  # (S,)
        sims_distractor = samples_norm @ gt_norm[j]  # (S,)
        
        # P(correct) ≈ fraction of samples where sim_correct > sim_distractor
        p_correct = (sims_correct > sims_distractor).mean()
        p_corrects.append(p_correct)
    
    p_corrects = np.array(p_corrects)
    
    return {
        'mean_p_correct': float(p_corrects.mean()),
        'accuracy': float((p_corrects > 0.5).mean()),
        'std_p_correct': float(p_corrects.std()),
    }


def calibration_analysis(
    z_gt: np.ndarray,
    mu: np.ndarray,
    logvar: np.ndarray,
    nominal_coverages: Tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)
) -> Dict[str, float]:
    """
    Calibration analysis for diagonal Gaussian predictions.
    
    For nominal coverage p (e.g., 90%), compute how often z_gt falls within
    the p-credible region (Mahalanobis distance threshold from Chi-square).
    
    Well-calibrated: empirical coverage ≈ nominal coverage.
    
    Args:
        z_gt: Ground truth (N, D)
        mu: Means (N, D)
        logvar: Log-variances (N, D)
        nominal_coverages: Nominal coverage levels to test
    
    Returns:
        Dictionary with empirical coverage for each nominal level
    """
    N, D = z_gt.shape
    var = np.exp(logvar)
    
    # Compute Mahalanobis distance squared for each sample
    # m² = Σ((z - μ)² / σ²)
    # Under the model, m² ~ Chi-square(D)
    mahalanobis_sq = np.sum((z_gt - mu) ** 2 / var, axis=1)  # (N,)
    
    coverage_results = {}
    
    for p in nominal_coverages:
        # Threshold from Chi-square(D) for coverage p
        threshold = chi2.ppf(p, df=D)
        
        # Empirical coverage
        empirical = (mahalanobis_sq <= threshold).mean()
        coverage_results[p] = float(empirical)
    
    # Calibration error: mean absolute deviation
    deviations = [abs(coverage_results[p] - p) for p in nominal_coverages]
    calibration_error = float(np.mean(deviations))
    
    return {
        'coverage': coverage_results,
        'calibration_error': calibration_error,
    }


def risk_coverage_curve(
    mu: np.ndarray,
    logvar: np.ndarray,
    z_gt: np.ndarray,
    gallery: np.ndarray,
    uncertainty_fn: str = 'trace',
    coverage_levels: Optional[np.ndarray] = None
) -> Dict[str, any]:
    """
    Risk-coverage curve for selective prediction.
    
    Idea: Model should abstain on uncertain predictions.
    1. Define uncertainty score u_i (e.g., mean variance)
    2. For coverage c ∈ [0, 1], keep lowest-uncertainty fraction c
    3. Compute error on retained samples
    4. Plot risk vs coverage (should decrease as coverage decreases)
    
    Args:
        mu: Means (N, D)
        logvar: Log-variances (N, D)
        z_gt: Ground truth (N, D)
        gallery: Gallery for retrieval evaluation (N, D)
        uncertainty_fn: 'trace' (mean variance) or 'logdet' (log-determinant)
        coverage_levels: Coverage fractions to evaluate (default: linspace(0.1, 1.0, 10))
    
    Returns:
        Dictionary with coverage levels, risks, and AURC
    """
    N = len(mu)
    
    # Compute uncertainty scores
    if uncertainty_fn == 'trace':
        uncertainties = np.exp(logvar).mean(axis=1)  # Mean variance
    elif uncertainty_fn == 'logdet':
        uncertainties = logvar.mean(axis=1)  # Mean log-variance
    else:
        raise ValueError(f"Unknown uncertainty_fn: {uncertainty_fn}")
    
    # Sort by uncertainty (ascending = most confident first)
    sorted_indices = np.argsort(uncertainties)
    
    if coverage_levels is None:
        coverage_levels = np.linspace(0.1, 1.0, 10)
    
    risks = []
    coverages_actual = []
    
    for coverage in coverage_levels:
        n_keep = int(coverage * N)
        if n_keep == 0:
            continue
        
        # Keep n_keep most confident samples
        keep_indices = sorted_indices[:n_keep]
        
        # Compute Bayesian Top-1 retrieval on kept samples
        mu_keep = mu[keep_indices]
        logvar_keep = logvar[keep_indices]
        gallery_keep = gallery[keep_indices]
        
        retrieval_results = bayesian_retrieval(mu_keep, logvar_keep, gallery_keep, ks=(1,))
        error = 1.0 - retrieval_results['top1_accuracy']
        
        risks.append(error)
        coverages_actual.append(n_keep / N)
    
    # Compute AURC (Area Under Risk-Coverage curve)
    # Lower is better
    aurc = np.trapz(risks, coverages_actual)
    
    return {
        'coverage_levels': coverages_actual,
        'risks': risks,
        'aurc': float(aurc),
    }


def conformal_prediction(
    mu_calib: np.ndarray,
    logvar_calib: np.ndarray,
    z_gt_calib: np.ndarray,
    mu_test: np.ndarray,
    logvar_test: np.ndarray,
    z_gt_test: np.ndarray,
    gallery_test: np.ndarray,
    alpha: float = 0.1
) -> Dict[str, float]:
    """
    Split conformal prediction for multivariate regression.
    
    Distribution-free uncertainty quantification:
    1. Compute nonconformity scores on calibration set
    2. Find (1-α) quantile
    3. Test set coverage should be ≥ 1-α (guaranteed under exchangeability)
    
    Nonconformity: Mahalanobis distance (or normalized L2)
    
    Args:
        mu_calib, logvar_calib, z_gt_calib: Calibration set
        mu_test, logvar_test, z_gt_test: Test set
        gallery_test: Gallery for candidate set evaluation
        alpha: Significance level (default: 0.1 for 90% coverage)
    
    Returns:
        coverage: Empirical coverage on test set
        avg_set_size: Average number of candidates in conformal set
    """
    N_calib = len(mu_calib)
    N_test = len(mu_test)
    D = mu_calib.shape[1]
    
    var_calib = np.exp(logvar_calib)
    var_test = np.exp(logvar_test)
    
    # Compute nonconformity scores on calibration set
    # Use Mahalanobis distance
    nonconformity_calib = np.sqrt(np.sum((z_gt_calib - mu_calib) ** 2 / var_calib, axis=1))
    
    # Find quantile
    q_level = np.ceil((1 - alpha) * (N_calib + 1)) / N_calib
    q_level = min(q_level, 1.0)
    threshold = np.quantile(nonconformity_calib, q_level)
    
    # Test set coverage
    nonconformity_test = np.sqrt(np.sum((z_gt_test - mu_test) ** 2 / var_test, axis=1))
    coverage = (nonconformity_test <= threshold).mean()
    
    # Average conformal set size (number of gallery candidates within threshold)
    # For each test sample, count how many gallery items fall within threshold
    set_sizes = []
    for i in range(N_test):
        # Compute nonconformity for all gallery candidates
        nonconformity_gallery = np.sqrt(
            np.sum((gallery_test - mu_test[i]) ** 2 / var_test[i], axis=1)
        )
        set_size = (nonconformity_gallery <= threshold).sum()
        set_sizes.append(set_size)
    
    avg_set_size = np.mean(set_sizes)
    
    return {
        'coverage': float(coverage),
        'avg_set_size': float(avg_set_size),
        'threshold': float(threshold),
        'target_coverage': 1 - alpha,
    }


def evaluate_probabilistic(
    mu: np.ndarray,
    logvar: np.ndarray,
    z_gt: np.ndarray,
    n_mc_samples: int = 64,
    run_conformal: bool = False,
    seed: int = 42
) -> ProbabilisticEvalResults:
    """
    Comprehensive probabilistic evaluation suite.
    
    Args:
        mu: Predicted means (N, D)
        logvar: Predicted log-variances (N, D)
        z_gt: Ground truth embeddings (N, D)
        n_mc_samples: Number of MC samples for Energy Score
        run_conformal: Whether to run conformal prediction (requires train/test split)
        seed: Random seed
    
    Returns:
        ProbabilisticEvalResults with all metrics
    """
    logger.info(f"Probabilistic evaluation: N={len(mu)}, D={mu.shape[1]}")
    
    # 1. Proper scoring rules
    nll = gaussian_nll(z_gt, mu, logvar)
    
    # Sample for Energy Score
    rng = np.random.RandomState(seed)
    std = np.sqrt(np.exp(logvar))
    samples = rng.randn(len(mu), n_mc_samples, mu.shape[1]) * std[:, None, :] + mu[:, None, :]
    es = energy_score(z_gt, samples)
    
    # 2. Bayesian retrieval
    bayesian_ret = bayesian_retrieval(mu, logvar, z_gt)
    cosine_ret = cosine_retrieval_from_mu(mu, z_gt)
    
    # 3. Probabilistic 2AFC
    prob_2afc = probabilistic_2afc(mu, logvar, z_gt, n_mc_samples=n_mc_samples, seed=seed)
    
    # 4. Calibration
    calib = calibration_analysis(z_gt, mu, logvar)
    
    # 5. Risk-coverage
    risk_cov = risk_coverage_curve(mu, logvar, z_gt, z_gt)
    
    # Build risk_at_coverage dict
    risk_at_cov_dict = {}
    for i, cov in enumerate(risk_cov['coverage_levels']):
        if cov >= 0.5 and 0.5 not in risk_at_cov_dict:
            risk_at_cov_dict[0.5] = risk_cov['risks'][i]
        if cov >= 0.9 and 0.9 not in risk_at_cov_dict:
            risk_at_cov_dict[0.9] = risk_cov['risks'][i]
    
    # 6. Conformal (optional)
    conformal_cov = None
    conformal_set_size = None
    if run_conformal:
        # Split into calib/test (50/50)
        N = len(mu)
        split = N // 2
        conf_results = conformal_prediction(
            mu[:split], logvar[:split], z_gt[:split],
            mu[split:], logvar[split:], z_gt[split:],
            z_gt[split:],
            alpha=0.1
        )
        conformal_cov = conf_results['coverage']
        conformal_set_size = conf_results['avg_set_size']
    
    return ProbabilisticEvalResults(
        nll_mean=float(nll.mean()),
        nll_std=float(nll.std()),
        energy_score_mean=float(es.mean()),
        energy_score_std=float(es.std()),
        bayesian_top1=bayesian_ret['top1_accuracy'],
        bayesian_top5=bayesian_ret['top5_accuracy'],
        bayesian_top10=bayesian_ret['top10_accuracy'],
        bayesian_mean_rank=bayesian_ret['mean_rank'],
        bayesian_mrr=bayesian_ret['mrr'],
        cosine_top1=cosine_ret['top1_accuracy'],
        improvement_top1=bayesian_ret['top1_accuracy'] - cosine_ret['top1_accuracy'],
        prob_2afc_mean=prob_2afc['mean_p_correct'],
        prob_2afc_accuracy=prob_2afc['accuracy'],
        calibration_coverage=calib['coverage'],
        calibration_error=calib['calibration_error'],
        aurc=risk_cov['aurc'],
        risk_at_coverage=risk_at_cov_dict,
        conformal_coverage=conformal_cov,
        conformal_avg_set_size=conformal_set_size,
    )
