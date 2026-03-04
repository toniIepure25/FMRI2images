"""
Standard Embedding Evaluation Metrics for Stage 1 (fMRI → CLIP)
================================================================

Research-grade metrics for evaluating deterministic and probabilistic embedding predictions.
Implements field-standard metrics used in neural decoding literature.

Metrics Categories:
  1. Retrieval: Top-K accuracy, Mean/Median rank, MRR
  2. Two-way identification (2AFC): Pairwise discrimination task
  3. Matched vs mismatched: Separability analysis with AUC
  4. RSA: Representational similarity analysis
  5. Collapse diagnostics: Detection of mode collapse

Scientific Context:
- Retrieval metrics: Standard in CLIP-based neural decoding (Ozcelik & VanRullen 2023, Scotti et al. 2023)
- 2AFC: Forced-choice identification from cognitive neuroscience (Green & Swets 1966)
- RSA: Measures preservation of pairwise relationships (Kriegeskorte et al. 2008)
- Collapse: Detects degenerate solutions common in variational methods

References:
- Ozcelik & VanRullen (2023). "Natural scene reconstruction from fMRI signals using GBDS"
- Scotti et al. (2023). "Reconstructing the Mind's Eye"
- Kriegeskorte et al. (2008). "Representational similarity analysis"
- Green & Swets (1966). "Signal Detection Theory and Psychophysics"

Author: Research-grade evaluation suite
Date: January 2026
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import logging

# Optional dependencies with graceful fallback
try:
    from scipy.stats import spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("scipy not available. RSA will be unavailable.")

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except (ImportError, ValueError) as e:
    # ValueError can occur from numpy incompatibility
    SKLEARN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"sklearn not available or incompatible: {e}. Separability AUC will use manual implementation.")

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingEvalResults:
    """Container for embedding evaluation results."""
    
    # Retrieval metrics
    top1_accuracy: float
    top5_accuracy: float
    top10_accuracy: float
    mean_rank: float
    median_rank: float
    mrr: float  # Mean reciprocal rank
    
    # 2AFC identification
    twoafc_accuracy: float
    twoafc_ci_lower: float  # 95% confidence interval
    twoafc_ci_upper: float
    
    # Matched vs mismatched
    separability_auc: float
    cohens_d: float  # Effect size
    
    # RSA
    rsa_spearman: float
    rsa_pvalue: float
    
    # Collapse diagnostics
    pred_std_mean: float  # Mean std across dimensions
    gt_std_mean: float    # For comparison
    collapse_ratio: float  # pred_std / gt_std (should be ~1.0)
    avg_pairwise_sim: float  # Average cosine among predictions (collapse indicator)
    mean_centering: float  # Cosine(pred, mean(GT)) - collapse to mean indicator
    
    # Gallery size analysis (if multiple sizes tested)
    retrieval_by_gallery_size: Optional[Dict[int, Dict[str, float]]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            'retrieval': {
                'top1_accuracy': float(self.top1_accuracy),
                'top5_accuracy': float(self.top5_accuracy),
                'top10_accuracy': float(self.top10_accuracy),
                'mean_rank': float(self.mean_rank),
                'median_rank': float(self.median_rank),
                'mrr': float(self.mrr),
            },
            'identification': {
                'twoafc_accuracy': float(self.twoafc_accuracy),
                'twoafc_ci': [float(self.twoafc_ci_lower), float(self.twoafc_ci_upper)],
            },
            'separability': {
                'auc': float(self.separability_auc),
                'cohens_d': float(self.cohens_d),
            },
            'rsa': {
                'spearman_r': float(self.rsa_spearman),
                'p_value': float(self.rsa_pvalue),
            },
            'collapse_diagnostics': {
                'pred_std_mean': float(self.pred_std_mean),
                'gt_std_mean': float(self.gt_std_mean),
                'collapse_ratio': float(self.collapse_ratio),
                'avg_pairwise_sim': float(self.avg_pairwise_sim),
                'mean_centering': float(self.mean_centering),
            }
        }
        
        if self.retrieval_by_gallery_size is not None:
            result['retrieval_by_gallery_size'] = {
                str(k): v for k, v in self.retrieval_by_gallery_size.items()
            }
        
        return result


def normalize_embeddings(embeddings: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """L2-normalize embeddings to unit length. Critical for cosine similarity."""
    if isinstance(embeddings, torch.Tensor):
        return F.normalize(embeddings, p=2, dim=-1)
    else:
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        return embeddings / norms


def compute_retrieval_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10),
    normalize: bool = True
) -> Dict[str, float]:
    """
    Compute retrieval metrics (Top-K, Mean rank, MRR).
    
    For each prediction, rank all ground truth embeddings by cosine similarity
    and check if the correct match appears in top-K.
    
    Args:
        predictions: Predicted embeddings (N, D)
        ground_truth: Ground truth embeddings (N, D), same order as predictions
        ks: K values for Top-K accuracy
        normalize: Whether to L2-normalize (should be True for cosine)
    
    Returns:
        Dictionary with retrieval metrics
        
    Note:
        - Assumes predictions[i] should match ground_truth[i]
        - Computes similarity matrix and finds rank of diagonal elements
        - Chance Top-1 = 1/N (important for interpretation)
    """
    if normalize:
        predictions = normalize_embeddings(predictions)
        ground_truth = normalize_embeddings(ground_truth)
    
    # Compute similarity matrix: (N, N)
    # similarities[i, j] = cosine(pred[i], gt[j])
    similarities = predictions @ ground_truth.T
    
    # For each prediction i, rank all ground truth by similarity
    # ranks[i] contains sorted indices of GT from most to least similar
    ranks = np.argsort(-similarities, axis=1)  # Descending order
    
    # Find position of correct match for each prediction
    # correct_ranks[i] = position of GT[i] in ranking for pred[i]
    N = len(predictions)
    correct_ranks = np.zeros(N, dtype=np.int32)
    for i in range(N):
        # Find where i appears in ranks[i]
        correct_ranks[i] = np.where(ranks[i] == i)[0][0]
    
    # Compute metrics
    results = {}
    for k in ks:
        topk_acc = (correct_ranks < k).mean()
        results[f'top{k}_accuracy'] = float(topk_acc)
    
    results['mean_rank'] = float(correct_ranks.mean() + 1)  # +1 for 1-indexed
    results['median_rank'] = float(np.median(correct_ranks) + 1)
    
    # Mean Reciprocal Rank (MRR)
    reciprocal_ranks = 1.0 / (correct_ranks + 1.0)  # +1 for 1-indexed
    results['mrr'] = float(reciprocal_ranks.mean())
    
    # Add chance baseline for reference
    results['chance_top1'] = 1.0 / N
    
    return results


def csls_similarity(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Cross-domain Similarity Local Scaling (Conneau et al., 2018).

    Corrects for hubness by penalising embeddings that are universally
    close to many points.  CSLS(x, y) = 2*cos(x,y) - r_X(x) - r_Y(y)
    where r_X(x) is the mean similarity of x to its k-NN in Y.

    Args:
        predictions:  (N, D) L2-normalised predicted embeddings.
        ground_truth: (M, D) L2-normalised gallery embeddings.
        k: Number of nearest neighbours for hubness estimation.

    Returns:
        (N, M) CSLS-corrected similarity matrix.
    """
    sim = predictions @ ground_truth.T  # (N, M)
    k = min(k, sim.shape[1] - 1, sim.shape[0] - 1)
    if k < 1:
        return sim
    r_x = np.sort(sim, axis=1)[:, -k:].mean(axis=1)   # (N,)
    r_y = np.sort(sim, axis=0)[-k:, :].mean(axis=0)    # (M,)
    return 2.0 * sim - r_x[:, None] - r_y[None, :]


def compute_retrieval_metrics_csls(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10),
    normalize: bool = True,
    csls_k: int = 10,
) -> Dict[str, float]:
    """Retrieval metrics using CSLS-corrected similarity (hubness-aware).

    Same interface as :func:`compute_retrieval_metrics` but replaces raw
    cosine with CSLS similarity to mitigate the hubness problem common in
    high-dimensional embedding spaces.
    """
    if normalize:
        predictions = normalize_embeddings(predictions)
        ground_truth = normalize_embeddings(ground_truth)

    similarities = csls_similarity(predictions, ground_truth, k=csls_k)
    ranks = np.argsort(-similarities, axis=1)

    N = len(predictions)
    correct_ranks = np.zeros(N, dtype=np.int32)
    for i in range(N):
        correct_ranks[i] = np.where(ranks[i] == i)[0][0]

    results: Dict[str, float] = {}
    for k in ks:
        results[f'top{k}_accuracy'] = float((correct_ranks < k).mean())
    results['mean_rank'] = float(correct_ranks.mean() + 1)
    results['median_rank'] = float(np.median(correct_ranks) + 1)
    results['mrr'] = float((1.0 / (correct_ranks + 1.0)).mean())
    results['chance_top1'] = 1.0 / N
    return results


def compute_retrieval_by_gallery_size(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    gallery_sizes: List[int],
    n_trials: int = 100,
    seed: int = 42,
    normalize: bool = True
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate retrieval performance vs gallery size.
    
    For each gallery size, subsample distractors and compute retrieval metrics.
    This shows how difficulty scales with candidate pool size.
    
    Args:
        predictions: Predicted embeddings (N, D)
        ground_truth: Ground truth embeddings (N, D)
        gallery_sizes: List of gallery sizes to test (e.g., [2, 10, 50, 100, 1000])
        n_trials: Number of trials per gallery size (for averaging)
        seed: Random seed for reproducibility
        normalize: Whether to L2-normalize
    
    Returns:
        Dictionary mapping gallery_size -> metrics
    """
    rng = np.random.RandomState(seed)
    N = len(predictions)
    
    if normalize:
        predictions = normalize_embeddings(predictions)
        ground_truth = normalize_embeddings(ground_truth)
    
    results = {}
    
    for gallery_size in gallery_sizes:
        if gallery_size > N:
            logger.warning(f"Gallery size {gallery_size} > dataset size {N}, skipping")
            continue
        
        top1_accs = []
        top5_accs = []
        mean_ranks = []
        
        for trial in range(n_trials):
            # For each sample, create a gallery of size gallery_size
            trial_top1 = []
            trial_top5 = []
            trial_ranks = []
            
            for i in range(N):
                # Gallery: correct item + (gallery_size-1) random distractors
                distractors = rng.choice(
                    [j for j in range(N) if j != i],
                    size=min(gallery_size - 1, N - 1),
                    replace=False
                )
                gallery_indices = np.concatenate([[i], distractors])
                
                # Compute similarities to gallery
                sims = predictions[i] @ ground_truth[gallery_indices].T
                
                # Rank gallery items
                ranks = np.argsort(-sims)
                
                # Find rank of correct item (which is at index 0 in gallery_indices)
                correct_rank = np.where(ranks == 0)[0][0]
                
                trial_top1.append(correct_rank == 0)
                trial_top5.append(correct_rank < min(5, gallery_size))
                trial_ranks.append(correct_rank + 1)  # 1-indexed
            
            top1_accs.append(np.mean(trial_top1))
            top5_accs.append(np.mean(trial_top5))
            mean_ranks.append(np.mean(trial_ranks))
        
        results[gallery_size] = {
            'top1_accuracy': float(np.mean(top1_accs)),
            'top1_std': float(np.std(top1_accs)),
            'top5_accuracy': float(np.mean(top5_accs)),
            'top5_std': float(np.std(top5_accs)),
            'mean_rank': float(np.mean(mean_ranks)),
            'mean_rank_std': float(np.std(mean_ranks)),
            'chance_top1': 1.0 / gallery_size,
        }
    
    return results


def compute_two_afc(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    n_trials: int = 1000,
    seed: int = 42,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Two-alternative forced choice (2AFC) identification task.
    
    For each trial:
      1. Pick a query prediction
      2. Pick a random distractor from ground truth
      3. Score: correct if sim(pred, gt_correct) > sim(pred, gt_distractor)
    
    Reports accuracy and bootstrap 95% CI.
    
    Args:
        predictions: Predicted embeddings (N, D)
        ground_truth: Ground truth embeddings (N, D)
        n_trials: Number of 2AFC trials
        seed: Random seed
        normalize: Whether to L2-normalize
    
    Returns:
        Dictionary with 'accuracy', 'ci_lower', 'ci_upper'
    """
    rng = np.random.RandomState(seed)
    N = len(predictions)
    
    if normalize:
        predictions = normalize_embeddings(predictions)
        ground_truth = normalize_embeddings(ground_truth)
    
    correct = 0
    
    for _ in range(n_trials):
        # Sample a random query
        i = rng.randint(N)
        
        # Sample a random distractor (different from i)
        j = rng.randint(N - 1)
        if j >= i:
            j += 1
        
        # Compute similarities
        sim_correct = predictions[i] @ ground_truth[i]
        sim_distractor = predictions[i] @ ground_truth[j]
        
        if sim_correct > sim_distractor:
            correct += 1
    
    accuracy = correct / n_trials
    
    # Bootstrap 95% CI
    n_bootstrap = 1000
    bootstrap_accs = []
    for _ in range(n_bootstrap):
        bootstrap_sample = rng.binomial(n_trials, accuracy) / n_trials
        bootstrap_accs.append(bootstrap_sample)
    
    ci_lower = np.percentile(bootstrap_accs, 2.5)
    ci_upper = np.percentile(bootstrap_accs, 97.5)
    
    return {
        'accuracy': float(accuracy),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'chance': 0.5,
    }


def compute_separability_auc(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    n_mismatches_per_sample: int = 100,
    seed: int = 42,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Compute ROC AUC for matched vs mismatched similarity distributions.
    
    For each prediction:
      - Positive: sim(pred, gt_correct)
      - Negatives: sim(pred, gt_mismatch) for random mismatches
    
    AUC measures how well we can discriminate matches from mismatches.
    Also computes Cohen's d effect size.
    
    Args:
        predictions: (N, D)
        ground_truth: (N, D)
        n_mismatches_per_sample: Number of negative samples per positive
        seed: Random seed
        normalize: L2-normalize
    
    Returns:
        Dictionary with 'auc' and 'cohens_d'
    """
    rng = np.random.RandomState(seed)
    N = len(predictions)
    
    if normalize:
        predictions = normalize_embeddings(predictions)
        ground_truth = normalize_embeddings(ground_truth)
    
    pos_sims = []
    neg_sims = []
    
    for i in range(N):
        # Positive: correct match
        pos_sim = predictions[i] @ ground_truth[i]
        pos_sims.append(pos_sim)
        
        # Negatives: random mismatches
        mismatch_indices = rng.choice(
            [j for j in range(N) if j != i],
            size=min(n_mismatches_per_sample, N - 1),
            replace=False
        )
        for j in mismatch_indices:
            neg_sim = predictions[i] @ ground_truth[j]
            neg_sims.append(neg_sim)
    
    # Combine and create labels
    pos_sims = np.array(pos_sims)
    neg_sims = np.array(neg_sims)
    
    all_sims = np.concatenate([pos_sims, neg_sims])
    labels = np.concatenate([np.ones(len(pos_sims)), np.zeros(len(neg_sims))])
    
    # Compute AUC
    if SKLEARN_AVAILABLE:
        auc = roc_auc_score(labels, all_sims)
    else:
        # Manual AUC computation using trapezoidal rule
        # Sort by score (descending)
        sorted_indices = np.argsort(-all_sims)
        sorted_labels = labels[sorted_indices]
        
        # Compute TPR and FPR at each threshold
        n_pos = sorted_labels.sum()
        n_neg = len(sorted_labels) - n_pos
        
        tpr = np.cumsum(sorted_labels) / n_pos
        fpr = np.cumsum(1 - sorted_labels) / n_neg
        
        # Trapezoidal rule
        auc = np.trapz(tpr, fpr)
    
    # Cohen's d effect size
    mean_pos = pos_sims.mean()
    mean_neg = neg_sims.mean()
    std_pooled = np.sqrt((pos_sims.var() + neg_sims.var()) / 2)
    cohens_d = (mean_pos - mean_neg) / std_pooled
    
    return {
        'auc': float(auc),
        'cohens_d': float(cohens_d),
        'mean_pos_sim': float(mean_pos),
        'mean_neg_sim': float(mean_neg),
        'std_pos_sim': float(pos_sims.std()),
        'std_neg_sim': float(neg_sims.std()),
    }


def compute_rsa(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    n_samples: int = 200,
    seed: int = 42,
    normalize: bool = True
) -> Dict[str, float]:
    if not SCIPY_AVAILABLE:
        logger.error("scipy not available for RSA computation")
        return {'spearman_r': 0.0, 'p_value': 1.0}
    
    """
    Representational Similarity Analysis (RSA).
    
    Computes Spearman correlation between pairwise distance matrices
    of predictions and ground truth. Measures preservation of relational structure.
    
    Args:
        predictions: (N, D)
        ground_truth: (N, D)
        n_samples: Subsample for efficiency (RSA is O(N^2))
        seed: Random seed
        normalize: L2-normalize
    
    Returns:
        Dictionary with 'spearman_r' and 'p_value'
    """
    rng = np.random.RandomState(seed)
    N = len(predictions)
    
    if n_samples < N:
        indices = rng.choice(N, size=n_samples, replace=False)
        predictions = predictions[indices]
        ground_truth = ground_truth[indices]
    
    if normalize:
        predictions = normalize_embeddings(predictions)
        ground_truth = normalize_embeddings(ground_truth)
    
    # Compute distance matrices (1 - cosine similarity)
    # Using cosine for CLIP embeddings
    dist_pred = 1 - (predictions @ predictions.T)
    dist_gt = 1 - (ground_truth @ ground_truth.T)
    
    # Extract upper triangular (exclude diagonal)
    mask = np.triu(np.ones_like(dist_pred, dtype=bool), k=1)
    vec_pred = dist_pred[mask]
    vec_gt = dist_gt[mask]
    
    # Spearman correlation
    r, p = spearmanr(vec_pred, vec_gt)
    
    return {
        'spearman_r': float(r),
        'p_value': float(p),
    }


def compute_collapse_diagnostics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Detect mode collapse in predictions.
    
    Indicators of collapse:
      1. Low per-dimension std compared to GT
      2. High average pairwise similarity (predictions are similar to each other)
      3. High similarity to mean GT (all predictions near GT centroid)
    
    Args:
        predictions: (N, D)
        ground_truth: (N, D)
        normalize: L2-normalize
    
    Returns:
        Dictionary with collapse diagnostics
    """
    if normalize:
        predictions = normalize_embeddings(predictions)
        ground_truth = normalize_embeddings(ground_truth)
    
    # Per-dimension std
    pred_std = predictions.std(axis=0).mean()
    gt_std = ground_truth.std(axis=0).mean()
    collapse_ratio = pred_std / (gt_std + 1e-8)
    
    # Average pairwise similarity among predictions
    # High value indicates all predictions are similar
    pairwise_sims = predictions @ predictions.T
    # Exclude diagonal
    N = len(predictions)
    mask = ~np.eye(N, dtype=bool)
    avg_pairwise_sim = pairwise_sims[mask].mean()
    
    # Similarity to mean GT (centroid)
    gt_mean = ground_truth.mean(axis=0, keepdims=True)
    gt_mean = normalize_embeddings(gt_mean)
    sims_to_mean = (predictions @ gt_mean.T).flatten()
    mean_centering = sims_to_mean.mean()
    
    return {
        'pred_std_mean': float(pred_std),
        'gt_std_mean': float(gt_std),
        'collapse_ratio': float(collapse_ratio),
        'avg_pairwise_sim': float(avg_pairwise_sim),
        'mean_centering': float(mean_centering),
    }


def evaluate_embeddings(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    gallery_sizes: Optional[List[int]] = None,
    normalize: bool = True,
    seed: int = 42
) -> EmbeddingEvalResults:
    """
    Comprehensive embedding evaluation suite.
    
    Runs all standard metrics and returns structured results.
    
    Args:
        predictions: Predicted embeddings (N, D)
        ground_truth: Ground truth embeddings (N, D)
        gallery_sizes: Optional list of gallery sizes for scaling analysis
        normalize: Whether to L2-normalize embeddings
        seed: Random seed for reproducibility
    
    Returns:
        EmbeddingEvalResults dataclass with all metrics
    """
    logger.info(f"Evaluating embeddings: N={len(predictions)}, D={predictions.shape[1]}")
    
    # Retrieval
    retrieval = compute_retrieval_metrics(predictions, ground_truth, normalize=normalize)
    
    # 2AFC
    twoafc = compute_two_afc(predictions, ground_truth, seed=seed, normalize=normalize)
    
    # Separability
    separability = compute_separability_auc(predictions, ground_truth, seed=seed, normalize=normalize)
    
    # RSA
    rsa = compute_rsa(predictions, ground_truth, seed=seed, normalize=normalize)
    
    # Collapse
    collapse = compute_collapse_diagnostics(predictions, ground_truth, normalize=normalize)
    
    # Gallery size analysis (optional)
    retrieval_by_size = None
    if gallery_sizes is not None:
        retrieval_by_size = compute_retrieval_by_gallery_size(
            predictions, ground_truth, gallery_sizes, seed=seed, normalize=normalize
        )
    
    return EmbeddingEvalResults(
        top1_accuracy=retrieval['top1_accuracy'],
        top5_accuracy=retrieval['top5_accuracy'],
        top10_accuracy=retrieval['top10_accuracy'],
        mean_rank=retrieval['mean_rank'],
        median_rank=retrieval['median_rank'],
        mrr=retrieval['mrr'],
        twoafc_accuracy=twoafc['accuracy'],
        twoafc_ci_lower=twoafc['ci_lower'],
        twoafc_ci_upper=twoafc['ci_upper'],
        separability_auc=separability['auc'],
        cohens_d=separability['cohens_d'],
        rsa_spearman=rsa['spearman_r'],
        rsa_pvalue=rsa['p_value'],
        pred_std_mean=collapse['pred_std_mean'],
        gt_std_mean=collapse['gt_std_mean'],
        collapse_ratio=collapse['collapse_ratio'],
        avg_pairwise_sim=collapse['avg_pairwise_sim'],
        mean_centering=collapse['mean_centering'],
        retrieval_by_gallery_size=retrieval_by_size,
    )
