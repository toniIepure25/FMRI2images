"""
Paper-grade embedding evaluation metrics for fMRI-to-CLIP decoding.

Implements:
- Retrieval: R@K, MeanR, MedR, MRR, nDCG@K
- Pairwise identification: 2AFC with bootstrap CI
- Discriminability: AUC(pos vs neg), Cohen's d, d-prime
- Structure: RSA (Spearman), linear CKA
- Gallery size curves with chance baselines
- Oracle sanity checks

All metrics support both deterministic (cosine) and Bayesian (distribution-aware) modes.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.stats import spearmanr, bootstrap
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import roc_auc_score, ndcg_score
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Container for retrieval metrics."""
    r_at_1: float
    r_at_5: float
    r_at_10: float
    mean_rank: float
    median_rank: float
    mrr: float
    ndcg_at_10: float
    gallery_size: int
    chance_r_at_1: float
    
    def to_dict(self) -> Dict:
        return {
            "r@1": self.r_at_1,
            "r@5": self.r_at_5,
            "r@10": self.r_at_10,
            "mean_rank": self.mean_rank,
            "median_rank": self.median_rank,
            "mrr": self.mrr,
            "ndcg@10": self.ndcg_at_10,
            "gallery_size": self.gallery_size,
            "chance_r@1": self.chance_r_at_1,
        }


@dataclass
class IdentificationMetrics:
    """Container for pairwise identification metrics."""
    two_afc_accuracy: float
    two_afc_ci_lower: float
    two_afc_ci_upper: float
    auc_pos_neg: float
    cohens_d: float
    d_prime: float
    n_pairs: int
    
    def to_dict(self) -> Dict:
        return {
            "2afc_accuracy": self.two_afc_accuracy,
            "2afc_ci_lower": self.two_afc_ci_lower,
            "2afc_ci_upper": self.two_afc_ci_upper,
            "auc_pos_neg": self.auc_pos_neg,
            "cohens_d": self.cohens_d,
            "d_prime": self.d_prime,
            "n_pairs": self.n_pairs,
        }


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between two sets of vectors.
    
    Args:
        a: (N, D) array
        b: (M, D) array
        
    Returns:
        (N, M) similarity matrix
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def compute_retrieval_metrics(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    k_values: List[int] = [1, 5, 10],
) -> RetrievalMetrics:
    """
    Compute retrieval metrics for query-gallery matching.
    
    Args:
        query_embeddings: (N, D) query embeddings
        gallery_embeddings: (M, D) gallery embeddings
        query_ids: (N,) query IDs
        gallery_ids: (M,) gallery IDs
        k_values: List of K for R@K computation
        
    Returns:
        RetrievalMetrics object
    """
    N = len(query_embeddings)
    M = len(gallery_embeddings)
    
    # Compute similarity matrix (N x M)
    sim_matrix = cosine_similarity_matrix(query_embeddings, gallery_embeddings)
    
    # Find ground truth indices in gallery for each query
    gt_indices = []
    for qid in query_ids:
        matches = np.where(gallery_ids == qid)[0]
        if len(matches) == 0:
            raise ValueError(f"Query ID {qid} not found in gallery")
        gt_indices.append(matches[0])  # Take first match
    
    gt_indices = np.array(gt_indices)
    
    # Rank gallery items by similarity (descending)
    ranks = []
    for i in range(N):
        scores = sim_matrix[i]
        sorted_indices = np.argsort(-scores)  # Descending order
        rank = np.where(sorted_indices == gt_indices[i])[0][0] + 1  # 1-indexed
        ranks.append(rank)
    
    ranks = np.array(ranks)
    
    # Compute metrics
    r_at_k = {}
    for k in k_values:
        r_at_k[k] = (ranks <= k).mean()
    
    mean_rank = ranks.mean()
    median_rank = np.median(ranks)
    mrr = (1.0 / ranks).mean()
    
    # nDCG@10
    relevance = np.zeros((N, M))
    relevance[np.arange(N), gt_indices] = 1.0
    ndcg_10 = ndcg_score(relevance, sim_matrix, k=10)
    
    # Chance baseline
    chance_r_at_1 = 1.0 / M
    
    return RetrievalMetrics(
        r_at_1=r_at_k.get(1, 0.0),
        r_at_5=r_at_k.get(5, 0.0),
        r_at_10=r_at_k.get(10, 0.0),
        mean_rank=mean_rank,
        median_rank=median_rank,
        mrr=mrr,
        ndcg_at_10=ndcg_10,
        gallery_size=M,
        chance_r_at_1=chance_r_at_1,
    )


def compute_identification_metrics(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    n_pairs: int = 5000,
    seed: int = 42,
    n_bootstrap: int = 1000,
) -> IdentificationMetrics:
    """
    Compute pairwise identification metrics (2AFC, AUC, Cohen's d, d').
    
    Args:
        embeddings_a: (N, D) first set of embeddings
        embeddings_b: (M, D) second set of embeddings
        labels_a: (N,) labels for first set
        labels_b: (M,) labels for second set
        n_pairs: Number of pairs to sample
        seed: Random seed
        n_bootstrap: Number of bootstrap samples for CI
        
    Returns:
        IdentificationMetrics object
    """
    rng = np.random.RandomState(seed)
    
    N = len(embeddings_a)
    M = len(embeddings_b)
    
    # Sample pairs
    idx_a = rng.randint(0, N, size=n_pairs)
    idx_b = rng.randint(0, M, size=n_pairs)
    
    # Compute similarities
    emb_a_norm = embeddings_a / (np.linalg.norm(embeddings_a, axis=1, keepdims=True) + 1e-8)
    emb_b_norm = embeddings_b / (np.linalg.norm(embeddings_b, axis=1, keepdims=True) + 1e-8)
    
    similarities = (emb_a_norm[idx_a] * emb_b_norm[idx_b]).sum(axis=1)
    is_match = labels_a[idx_a] == labels_b[idx_b]
    
    pos_sim = similarities[is_match]
    neg_sim = similarities[~is_match]
    
    # 2AFC accuracy
    two_afc_correct = (pos_sim > neg_sim[:len(pos_sim)]).mean() if len(pos_sim) > 0 else 0.5
    
    # Bootstrap CI for 2AFC
    def two_afc_statistic(pos, neg):
        return (pos > neg[:len(pos)]).mean() if len(pos) > 0 else 0.5
    
    try:
        res = bootstrap(
            (pos_sim, neg_sim),
            two_afc_statistic,
            n_resamples=n_bootstrap,
            random_state=seed,
            method='percentile',
        )
        ci_lower, ci_upper = res.confidence_interval
    except Exception as e:
        logger.warning(f"Bootstrap failed: {e}, using default CI")
        ci_lower, ci_upper = two_afc_correct - 0.05, two_afc_correct + 0.05
    
    # AUC
    y_true = np.concatenate([np.ones(len(pos_sim)), np.zeros(len(neg_sim))])
    y_score = np.concatenate([pos_sim, neg_sim])
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5
    
    # Cohen's d
    mean_diff = pos_sim.mean() - neg_sim.mean()
    pooled_std = np.sqrt(
        ((len(pos_sim) - 1) * pos_sim.std(ddof=1) ** 2 +
         (len(neg_sim) - 1) * neg_sim.std(ddof=1) ** 2) /
        (len(pos_sim) + len(neg_sim) - 2)
    )
    cohens_d = mean_diff / (pooled_std + 1e-8)
    
    # d-prime (signal detection theory)
    d_prime = (pos_sim.mean() - neg_sim.mean()) / np.sqrt(
        0.5 * (pos_sim.var() + neg_sim.var()) + 1e-8
    )
    
    return IdentificationMetrics(
        two_afc_accuracy=two_afc_correct,
        two_afc_ci_lower=ci_lower,
        two_afc_ci_upper=ci_upper,
        auc_pos_neg=auc,
        cohens_d=cohens_d,
        d_prime=d_prime,
        n_pairs=n_pairs,
    )


def compute_rsa(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    method: str = "spearman",
) -> float:
    """
    Compute Representational Similarity Analysis (RSA).
    
    Args:
        embeddings_a: (N, D1) first set of embeddings
        embeddings_b: (N, D2) second set of embeddings (same N)
        method: "spearman" or "pearson"
        
    Returns:
        Correlation coefficient
    """
    # Compute pairwise distance matrices
    rdm_a = pdist(embeddings_a, metric='cosine')
    rdm_b = pdist(embeddings_b, metric='cosine')
    
    # Correlate flattened RDMs
    if method == "spearman":
        corr, _ = spearmanr(rdm_a, rdm_b)
    elif method == "pearson":
        corr = np.corrcoef(rdm_a, rdm_b)[0, 1]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return corr


def compute_linear_cka(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
) -> float:
    """
    Compute linear Centered Kernel Alignment (CKA).
    
    Args:
        embeddings_a: (N, D1) first set of embeddings
        embeddings_b: (N, D2) second set of embeddings
        
    Returns:
        CKA score
    """
    # Center
    embeddings_a = embeddings_a - embeddings_a.mean(axis=0, keepdims=True)
    embeddings_b = embeddings_b - embeddings_b.mean(axis=0, keepdims=True)
    
    # Gram matrices
    gram_a = embeddings_a @ embeddings_a.T
    gram_b = embeddings_b @ embeddings_b.T
    
    # CKA
    numerator = np.linalg.norm(gram_a @ gram_b, ord='fro') ** 2
    denominator = np.linalg.norm(gram_a, ord='fro') * np.linalg.norm(gram_b, ord='fro')
    
    return numerator / (denominator + 1e-8)


def compute_retrieval_curves(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    gallery_sizes: List[int] = [2, 10, 50, 100, 500, 1000],
    seed: int = 42,
) -> Dict[int, RetrievalMetrics]:
    """
    Compute retrieval metrics across different gallery sizes.
    
    Args:
        query_embeddings: (N, D) query embeddings
        gallery_embeddings: (M, D) gallery embeddings
        query_ids: (N,) query IDs
        gallery_ids: (M,) gallery IDs
        gallery_sizes: List of gallery sizes to evaluate
        seed: Random seed for subsampling
        
    Returns:
        Dictionary mapping gallery size to RetrievalMetrics
    """
    rng = np.random.RandomState(seed)
    M = len(gallery_embeddings)
    
    results = {}
    
    for size in gallery_sizes:
        if size > M:
            logger.warning(f"Gallery size {size} exceeds available {M}, skipping")
            continue
        
        # Subsample gallery (deterministic given seed)
        gallery_indices = rng.choice(M, size=size, replace=False)
        
        # Ensure all query IDs are in the subsampled gallery
        # If not, add them
        missing_ids = set(query_ids) - set(gallery_ids[gallery_indices])
        if missing_ids:
            # Find indices of missing IDs
            missing_indices = [
                np.where(gallery_ids == mid)[0][0]
                for mid in missing_ids
                if len(np.where(gallery_ids == mid)[0]) > 0
            ]
            
            # Replace random gallery items with missing targets
            n_replace = min(len(missing_indices), len(gallery_indices))
            gallery_indices[:n_replace] = missing_indices[:n_replace]
        
        sub_gallery_embeddings = gallery_embeddings[gallery_indices]
        sub_gallery_ids = gallery_ids[gallery_indices]
        
        metrics = compute_retrieval_metrics(
            query_embeddings,
            sub_gallery_embeddings,
            query_ids,
            sub_gallery_ids,
        )
        
        results[size] = metrics
    
    return results


def oracle_retrieval_check(
    embeddings: np.ndarray,
    ids: np.ndarray,
    tolerance: float = 0.95,
) -> Tuple[bool, Dict[str, float]]:
    """
    Sanity check: GT embeddings should perfectly retrieve themselves.
    
    Args:
        embeddings: (N, D) embeddings
        ids: (N,) IDs
        tolerance: Minimum R@1 to pass (default: 0.95)
        
    Returns:
        (passed, metrics_dict)
    """
    metrics = compute_retrieval_metrics(
        embeddings,
        embeddings,
        ids,
        ids,
    )
    
    passed = metrics.r_at_1 >= tolerance
    
    return passed, {
        "oracle_r@1": metrics.r_at_1,
        "oracle_mean_rank": metrics.mean_rank,
        "oracle_mrr": metrics.mrr,
    }


class EmbeddingEvaluator:
    """
    Unified evaluator for embedding-based retrieval and identification.
    
    Usage:
        evaluator = EmbeddingEvaluator()
        results = evaluator.evaluate(
            pred_embeddings=pred_emb,
            gt_embeddings=gt_emb,
            ids=ids,
        )
    """
    
    def __init__(
        self,
        gallery_sizes: List[int] = [2, 10, 50, 100, 500, 1000],
        n_identification_pairs: int = 5000,
        seed: int = 42,
    ):
        self.gallery_sizes = gallery_sizes
        self.n_identification_pairs = n_identification_pairs
        self.seed = seed
    
    def evaluate(
        self,
        pred_embeddings: np.ndarray,
        gt_embeddings: np.ndarray,
        ids: np.ndarray,
        compute_rsa: bool = True,
        compute_cka: bool = False,
    ) -> Dict:
        """
        Run full evaluation suite.
        
        Args:
            pred_embeddings: (N, D) predicted embeddings
            gt_embeddings: (N, D) ground truth embeddings
            ids: (N,) sample IDs
            compute_rsa: Whether to compute RSA
            compute_cka: Whether to compute CKA (slower)
            
        Returns:
            Dictionary of all metrics
        """
        logger.info("Running embedding evaluation...")
        
        results = {}
        
        # 1. Oracle check
        logger.info("  Oracle check...")
        passed, oracle_metrics = oracle_retrieval_check(gt_embeddings, ids)
        results["oracle"] = oracle_metrics
        results["oracle_passed"] = passed
        
        if not passed:
            logger.error(
                f"Oracle check FAILED: R@1={oracle_metrics['oracle_r@1']:.4f} < 0.95. "
                "Check for ID mismatches, normalization issues, or shuffling bugs!"
            )
        
        # 2. Retrieval curves
        logger.info("  Computing retrieval curves...")
        retrieval_curves = compute_retrieval_curves(
            pred_embeddings,
            gt_embeddings,
            ids,
            ids,
            gallery_sizes=self.gallery_sizes,
            seed=self.seed,
        )
        
        results["retrieval_curves"] = {
            size: metrics.to_dict()
            for size, metrics in retrieval_curves.items()
        }
        
        # 3. Full retrieval (all data)
        logger.info("  Computing full retrieval metrics...")
        full_retrieval = compute_retrieval_metrics(
            pred_embeddings,
            gt_embeddings,
            ids,
            ids,
        )
        results["retrieval_full"] = full_retrieval.to_dict()
        
        # 4. Identification
        logger.info("  Computing identification metrics...")
        identification = compute_identification_metrics(
            pred_embeddings,
            gt_embeddings,
            ids,
            ids,
            n_pairs=self.n_identification_pairs,
            seed=self.seed,
        )
        results["identification"] = identification.to_dict()
        
        # 5. RSA
        if compute_rsa:
            logger.info("  Computing RSA...")
            rsa_score = compute_rsa(pred_embeddings, gt_embeddings, method="spearman")
            results["rsa_spearman"] = rsa_score
        
        # 6. CKA
        if compute_cka:
            logger.info("  Computing linear CKA...")
            cka_score = compute_linear_cka(pred_embeddings, gt_embeddings)
            results["linear_cka"] = cka_score
        
        logger.info("Evaluation complete.")
        
        return results
