"""Shared numerical utilities for manifold analysis.

Pure-numpy helpers used across all analysis modules.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


def safe_normalize(
    x: np.ndarray, axis: int = -1, eps: float = 1e-8
) -> np.ndarray:
    """L2-normalize along *axis*, mapping zero vectors to zero."""
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return np.where(norms > eps, x / norms, 0.0)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of *a* and every row of *b*.

    Args:
        a: (N, D) L2-normalized embeddings.
        b: (M, D) L2-normalized embeddings.

    Returns:
        (N, M) similarity matrix.
    """
    return a @ b.T


def csls_scores(
    query: np.ndarray,
    gallery: np.ndarray,
    k: int = 10,
    cosine_scores: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Cross-domain Similarity Local Scaling (Conneau et al., 2018).

    Args:
        query: (N, D) L2-normalized query embeddings.
        gallery: (M, D) L2-normalized gallery embeddings.
        k: Number of neighbours for mean similarity penalty.
        cosine_scores: Pre-computed (N, M) cosine matrix; avoids recomputation.

    Returns:
        (N, M) CSLS score matrix.
    """
    if cosine_scores is None:
        cosine_scores = cosine_similarity_matrix(query, gallery)
    k = min(k, cosine_scores.shape[1])
    r_q = np.mean(np.sort(cosine_scores, axis=1)[:, -k:], axis=1, keepdims=True)
    r_g = np.mean(np.sort(cosine_scores, axis=0)[-k:, :], axis=0, keepdims=True)
    return 2 * cosine_scores - r_q - r_g


def gini_coefficient(counts: np.ndarray) -> float:
    """Gini coefficient of a non-negative array (0 = perfect equality)."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.sum() == 0:
        return 0.0
    sorted_c = np.sort(counts)
    n = len(sorted_c)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_c) - (n + 1) * np.sum(sorted_c)) / (n * np.sum(sorted_c)))


def slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between unit vectors *v0* and *v1*."""
    v0 = safe_normalize(v0.ravel())
    v1 = safe_normalize(v1.ravel())
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    omega = np.arccos(dot)
    if np.abs(omega) < 1e-10:
        return (1 - t) * v0 + t * v1
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * v0 + np.sin(t * omega) / so * v1


def weighted_harmonic_mean(
    values: dict[str, float],
    weights: dict[str, float],
) -> Tuple[float, dict[str, float]]:
    """Weighted harmonic mean over available components.

    Missing or non-positive values are excluded. Returns the score and
    a dict of actually-used weights (re-normalised to sum to 1).
    """
    used: dict[str, float] = {}
    for k, v in values.items():
        w = weights.get(k, 1.0)
        if v is not None and np.isfinite(v) and v > 0 and w > 0:
            used[k] = w
    if not used:
        return 0.0, {}
    total_w = sum(used.values())
    norm_w = {k: w / total_w for k, w in used.items()}
    denom = sum(norm_w[k] / values[k] for k in norm_w)
    return float(1.0 / denom), norm_w


def upper_triangle(mat: np.ndarray) -> np.ndarray:
    """Extract strict upper-triangle entries of a square matrix."""
    idx = np.triu_indices_from(mat, k=1)
    return mat[idx]


def percentile_normalize(x: np.ndarray) -> np.ndarray:
    """Map values to [0, 1] via rank percentiles."""
    x = np.asarray(x, dtype=np.float64)
    ranks = sp_stats.rankdata(x, method="average")
    return (ranks - 1) / max(len(ranks) - 1, 1)


def zscore_normalize(x: np.ndarray) -> np.ndarray:
    """Standard z-score normalisation (mean 0, std 1)."""
    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x)
    std = np.std(x)
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / std


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k scores per row (descending).

    Args:
        scores: (N, M) score matrix.
        k: Number of top entries.

    Returns:
        (N, k) index array.
    """
    k = min(k, scores.shape[1])
    return np.argpartition(scores, -k, axis=1)[:, -k:]


def topk_entropy(scores: np.ndarray, k: int) -> np.ndarray:
    """Entropy of softmax over top-k retrieval scores per query.

    Args:
        scores: (N, M) score matrix (higher = more similar).
        k: Number of top entries to consider.

    Returns:
        (N,) entropy values.
    """
    idx = topk_indices(scores, k)
    top_scores = np.take_along_axis(scores, idx, axis=1)
    top_scores = top_scores - top_scores.max(axis=1, keepdims=True)
    exp_s = np.exp(top_scores)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    return -np.sum(probs * np.log(probs + 1e-12), axis=1)


def bootstrap_ci(
    values: np.ndarray,
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 500,
    seed: int = 42,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Nonparametric bootstrap confidence interval for a 1D statistic."""
    values = np.asarray(values)
    if n_bootstrap <= 0 or values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    n = len(values)
    stats = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        stats[i] = stat_fn(values[idx])
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.percentile(stats, 100.0 * alpha)),
        float(np.percentile(stats, 100.0 * (1.0 - alpha))),
    )
