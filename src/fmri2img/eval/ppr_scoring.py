"""
Posterior Predictive Retrieval (PPR) Scoring for vMF Neural Decoding.

Replaces ad-hoc CSLS hubness correction with principled Bayesian retrieval
scoring using the full von Mises-Fisher posterior.

Mathematical Foundation:

Given a vMF decoder with output (mu, kappa) for a brain scan, the log-
likelihood of candidate embedding z under the vMF distribution is:

    log p(z | mu, kappa) = kappa * cos(mu, z) + log C_d(kappa)

where C_d(kappa) is the normalizing constant:

    C_d(kappa) = kappa^{d/2-1} / ((2*pi)^{d/2} * I_{d/2-1}(kappa))

and I_v(kappa) is the modified Bessel function of the first kind.

Key insight: The log C_d(kappa) term acts as a *natural* hubness correction.
High-kappa (confident) predictions get sharp, discriminative rankings.
Low-kappa (uncertain) predictions get diffuse rankings that are less likely
to produce false positives. This replaces the post-hoc CSLS penalty with
an intrinsic Bayesian mechanism.

For large d (e.g. d=768), we use the stable asymptotic approximation:

    log C_d(kappa) ~ (d/2 - 1) * log(kappa) - d/2 * log(2*pi)
                     - kappa - 0.5 * log(kappa) + (d-1)/2 * log(2)
                     + simplified Bessel tail

In practice, the scoring function reduces to:

    PPR(z | mu, kappa) = kappa * cos(mu, z) + log_normalizer(kappa, d)

The log_normalizer only depends on kappa (not z), so for a SINGLE query
it does not change the ranking. However, when comparing across queries
(e.g., in gallery-side normalization or fusion), it matters because
different queries have different kappas.

For within-query ranking, PPR with kappa scaling:
    PPR_score(z | mu, kappa) = kappa * cos(mu, z)

is already different from raw cosine because it weights by confidence.

For cross-query normalized scoring (the hubness-aware version):
    PPR_csls(z | mu, kappa) = kappa * cos(mu, z) + log_C_d(kappa)
                              - r_PPR(mu, kappa) - r_gallery(z)

where the mean-NN penalties use PPR scores instead of raw cosines.

References:
    Banerjee et al. (2005). "Clustering on the Unit Hypersphere using vMF"
    Conneau et al. (2018). CSLS for cross-lingual word embedding alignment
    Wang & Isola (2020). "Understanding Contrastive Representation Learning"

Author: Generated for the PPNR roadmap (V55 experiment line)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Precomputed constant: log(2*pi) for d-dimensional normalization
_LOG_2PI = np.log(2.0 * np.pi)


def _log_vmf_normalizer(kappa: np.ndarray, d: int) -> np.ndarray:
    """Compute log C_d(kappa) using a numerically stable approximation.

    For d >> 1 (e.g. d=768), the exact Bessel function is numerically
    intractable. We use the saddle-point / large-d asymptotic:

        log C_d(kappa) ~ (d/2 - 1) * log(kappa)
                        - (d/2) * log(2*pi)
                        - sqrt(kappa^2 + (d/2 - 1)^2)
                        + (d/2 - 1) * log((d/2 - 1) + sqrt(kappa^2 + (d/2 - 1)^2))

    This is derived from the uniform asymptotic expansion of I_v(z) for
    large v (Abramowitz & Stegun, 9.7.7; Amos, 1974).

    Args:
        kappa: Concentration parameters, shape (N,).
        d: Embedding dimensionality.

    Returns:
        log C_d(kappa) with shape (N,).
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    nu = d / 2.0 - 1.0  # Bessel order

    kappa_safe = np.maximum(kappa, 1e-10)

    hypot = np.sqrt(kappa_safe ** 2 + nu ** 2)

    log_bessel_approx = (
        -hypot
        + nu * np.log(nu + hypot)
        - 0.5 * np.log(2.0 * np.pi * hypot)
    )

    log_c = (
        nu * np.log(kappa_safe)
        - (d / 2.0) * _LOG_2PI
        - log_bessel_approx
    )

    return log_c.astype(np.float32)


def ppr_score_matrix(
    predictions: np.ndarray,
    gallery: np.ndarray,
    kappas: np.ndarray,
    d: Optional[int] = None,
    normalize_gallery: bool = True,
) -> np.ndarray:
    """Compute PPR (Posterior Predictive Retrieval) score matrix.

    PPR_score(i, j) = kappa_i * cos(mu_i, z_j) + log C_d(kappa_i)

    The log-normalizer term means that confident queries (high kappa)
    get boosted scores relative to uncertain queries. This naturally
    downweights hub-prone uncertain predictions.

    Args:
        predictions: (N, D) L2-normalized predicted mean directions (mu).
        gallery: (M, D) L2-normalized gallery embeddings.
        kappas: (N,) positive concentration parameters per query.
        d: Embedding dimension. If None, inferred from predictions.shape[1].
        normalize_gallery: If True, L2-normalize gallery (safety check).

    Returns:
        (N, M) PPR score matrix.
    """
    predictions = np.asarray(predictions, dtype=np.float32)
    gallery = np.asarray(gallery, dtype=np.float32)
    kappas = np.asarray(kappas, dtype=np.float32).ravel()

    if d is None:
        d = predictions.shape[1]

    N, D = predictions.shape
    M = gallery.shape[0]

    if kappas.shape[0] != N:
        raise ValueError(
            f"kappas length {kappas.shape[0]} != predictions rows {N}"
        )

    if normalize_gallery:
        norms = np.linalg.norm(gallery, axis=1, keepdims=True)
        gallery = gallery / np.maximum(norms, 1e-8)

    cosine_sim = predictions @ gallery.T  # (N, M)

    log_c = _log_vmf_normalizer(kappas, d)  # (N,)

    scores = kappas[:, None] * cosine_sim + log_c[:, None]

    return scores


def ppr_kappa_weighted_score_matrix(
    predictions: np.ndarray,
    gallery: np.ndarray,
    kappas: np.ndarray,
) -> np.ndarray:
    """Simplified PPR: kappa-weighted cosine similarity (no normalizer).

    score(i, j) = kappa_i * cos(mu_i, z_j)

    Within a single query row, this produces the same ranking as raw cosine.
    The kappa weighting only matters for cross-query comparisons (e.g.,
    gallery-side hubness correction). This is the minimal PPR variant.

    Args:
        predictions: (N, D) L2-normalized predictions.
        gallery: (M, D) L2-normalized gallery.
        kappas: (N,) concentrations.

    Returns:
        (N, M) kappa-weighted score matrix.
    """
    predictions = np.asarray(predictions, dtype=np.float32)
    gallery = np.asarray(gallery, dtype=np.float32)
    kappas = np.asarray(kappas, dtype=np.float32).ravel()

    cosine_sim = predictions @ gallery.T
    return kappas[:, None] * cosine_sim


def ppr_csls_score_matrix(
    predictions: np.ndarray,
    gallery: np.ndarray,
    kappas: np.ndarray,
    k: int = 10,
    d: Optional[int] = None,
    use_full_normalizer: bool = True,
) -> np.ndarray:
    """PPR scoring with CSLS-style local scaling on PPR scores.

    Applies the CSLS mean-NN correction to PPR scores rather than
    raw cosines. This combines Bayesian confidence weighting (PPR)
    with cross-domain hubness correction (CSLS).

    CSLS_PPR(i,j) = 2 * PPR(i,j) - r_X(i) - r_Y(j)

    where r_X(i) = mean of top-k PPR(i, :) and
          r_Y(j) = mean of top-k PPR(:, j).

    Args:
        predictions: (N, D) L2-normalized.
        gallery: (M, D) L2-normalized.
        kappas: (N,) concentrations.
        k: Number of neighbors for CSLS.
        d: Embedding dimension.
        use_full_normalizer: If True, include log C_d(kappa) in PPR scores.

    Returns:
        (N, M) PPR-CSLS score matrix.
    """
    if use_full_normalizer:
        scores = ppr_score_matrix(predictions, gallery, kappas, d=d)
    else:
        scores = ppr_kappa_weighted_score_matrix(predictions, gallery, kappas)

    k_eff = min(k, scores.shape[1] - 1, scores.shape[0] - 1)
    if k_eff < 1:
        return scores

    r_x = np.sort(scores, axis=1)[:, -k_eff:].mean(axis=1)
    r_y = np.sort(scores, axis=0)[-k_eff:, :].mean(axis=0)

    return 2.0 * scores - r_x[:, None] - r_y[None, :]


def compute_ppr_retrieval_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    kappas: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10),
    csls_k: int = 10,
    d: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute retrieval metrics under multiple scoring regimes.

    Compares: raw cosine, CSLS, PPR (kappa-weighted), PPR (full normalizer),
    PPR-CSLS (kappa-weighted + CSLS), PPR-CSLS (full + CSLS).

    Args:
        predictions: (N, D) L2-normalized predictions.
        ground_truth: (N, D) L2-normalized targets (gallery = ground_truth).
        kappas: (N,) concentration parameters per query.
        ks: Top-k values for accuracy computation.
        csls_k: Number of neighbors for CSLS variants.
        d: Embedding dimension.

    Returns:
        Dict mapping scoring method name to metric dict.
    """
    N, D = predictions.shape
    if d is None:
        d = D

    results: Dict[str, Dict[str, float]] = {}

    scoring_methods = {
        "raw_cosine": lambda: predictions @ ground_truth.T,
        "csls": lambda: _csls_from_cosine(predictions, ground_truth, csls_k),
        "ppr_kappa_only": lambda: ppr_kappa_weighted_score_matrix(
            predictions, ground_truth, kappas
        ),
        "ppr_full": lambda: ppr_score_matrix(
            predictions, ground_truth, kappas, d=d
        ),
        "ppr_csls_kappa": lambda: ppr_csls_score_matrix(
            predictions, ground_truth, kappas, k=csls_k,
            use_full_normalizer=False,
        ),
        "ppr_csls_full": lambda: ppr_csls_score_matrix(
            predictions, ground_truth, kappas, k=csls_k, d=d,
            use_full_normalizer=True,
        ),
    }

    for method_name, score_fn in scoring_methods.items():
        scores = score_fn()
        ranks = _correct_ranks_from_scores(scores)
        metrics = _retrieval_metrics_from_ranks(ranks, ks)

        kappa_stats = {
            "kappa_mean": float(np.mean(kappas)),
            "kappa_std": float(np.std(kappas)),
            "kappa_min": float(np.min(kappas)),
            "kappa_max": float(np.max(kappas)),
        }
        metrics.update(kappa_stats)

        results[method_name] = metrics
        logger.info(
            "PPR [%s]: R@1=%.3f  R@5=%.3f  R@10=%.3f  MRR=%.4f  MedR=%.1f",
            method_name,
            metrics.get("top1_accuracy", 0),
            metrics.get("top5_accuracy", 0),
            metrics.get("top10_accuracy", 0),
            metrics.get("mrr", 0),
            metrics.get("median_rank", 0),
        )

    return results


def _csls_from_cosine(
    predictions: np.ndarray,
    gallery: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Standard CSLS on cosine similarity."""
    sim = predictions @ gallery.T
    k_eff = min(k, sim.shape[1] - 1, sim.shape[0] - 1)
    if k_eff < 1:
        return sim
    r_x = np.sort(sim, axis=1)[:, -k_eff:].mean(axis=1)
    r_y = np.sort(sim, axis=0)[-k_eff:, :].mean(axis=0)
    return 2.0 * sim - r_x[:, None] - r_y[None, :]


def _correct_ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Get 0-indexed rank of diagonal (correct) entries."""
    N = scores.shape[0]
    ranks = np.argsort(-scores, axis=1)
    correct_ranks = np.zeros(N, dtype=np.int32)
    for i in range(N):
        correct_ranks[i] = np.where(ranks[i] == i)[0][0]
    return correct_ranks


def _retrieval_metrics_from_ranks(
    correct_ranks: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """Standard retrieval metrics from correct rank array."""
    results: Dict[str, float] = {}
    for k in ks:
        results[f"top{k}_accuracy"] = float((correct_ranks < k).mean())
    results["mean_rank"] = float(correct_ranks.mean() + 1)
    results["median_rank"] = float(np.median(correct_ranks) + 1)
    results["mrr"] = float((1.0 / (correct_ranks + 1.0)).mean())
    return results
