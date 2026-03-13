"""
Kappa calibration and reliability diagnostics for vMF models.

Provides:
- Binned analysis: R@1, mean rank, cosine similarity per kappa percentile bin
- Spearman correlations: kappa vs rank, kappa vs correctness
- Reliability-style diagram data: expected accuracy vs mean kappa per bin
- Kappa vs regression error (if rich predictions available)
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


def _retrieval_ranks(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """Compute the rank of the diagonal (correct) match for each query.

    Uses cosine similarity.  Returns (N,) array of 1-based ranks.
    """
    p_norm = preds / np.maximum(np.linalg.norm(preds, axis=-1, keepdims=True), 1e-8)
    g_norm = gts / np.maximum(np.linalg.norm(gts, axis=-1, keepdims=True), 1e-8)
    sim = p_norm @ g_norm.T  # (N, N)
    ranks = np.zeros(sim.shape[0], dtype=np.int64)
    for i in range(sim.shape[0]):
        ranks[i] = int((sim[i] > sim[i, i]).sum()) + 1
    return ranks


def compute_kappa_calibration_report(
    preds: np.ndarray,
    gts: np.ndarray,
    kappas: np.ndarray,
    n_bins: int = 10,
    rich_preds: Optional[np.ndarray] = None,
    rich_gts: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Bin samples by kappa percentile and report per-bin retrieval quality.

    Parameters
    ----------
    preds : (N, D) compact predictions (L2-normed recommended)
    gts : (N, D) compact ground truth
    kappas : (N,) kappa values
    n_bins : number of equal-frequency bins
    rich_preds : (N, D_rich) optional rich regression predictions
    rich_gts : (N, D_rich) optional rich ground truth

    Returns
    -------
    dict with keys:
        bins : list of per-bin dicts {kappa_lo, kappa_hi, kappa_mean, count,
               r_at_1, mean_rank, mean_cosine, [mean_reg_mse]}
        spearman_kappa_rank : Spearman(kappa, rank) correlation
        spearman_kappa_correct : Spearman(kappa, top1_correct)
        spearman_kappa_cosine : Spearman(kappa, diagonal_cosine)
        overall_r_at_1 : overall R@1
    """
    N = len(kappas)
    assert preds.shape[0] == N == gts.shape[0], "Shape mismatch"

    ranks = _retrieval_ranks(preds, gts)
    correct = (ranks == 1).astype(np.float64)

    p_norm = preds / np.maximum(np.linalg.norm(preds, axis=-1, keepdims=True), 1e-8)
    g_norm = gts / np.maximum(np.linalg.norm(gts, axis=-1, keepdims=True), 1e-8)
    diag_cosine = np.sum(p_norm * g_norm, axis=-1)

    has_rich = rich_preds is not None and rich_gts is not None
    if has_rich:
        per_sample_mse = np.mean((rich_preds - rich_gts) ** 2, axis=-1)

    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(kappas, percentiles)

    bins = []
    for b in range(n_bins):
        lo, hi = float(bin_edges[b]), float(bin_edges[b + 1])
        if b < n_bins - 1:
            mask = (kappas >= lo) & (kappas < hi)
        else:
            mask = (kappas >= lo) & (kappas <= hi)

        if mask.sum() == 0:
            continue

        entry: Dict[str, Any] = {
            "kappa_lo": round(lo, 4),
            "kappa_hi": round(hi, 4),
            "kappa_mean": round(float(kappas[mask].mean()), 4),
            "count": int(mask.sum()),
            "r_at_1": round(float(correct[mask].mean()), 4),
            "mean_rank": round(float(ranks[mask].mean()), 2),
            "mean_cosine": round(float(diag_cosine[mask].mean()), 4),
        }
        if has_rich:
            entry["mean_reg_mse"] = round(float(per_sample_mse[mask].mean()), 6)
        bins.append(entry)

    sp_rank = sp_stats.spearmanr(kappas, ranks)
    sp_correct = sp_stats.spearmanr(kappas, correct)
    sp_cosine = sp_stats.spearmanr(kappas, diag_cosine)

    report: Dict[str, Any] = {
        "n_samples": N,
        "n_bins": n_bins,
        "bins": bins,
        "spearman_kappa_rank": {
            "rho": round(float(sp_rank.statistic), 4),
            "pvalue": float(sp_rank.pvalue),
        },
        "spearman_kappa_correct": {
            "rho": round(float(sp_correct.statistic), 4),
            "pvalue": float(sp_correct.pvalue),
        },
        "spearman_kappa_cosine": {
            "rho": round(float(sp_cosine.statistic), 4),
            "pvalue": float(sp_cosine.pvalue),
        },
        "overall_r_at_1": round(float(correct.mean()), 4),
        "kappa_stats": {
            "mean": round(float(kappas.mean()), 4),
            "std": round(float(kappas.std()), 4),
            "min": round(float(kappas.min()), 4),
            "max": round(float(kappas.max()), 4),
        },
    }

    if has_rich:
        sp_reg = sp_stats.spearmanr(kappas, per_sample_mse)
        report["spearman_kappa_reg_mse"] = {
            "rho": round(float(sp_reg.statistic), 4),
            "pvalue": float(sp_reg.pvalue),
        }

    logger.info(
        "Kappa calibration: Spearman(kappa,rank)=%.3f  "
        "Spearman(kappa,correct)=%.3f  overall R@1=%.1f%%",
        report["spearman_kappa_rank"]["rho"],
        report["spearman_kappa_correct"]["rho"],
        report["overall_r_at_1"] * 100,
    )

    return report
