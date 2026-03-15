"""
Two-stage retrieval pipeline for V30 triple-head architecture.

Stage A (shortlist):
    Uses compact retrieval space (e.g. 768-D L2-normed) to find the top-K
    candidates per query.  Optionally applies CSLS correction.

Stage B (rerank):
    Reranks the shortlist using the dedicated stage-2 embeddings
    (e.g. rerank-head outputs) with cosine similarity
    (inputs are normalised internally).

Reports:
    - Shortlist recall at K
    - Reranked R@1, R@5, R@10
    - Hubness comparison between compact and rich spaces
"""

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_FUSION_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "compact_score": "csls",
    "family": "normalized_weighted",
    "normalization": "zscore",
    "shortlist_k": 50,
    "alpha": 0.8,
    "csls_k": 10,
}


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, D) x (M, D) -> (N, M) cosine similarity matrix."""
    a_n = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b_n = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    return a_n @ b_n.T


def _csls_scores(
    preds: np.ndarray, gallery: np.ndarray, k: int = 10,
) -> np.ndarray:
    """CSLS similarity: sim(p,g) - 0.5*(r_T(p) + r_S(g))."""
    sim = _cosine_sim(preds, gallery)
    top_k_pred = np.sort(sim, axis=1)[:, -k:].mean(axis=1, keepdims=True)
    top_k_gal = np.sort(sim, axis=0)[-k:, :].mean(axis=0, keepdims=True)
    return sim - 0.5 * (top_k_pred + top_k_gal)


def shortlist_retrieval(
    compact_preds: np.ndarray,
    compact_gallery: np.ndarray,
    k: int = 100,
    use_csls: bool = True,
    csls_k: int = 10,
) -> np.ndarray:
    """Stage A: build per-query shortlist from compact space.

    Parameters
    ----------
    compact_preds : (N, D_compact) L2-normed query embeddings
    compact_gallery : (M, D_compact) L2-normed gallery embeddings
    k : shortlist size
    use_csls : whether to use CSLS scoring
    csls_k : neighbourhood size for CSLS

    Returns
    -------
    shortlist_indices : (N, k) gallery indices per query, sorted by score desc
    """
    if use_csls:
        scores = _csls_scores(compact_preds, compact_gallery, k=csls_k)
    else:
        scores = _cosine_sim(compact_preds, compact_gallery)

    k_eff = min(k, scores.shape[1])
    indices = np.argpartition(-scores, k_eff, axis=1)[:, :k_eff]
    for i in range(len(indices)):
        order = np.argsort(-scores[i, indices[i]])
        indices[i] = indices[i, order]

    return indices


def rerank_shortlist(
    rich_preds: np.ndarray,
    rich_gallery: np.ndarray,
    shortlist_indices: np.ndarray,
    mode: str = "cosine",
) -> np.ndarray:
    """Stage B: rerank shortlist using stage-2 embedding similarity.

    Both rich_preds and rich_gallery are normalised internally when
    mode='cosine' (default).  mode='dot' skips normalisation.

    Parameters
    ----------
    rich_preds : (N, D_stage2) query embeddings
    rich_gallery : (M, D_stage2) gallery embeddings
    shortlist_indices : (N, K) indices into gallery
    mode : 'cosine' or 'dot'

    Returns
    -------
    reranked_indices : (N, K) indices reordered by stage-2 score desc
    """
    N, K = shortlist_indices.shape

    if mode == "cosine":
        rp = rich_preds / np.maximum(
            np.linalg.norm(rich_preds, axis=-1, keepdims=True), 1e-8)
        rg = rich_gallery / np.maximum(
            np.linalg.norm(rich_gallery, axis=-1, keepdims=True), 1e-8)
    else:
        rp, rg = rich_preds, rich_gallery

    reranked = np.zeros_like(shortlist_indices)
    for i in range(N):
        sl_idx = shortlist_indices[i]
        sl_embs = rg[sl_idx]  # (K, D_rich)
        sims = sl_embs @ rp[i]  # (K,)
        order = np.argsort(-sims)
        reranked[i] = sl_idx[order]

    return reranked


def _recall_at_k(
    ranked_indices: np.ndarray,
    ground_truth_indices: np.ndarray,
    ks: Sequence[int],
) -> Dict[str, float]:
    """Compute recall@K given ranked retrieval indices.

    ground_truth_indices[i] is the correct gallery index for query i.
    Assumes identity mapping (GT index for query i is i) if
    ground_truth_indices is None.
    """
    results = {}
    N = ranked_indices.shape[0]
    for k in ks:
        k_eff = min(k, ranked_indices.shape[1])
        correct = 0
        for i in range(N):
            gt = ground_truth_indices[i] if ground_truth_indices is not None else i
            if gt in ranked_indices[i, :k_eff]:
                correct += 1
        results[f"r@{k}"] = correct / max(N, 1)
    return results


def _gt_rank_from_scores(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ranked gallery indices and 1-based GT rank for each query."""
    order = np.argsort(-scores, axis=1)
    gt_rank = np.argmax(order == np.arange(scores.shape[0])[:, None], axis=1) + 1
    return order, gt_rank.astype(np.int32)


def _metrics_from_gt_rank(
    gt_rank: np.ndarray,
    ks: Sequence[int],
) -> Dict[str, float]:
    """Compute retrieval metrics from 1-based GT ranks."""
    gt_rank = gt_rank.astype(np.int32)
    metrics: Dict[str, float] = {
        "median_rank": float(np.median(gt_rank)),
        "mrr": float(np.mean(1.0 / np.maximum(gt_rank, 1))),
    }
    for k in ks:
        metrics[f"r@{k}"] = float(np.mean(gt_rank <= k))
    return metrics


def _prefix_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_{k}": float(v)
        for k, v in metrics.items()
    }


def _normalize_shortlist_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    """Normalise shortlist-local scores per query before weighted fusion."""
    if mode == "none":
        return scores
    if mode == "zscore":
        mean = scores.mean(axis=1, keepdims=True)
        std = scores.std(axis=1, keepdims=True)
        return (scores - mean) / np.maximum(std, 1e-8)
    if mode == "minmax":
        s_min = scores.min(axis=1, keepdims=True)
        s_max = scores.max(axis=1, keepdims=True)
        return (scores - s_min) / np.maximum(s_max - s_min, 1e-8)
    if mode == "stdscale":
        std = scores.std(axis=1, keepdims=True)
        return scores / np.maximum(std, 1e-8)
    raise ValueError(f"Unknown fusion normalization mode: {mode}")


def _rank_within_shortlist(scores: np.ndarray) -> np.ndarray:
    """Convert scores to 1-based ranks within each shortlist row."""
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(scores.shape[0])[:, None], order] = (
        np.arange(scores.shape[1])[None, :] + 1
    )
    return ranks


def _k_occurrence_stats(
    ranked_indices: np.ndarray,
    gallery_size: int,
    k: int = 1,
) -> Dict[str, float]:
    """Hubness statistics from top-K occurrences."""
    topk = ranked_indices[:, :k]
    counts = np.bincount(topk.ravel(), minlength=gallery_size)
    from scipy.stats import skew as sp_skew
    return {
        "skewness": float(sp_skew(counts)),
        "hub_fraction": float((counts > (2 * k)).mean()),
        "antihub_fraction": float((counts == 0).mean()),
    }


def _mean_inter_embedding_cosine(x: np.ndarray, sample_size: int = 2000) -> float:
    """Mean off-diagonal cosine similarity among embeddings."""
    x_n = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
    n = x_n.shape[0]
    if n <= 1:
        return 0.0
    if n > sample_size:
        idx = np.random.RandomState(99).choice(n, sample_size, replace=False)
        x_n = x_n[idx]
        n = x_n.shape[0]
    sim = x_n @ x_n.T
    np.fill_diagonal(sim, 0.0)
    return float(sim.sum() / (n * (n - 1)))


def fusion_metrics(
    compact_preds: np.ndarray,
    compact_gts: np.ndarray,
    rich_preds: np.ndarray,
    rich_gts: np.ndarray,
    shortlist_k: int = 50,
    ks: Sequence[int] = (1, 5, 10),
    compact_score: str = "csls",
    family: str = "normalized_weighted",
    normalization: str = "zscore",
    alpha: float = 0.8,
    csls_k: int = 10,
    rerank_mode: str = "cosine",
) -> Dict[str, Any]:
    """Evaluate fixed score fusion inside a compact shortlist.

    The GT rank is computed in the full gallery by taking the fused shortlist
    rank when the GT is shortlisted, and falling back to the chosen compact
    score rank otherwise. This mirrors the post-hoc fusion sweep logic.
    """
    n = compact_preds.shape[0]
    gt_indices = np.arange(n)

    compact_raw_scores = _cosine_sim(compact_preds, compact_gts)
    compact_csls_scores = _csls_scores(compact_preds, compact_gts, k=csls_k)
    compact_scores = compact_csls_scores if compact_score == "csls" else compact_raw_scores
    compact_order, compact_gt_rank = _gt_rank_from_scores(compact_scores)

    if rerank_mode == "cosine":
        rerank_scores = _cosine_sim(rich_preds, rich_gts)
    elif rerank_mode == "dot":
        rerank_scores = rich_preds @ rich_gts.T
    else:
        raise ValueError(f"Unknown rerank_mode: {rerank_mode}")

    shortlist_k_eff = min(shortlist_k, compact_scores.shape[1])
    shortlist = compact_order[:, :shortlist_k_eff]
    row_idx = np.arange(n)[:, None]
    compact_sl = compact_scores[row_idx, shortlist]
    rerank_sl = rerank_scores[row_idx, shortlist]

    shortlist_recall = {
        f"r@{shortlist_k_eff}": float(np.mean(np.any(shortlist == gt_indices[:, None], axis=1)))
    }

    if family == "weighted":
        fused_scores = alpha * compact_sl + (1.0 - alpha) * rerank_sl
    elif family == "normalized_weighted":
        compact_norm = _normalize_shortlist_scores(compact_sl, normalization)
        rerank_norm = _normalize_shortlist_scores(rerank_sl, normalization)
        fused_scores = alpha * compact_norm + (1.0 - alpha) * rerank_norm
    elif family == "rrf":
        compact_rank = _rank_within_shortlist(compact_sl)
        rerank_rank = _rank_within_shortlist(rerank_sl)
        fused_scores = 1.0 / (60.0 + compact_rank) + 1.0 / (60.0 + rerank_rank)
    else:
        raise ValueError(f"Unknown fusion family: {family}")

    fused_order_local = np.argsort(-fused_scores, axis=1)
    fused_shortlist = shortlist[row_idx, fused_order_local]
    fused_gt_rank = compact_gt_rank.copy()
    hit_rows = np.where(np.any(shortlist == gt_indices[:, None], axis=1))[0]
    if hit_rows.size > 0:
        local_gt_rank = np.argmax(
            fused_shortlist[hit_rows] == hit_rows[:, None],
            axis=1,
        ) + 1
        fused_gt_rank[hit_rows] = local_gt_rank.astype(np.int32)

    rerank_only_gt_rank = _gt_rank_from_scores(rerank_scores)[1]
    rerank_replacement_local = np.argsort(-rerank_sl, axis=1)
    reranked_shortlist = shortlist[row_idx, rerank_replacement_local]
    rerank_replacement_gt_rank = compact_gt_rank.copy()
    if hit_rows.size > 0:
        local_gt_rank = np.argmax(
            reranked_shortlist[hit_rows] == hit_rows[:, None],
            axis=1,
        ) + 1
        rerank_replacement_gt_rank[hit_rows] = local_gt_rank.astype(np.int32)

    compact_raw_metrics = _metrics_from_gt_rank(
        _gt_rank_from_scores(compact_raw_scores)[1], ks
    )
    compact_csls_metrics = _metrics_from_gt_rank(
        _gt_rank_from_scores(compact_csls_scores)[1], ks
    )
    rerank_only_metrics = _metrics_from_gt_rank(rerank_only_gt_rank, ks)
    rerank_replacement_metrics = _metrics_from_gt_rank(rerank_replacement_gt_rank, ks)
    fused = _metrics_from_gt_rank(fused_gt_rank, ks)

    report: Dict[str, Any] = {
        "config": {
            "compact_score": compact_score,
            "family": family,
            "normalization": normalization,
            "shortlist_k": int(shortlist_k_eff),
            "alpha": float(alpha),
            "csls_k": int(csls_k),
            "rerank_mode": rerank_mode,
        },
        "shortlist_recall": shortlist_recall,
        "compact_raw": _prefix_metrics(compact_raw_metrics, "compact"),
        "compact_csls": _prefix_metrics(compact_csls_metrics, "compact_csls"),
        "rerank_only": _prefix_metrics(rerank_only_metrics, "rerank"),
        "rerank_replacement": _prefix_metrics(rerank_replacement_metrics, "reranked"),
        "fused": _prefix_metrics(fused, "fused"),
        "fused_gain_over_compact_raw": round(
            fused.get(f"r@{ks[0]}", 0.0) - compact_raw_metrics.get(f"r@{ks[0]}", 0.0),
            4,
        ),
        "fused_gain_over_compact_csls": round(
            fused.get(f"r@{ks[0]}", 0.0) - compact_csls_metrics.get(f"r@{ks[0]}", 0.0),
            4,
        ),
        "fused_gain_over_rerank_replacement": round(
            fused.get(f"r@{ks[0]}", 0.0) - rerank_replacement_metrics.get(f"r@{ks[0]}", 0.0),
            4,
        ),
    }

    logger.info(
        "Fusion retrieval: compact=%s  family=%s  norm=%s  k=%d  alpha=%.2f  "
        "fused_R@1=%.1f%%  compact_csls_R@1=%.1f%%  reranked_R@1=%.1f%%",
        compact_score,
        family,
        normalization,
        shortlist_k_eff,
        alpha,
        report["fused"].get("fused_r@1", 0.0) * 100,
        report["compact_csls"].get("compact_csls_r@1", 0.0) * 100,
        report["rerank_replacement"].get("reranked_r@1", 0.0) * 100,
    )

    return report


def two_stage_metrics(
    compact_preds: np.ndarray,
    compact_gts: np.ndarray,
    rich_preds: np.ndarray,
    rich_gts: np.ndarray,
    shortlist_k: int = 100,
    ks: Sequence[int] = (1, 5, 10),
    use_csls_shortlist: bool = True,
    rerank_mode: str = "cosine",
) -> Dict[str, Any]:
    """Full two-stage retrieval evaluation.

    Assumes aligned samples: query i's ground truth is gallery index i
    (i.e. preds and gts share the same ordering).

    Returns a dict with shortlist recall, reranked recall, and hubness
    comparison between compact and reranked results.
    """
    N = compact_preds.shape[0]
    gt_indices = np.arange(N)

    sl_indices = shortlist_retrieval(
        compact_preds, compact_gts,
        k=shortlist_k, use_csls=use_csls_shortlist,
    )
    sl_recall = _recall_at_k(sl_indices, gt_indices, ks=[shortlist_k])

    reranked = rerank_shortlist(
        rich_preds, rich_gts, sl_indices, mode=rerank_mode,
    )
    reranked_recall = _recall_at_k(reranked, gt_indices, ks=list(ks))

    compact_raw_indices = shortlist_retrieval(
        compact_preds, compact_gts,
        k=max(ks), use_csls=False,
    )
    compact_recall = _recall_at_k(compact_raw_indices, gt_indices, ks=list(ks))

    compact_csls_indices = shortlist_retrieval(
        compact_preds, compact_gts,
        k=max(ks), use_csls=True,
    )
    compact_csls_recall = _recall_at_k(compact_csls_indices, gt_indices, ks=list(ks))

    compact_hub = _k_occurrence_stats(compact_raw_indices, N, k=1)
    reranked_hub = _k_occurrence_stats(reranked, N, k=1)

    # --- Rich-only full-gallery retrieval ---
    rich_sim = _cosine_sim(rich_preds, rich_gts)
    rich_ranks = np.argsort(-rich_sim, axis=1)
    rich_recall = _recall_at_k(rich_ranks, gt_indices, ks=list(ks))

    # --- Rich-space diagnostics ---
    rp_norms = np.linalg.norm(rich_preds, axis=-1)
    rg_norms = np.linalg.norm(rich_gts, axis=-1)
    diag_sims = np.array([rich_sim[i, i] for i in range(N)])
    # Sample off-diagonal negatives
    _rng = np.random.RandomState(42)
    _n_neg = min(50000, N * (N - 1))
    _ni = _rng.randint(0, N, _n_neg)
    _nj = _rng.randint(0, N - 1, _n_neg)
    _nj[_nj >= _ni] += 1
    neg_sims = rich_sim[_ni, _nj]
    rich_separability = float(
        (diag_sims.mean() - neg_sims.mean()) / max(neg_sims.std(), 1e-8))

    # --- Oracle shortlist reranking ---
    oracle_k = min(shortlist_k, N - 1)
    oracle_sl = np.zeros((N, oracle_k), dtype=int)
    _rng2 = np.random.RandomState(123)
    for i in range(N):
        candidates = np.delete(np.arange(N), i)
        chosen = _rng2.choice(candidates, oracle_k - 1, replace=False)
        gt_pos = _rng2.randint(0, oracle_k)
        oracle_sl[i] = np.insert(chosen, gt_pos, i)[:oracle_k]
    oracle_reranked = rerank_shortlist(
        rich_preds, rich_gts, oracle_sl, mode=rerank_mode)
    oracle_recall = _recall_at_k(oracle_reranked, gt_indices, ks=list(ks))

    report: Dict[str, Any] = {
        "shortlist_k": shortlist_k,
        "shortlist_recall": sl_recall,
        "compact_raw": {f"compact_{k}": v for k, v in compact_recall.items()},
        "compact_csls": {f"compact_csls_{k}": v for k, v in compact_csls_recall.items()},
        "reranked": {f"reranked_{k}": v for k, v in reranked_recall.items()},
        "rich_only": {f"rich_{k}": v for k, v in rich_recall.items()},
        "oracle_rerank": {f"oracle_{k}": v for k, v in oracle_recall.items()},
        "hubness_compact": compact_hub,
        "hubness_reranked": reranked_hub,
        "hubness_gap_compact": round(
            compact_csls_recall.get(f"r@{ks[0]}", 0)
            - compact_recall.get(f"r@{ks[0]}", 0), 4),
        "rerank_gain_over_compact_raw": round(
            reranked_recall.get(f"r@{ks[0]}", 0)
            - compact_recall.get(f"r@{ks[0]}", 0), 4),
        "rerank_gain_over_csls": round(
            reranked_recall.get(f"r@{ks[0]}", 0)
            - compact_csls_recall.get(f"r@{ks[0]}", 0), 4),
        "rich_diagnostics": {
            "rich_pred_norm_mean": float(rp_norms.mean()),
            "rich_pred_norm_std": float(rp_norms.std()),
            "rich_gt_norm_mean": float(rg_norms.mean()),
            "rich_gt_norm_std": float(rg_norms.std()),
            "inter_pred_cosine_mean": _mean_inter_embedding_cosine(rich_preds),
            "inter_gt_cosine_mean": _mean_inter_embedding_cosine(rich_gts),
            "pos_cosine_mean": float(diag_sims.mean()),
            "pos_cosine_std": float(diag_sims.std()),
            "neg_cosine_mean": float(neg_sims.mean()),
            "neg_cosine_std": float(neg_sims.std()),
            "separability": rich_separability,
        },
    }

    logger.info(
        "Two-stage retrieval: shortlist_recall@%d=%.1f%%  "
        "reranked_R@1=%.1f%%  compact_raw_R@1=%.1f%%  "
        "compact_csls_R@1=%.1f%%  rerank_gain=%.1f pp",
        shortlist_k,
        sl_recall.get(f"r@{shortlist_k}", 0) * 100,
        reranked_recall.get("r@1", 0) * 100,
        compact_recall.get("r@1", 0) * 100,
        compact_csls_recall.get("r@1", 0) * 100,
        report["rerank_gain_over_compact_raw"] * 100,
    )
    logger.info(
        "Rich-only R@1=%.1f%%  oracle_rerank_R@1=%.1f%%  "
        "rich_separability=%.2f  pos_cos=%.4f  neg_cos=%.4f",
        rich_recall.get("r@1", 0) * 100,
        oracle_recall.get("r@1", 0) * 100,
        rich_separability,
        diag_sims.mean(), neg_sims.mean(),
    )

    return report
