#!/usr/bin/env python3
"""Sweep tri-expert post-hoc retrieval fusion on saved experiment outputs.

Experts:
    1. compact shortlist expert from V32/V33
    2. rerank expert from V32/V33
    3. legacy N1v28a token-space retrieval expert

Selection happens on VAL only. The best VAL setting is then frozen and
evaluated on SHARED1000 exactly once.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SHORTLIST_KS = [50, 100, 150, 200]
WEIGHT_STEP = 0.05
WEIGHT_TRIPLES = [
    (round(a * WEIGHT_STEP, 2), round(b * WEIGHT_STEP, 2), round(c * WEIGHT_STEP, 2))
    for a in range(int(1 / WEIGHT_STEP) + 1)
    for b in range(int(1 / WEIGHT_STEP) + 1 - a)
    for c in [int(round((1.0 - a * WEIGHT_STEP - b * WEIGHT_STEP) / WEIGHT_STEP))]
    if 0 <= c <= int(1 / WEIGHT_STEP) and abs((a + b + c) * WEIGHT_STEP - 1.0) < 1e-8
]
COMPACT_SCORE_VARIANTS = ["raw_cosine", "csls"]
LEGACY_SCORE_VARIANTS = ["raw_cosine", "csls"]
NORMALIZATION_MODES = ["none", "zscore", "minmax", "stdscale"]
RRF_K = 60.0
ADAPTIVE_RIDGE = 1e-3


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _l2_normalize(a) @ _l2_normalize(b).T


def _csls_scores(preds: np.ndarray, gallery: np.ndarray, k: int = 10) -> np.ndarray:
    sim = _cosine_sim(preds, gallery)
    top_k_pred = np.sort(sim, axis=1)[:, -k:].mean(axis=1, keepdims=True)
    top_k_gal = np.sort(sim, axis=0)[-k:, :].mean(axis=0, keepdims=True)
    return sim - 0.5 * (top_k_pred + top_k_gal)


def _metrics_from_gt_rank(gt_rank: np.ndarray) -> dict[str, float]:
    gt_rank = gt_rank.astype(np.int32)
    return {
        "R@1": float(np.mean(gt_rank <= 1)),
        "R@5": float(np.mean(gt_rank <= 5)),
        "R@10": float(np.mean(gt_rank <= 10)),
        "median_rank": float(np.median(gt_rank)),
        "MRR": float(np.mean(1.0 / np.maximum(gt_rank, 1))),
    }


def _gt_rank_from_scores(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, axis=1)
    gt_rank = np.argmax(order == np.arange(scores.shape[0])[:, None], axis=1) + 1
    return order, gt_rank.astype(np.int32)


def _normalize_shortlist_scores(scores: np.ndarray, mode: str) -> np.ndarray:
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
    raise ValueError(f"Unknown normalization mode: {mode}")


def _rank_within_shortlist(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(scores.shape[0])[:, None], order] = np.arange(scores.shape[1])[None, :] + 1
    return ranks


def _score_variant_name(kind: str, variant: str) -> str:
    return f"{kind}_{variant}" if variant != "rerank" else "rerank"


def _best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        results,
        key=lambda r: (
            r["R@1"],
            r["R@5"],
            r["R@10"],
            r["MRR"],
            -r["median_rank"],
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_required(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path)


def _load_optional(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def _row_key_matrix(arr: np.ndarray, decimals: int = 6) -> list[bytes]:
    arr32 = np.ascontiguousarray(np.round(arr.astype(np.float32, copy=False), decimals))
    return [arr32[i].tobytes() for i in range(arr32.shape[0])]


def _infer_subject_from_results_dir(results_dir: Path) -> str:
    for candidate in [results_dir.name, results_dir.parent.name]:
        if candidate.startswith("subj"):
            return candidate
    raise ValueError(f"Could not infer subject from results dir: {results_dir}")


def _load_canonical_shared1000_ids(subject: str) -> np.ndarray:
    index_path = Path("data/indices/nsd_index") / f"subject={subject}" / "index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(
            f"shared1000 fallback needs index parquet, but it was not found: {index_path}"
        )
    index_df = pd.read_parquet(index_path)
    if "shared1000" not in index_df.columns or "nsdId" not in index_df.columns:
        raise ValueError(
            f"shared1000 fallback needs 'shared1000' and 'nsdId' columns in {index_path}"
        )
    s1000_mask = index_df["shared1000"].fillna(False).astype(bool).values
    if not np.any(s1000_mask):
        raise ValueError(f"shared1000 fallback found zero shared1000 rows in {index_path}")
    return np.unique(index_df.loc[s1000_mask, "nsdId"].astype(np.int32).values)


def _recover_nsd_ids_from_reference(
    prefix: str,
    gt_embeddings: np.ndarray,
    reference_split: dict[str, np.ndarray],
) -> np.ndarray:
    ref_gts = reference_split.get("compact_gts")
    ref_ids = reference_split.get("nsd_ids")
    if ref_gts is None or ref_ids is None:
        raise FileNotFoundError(
            f"{prefix}: cannot recover nsd_ids without a reference split containing "
            "compact_gts and nsd_ids"
        )
    if gt_embeddings.shape[1] != ref_gts.shape[1]:
        raise ValueError(
            f"{prefix}: cannot recover nsd_ids from reference because GT dims differ "
            f"({gt_embeddings.shape[1]} vs {ref_gts.shape[1]})"
        )

    ref_map: dict[bytes, int] = {}
    for nid, key in zip(ref_ids.astype(np.int32), _row_key_matrix(ref_gts)):
        if key in ref_map and ref_map[key] != int(nid):
            raise ValueError(f"{prefix}: duplicate reference GT embedding maps to multiple nsd_ids")
        ref_map[key] = int(nid)

    recovered: list[int] = []
    missing = 0
    for key in _row_key_matrix(gt_embeddings):
        nid = ref_map.get(key)
        if nid is None:
            missing += 1
            continue
        recovered.append(nid)
    if missing:
        raise ValueError(
            f"{prefix}: failed to recover nsd_ids for {missing}/{gt_embeddings.shape[0]} rows "
            "from reference GT embeddings"
        )
    return np.asarray(recovered, dtype=np.int32)


def _load_legacy_nsd_ids(
    metrics_dir: Path,
    results_dir: Path,
    prefix: str,
    expected_len: int,
    gt_embeddings: np.ndarray,
    reference_split: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    ids_path = metrics_dir / f"{prefix}_nsd_ids.npy"
    if ids_path.exists():
        nsd_ids = np.load(ids_path).astype(np.int32)
        if nsd_ids.shape[0] != expected_len:
            raise ValueError(
                f"{prefix}: {ids_path.name} has {nsd_ids.shape[0]} ids, expected {expected_len}"
            )
        return nsd_ids

    if prefix == "val":
        split_path = results_dir / "split.json"
        if split_path.exists():
            with open(split_path) as f:
                split_payload = json.load(f)
            val_ids = np.asarray(split_payload.get("val_nsd_ids", []), dtype=np.int32)
            if val_ids.shape[0] == expected_len:
                logger.info(
                    "%s: recovered legacy nsd_ids from %s",
                    prefix,
                    split_path,
                )
                return val_ids
            logger.warning(
                "%s: split.json exists at %s but val_nsd_ids has %d rows, expected %d",
                prefix,
                split_path,
                val_ids.shape[0],
                expected_len,
            )

    if prefix == "shared1000":
        subject = _infer_subject_from_results_dir(results_dir)
        shared_ids = _load_canonical_shared1000_ids(subject)
        if shared_ids.shape[0] == expected_len:
            logger.info(
                "%s: recovered legacy nsd_ids from canonical shared1000 index for %s",
                prefix,
                subject,
            )
            return shared_ids
        logger.warning(
            "%s: canonical shared1000 ids for %s has %d rows, expected %d",
            prefix,
            subject,
            shared_ids.shape[0],
            expected_len,
        )

    if reference_split is not None:
        logger.info(
            "%s: recovering legacy nsd_ids by matching GT embeddings to tri-expert reference",
            prefix,
        )
        return _recover_nsd_ids_from_reference(prefix, gt_embeddings, reference_split)

    raise FileNotFoundError(
        f"Missing required file: {ids_path}. No safe fallback was available."
    )


def _resolve_metric_file(metrics_dir: Path, prefix: str, candidates: list[str]) -> Path:
    for name in candidates:
        path = metrics_dir / f"{prefix}_{name}.npy"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not resolve {prefix} file in {metrics_dir}. Tried: {candidates}"
    )


def _load_tri_split(metrics_dir: Path, prefix: str) -> dict[str, np.ndarray]:
    compact_preds = _load_required(metrics_dir / f"{prefix}_predictions_compact.npy")
    compact_gts = _load_required(metrics_dir / f"{prefix}_ground_truth_compact.npy")
    rerank_preds = _load_required(metrics_dir / f"{prefix}_predictions_rerank.npy")
    rerank_gts = _load_required(metrics_dir / f"{prefix}_ground_truth_rerank.npy")
    nsd_ids = _load_required(metrics_dir / f"{prefix}_nsd_ids.npy").astype(np.int32)
    kappas = _load_optional(metrics_dir / f"{prefix}_kappas.npy")

    def _optional_component(name_candidates: list[str]) -> np.ndarray | None:
        for name in name_candidates:
            path = metrics_dir / f"{prefix}_{name}.npy"
            if path.exists():
                return np.load(path)
        return None

    component_mu = _optional_component([
        "predictions_compact_component_mu",
        "predictions_compact_components_mu",
    ])
    component_kappa = _optional_component([
        "predictions_compact_component_kappa",
        "predictions_compact_components_kappa",
    ])
    component_logits = _optional_component([
        "predictions_compact_component_logits",
        "predictions_compact_components_logits",
    ])

    if compact_preds.shape[0] != compact_gts.shape[0] or compact_preds.shape[0] != nsd_ids.shape[0]:
        raise ValueError(f"{prefix}: compact arrays and nsd_ids are misaligned")
    if rerank_preds.shape[0] != rerank_gts.shape[0] or rerank_preds.shape[0] != nsd_ids.shape[0]:
        raise ValueError(f"{prefix}: rerank arrays and nsd_ids are misaligned")
    for name, arr in {
        "compact_component_mu": component_mu,
        "compact_component_kappa": component_kappa,
        "compact_component_logits": component_logits,
    }.items():
        if arr is not None and arr.shape[0] != nsd_ids.shape[0]:
            raise ValueError(f"{prefix}: {name} rows {arr.shape[0]} != nsd_ids rows {nsd_ids.shape[0]}")
    out = {
        "nsd_ids": nsd_ids,
        "compact_preds": compact_preds,
        "compact_gts": compact_gts,
        "rerank_preds": rerank_preds,
        "rerank_gts": rerank_gts,
        "compact_kappas": kappas,
    }
    if component_mu is not None:
        out["compact_component_mu"] = component_mu
    if component_kappa is not None:
        out["compact_component_kappa"] = component_kappa
    if component_logits is not None:
        out["compact_component_logits"] = component_logits
    return out


def _load_legacy_split(
    metrics_dir: Path,
    results_dir: Path,
    prefix: str,
    reference_split: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    pred_path = _resolve_metric_file(
        metrics_dir,
        prefix,
        ["predictions_compact", "predictions"],
    )
    gt_path = _resolve_metric_file(
        metrics_dir,
        prefix,
        ["ground_truth_compact", "ground_truth"],
    )
    preds = np.load(pred_path)
    gts = np.load(gt_path)
    nsd_ids = _load_legacy_nsd_ids(
        metrics_dir=metrics_dir,
        results_dir=results_dir,
        prefix=prefix,
        expected_len=preds.shape[0],
        gt_embeddings=gts,
        reference_split=reference_split,
    )
    if preds.shape[0] != gts.shape[0] or preds.shape[0] != nsd_ids.shape[0]:
        raise ValueError(f"{prefix}: legacy arrays and nsd_ids are misaligned")
    return {
        "nsd_ids": nsd_ids,
        "legacy_preds": preds,
        "legacy_gts": gts,
    }


def _align_common_ids(
    tri_split: dict[str, np.ndarray],
    legacy_split: dict[str, np.ndarray],
    split_name: str,
) -> dict[str, np.ndarray]:
    tri_ids = tri_split["nsd_ids"].astype(np.int32)
    legacy_ids = legacy_split["nsd_ids"].astype(np.int32)
    tri_pos = {int(nid): i for i, nid in enumerate(tri_ids)}
    legacy_pos = {int(nid): i for i, nid in enumerate(legacy_ids)}
    common_ids = [int(nid) for nid in tri_ids if int(nid) in legacy_pos]
    if not common_ids:
        raise ValueError(f"{split_name}: no overlapping nsd_ids between tri and legacy results")
    if len(common_ids) != len(tri_ids) or len(common_ids) != len(legacy_ids):
        logger.warning(
            "%s: aligning on %d common nsd_ids (tri=%d, legacy=%d)",
            split_name,
            len(common_ids),
            len(tri_ids),
            len(legacy_ids),
        )
    tri_idx = np.array([tri_pos[nid] for nid in common_ids], dtype=np.int64)
    legacy_idx = np.array([legacy_pos[nid] for nid in common_ids], dtype=np.int64)

    aligned = {
        "nsd_ids": np.array(common_ids, dtype=np.int32),
        "compact_preds": tri_split["compact_preds"][tri_idx],
        "compact_gts": tri_split["compact_gts"][tri_idx],
        "rerank_preds": tri_split["rerank_preds"][tri_idx],
        "rerank_gts": tri_split["rerank_gts"][tri_idx],
        "legacy_preds": legacy_split["legacy_preds"][legacy_idx],
        "legacy_gts": legacy_split["legacy_gts"][legacy_idx],
    }
    tri_kappas = tri_split.get("compact_kappas")
    if tri_kappas is not None:
        aligned["compact_kappas"] = tri_kappas[tri_idx]
    for key in [
        "compact_component_mu",
        "compact_component_kappa",
        "compact_component_logits",
    ]:
        arr = tri_split.get(key)
        if arr is not None:
            aligned[key] = arr[tri_idx]
    return aligned


def _assert_split_disjointness(val_ids: np.ndarray, shared_ids: np.ndarray, label: str) -> None:
    overlap = set(int(x) for x in val_ids).intersection(int(x) for x in shared_ids)
    if overlap:
        raise ValueError(f"{label}: VAL and SHARED1000 overlap on {len(overlap)} nsd_ids")


def _load_existing_two_expert_fusion(metrics_dir: Path, prefix: str) -> dict[str, Any] | None:
    path = metrics_dir / f"{prefix}_fused_metrics.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _build_split_scores(split_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "compact_raw": _cosine_sim(split_arrays["compact_preds"], split_arrays["compact_gts"]),
        "compact_csls": _csls_scores(split_arrays["compact_preds"], split_arrays["compact_gts"], k=10),
        "rerank": _cosine_sim(split_arrays["rerank_preds"], split_arrays["rerank_gts"]),
        "legacy_raw": _cosine_sim(split_arrays["legacy_preds"], split_arrays["legacy_gts"]),
        "legacy_csls": _csls_scores(split_arrays["legacy_preds"], split_arrays["legacy_gts"], k=10),
    }


def _compute_baselines(scores: dict[str, np.ndarray]) -> dict[str, Any]:
    baselines: dict[str, Any] = {}
    for key, score_name in [
        ("compact_raw", "compact_raw"),
        ("compact_csls", "compact_csls"),
        ("rerank_only", "rerank"),
        ("legacy_raw", "legacy_raw"),
        ("legacy_csls", "legacy_csls"),
    ]:
        _, gt_rank = _gt_rank_from_scores(scores[score_name])
        baselines[key] = _metrics_from_gt_rank(gt_rank)
    return baselines


def _query_margins(scores: np.ndarray, shortlist: np.ndarray) -> np.ndarray:
    row_idx = np.arange(scores.shape[0])[:, None]
    local_scores = scores[row_idx, shortlist]
    order = np.argsort(-local_scores, axis=1)
    sorted_scores = np.take_along_axis(local_scores, order, axis=1)
    if sorted_scores.shape[1] < 2:
        return np.zeros(scores.shape[0], dtype=np.float32)
    return (sorted_scores[:, 0] - sorted_scores[:, 1]).astype(np.float32)


def _fit_adaptive_linear(
    compact_sl: np.ndarray,
    rerank_sl: np.ndarray,
    legacy_sl: np.ndarray,
    shortlist: np.ndarray,
    compact_kappas: np.ndarray | None,
) -> dict[str, Any]:
    n, k = shortlist.shape
    compact_margin = _query_margins(compact_sl, np.tile(np.arange(k), (n, 1)))
    rerank_margin = _query_margins(rerank_sl, np.tile(np.arange(k), (n, 1)))
    legacy_margin = _query_margins(legacy_sl, np.tile(np.arange(k), (n, 1)))

    compact_top = np.argmax(compact_sl, axis=1)
    rerank_top = np.argmax(rerank_sl, axis=1)
    legacy_top = np.argmax(legacy_sl, axis=1)
    agree_cr = (compact_top == rerank_top).astype(np.float32)
    agree_cl = (compact_top == legacy_top).astype(np.float32)
    agree_rl = (rerank_top == legacy_top).astype(np.float32)
    kappa_term = compact_kappas.astype(np.float32) if compact_kappas is not None else np.zeros(n, dtype=np.float32)

    feats = []
    labels = []
    for i in range(n):
        for j in range(k):
            feats.append([
                1.0,
                compact_sl[i, j],
                rerank_sl[i, j],
                legacy_sl[i, j],
                compact_sl[i, j] * compact_margin[i],
                rerank_sl[i, j] * rerank_margin[i],
                legacy_sl[i, j] * legacy_margin[i],
                compact_sl[i, j] * kappa_term[i],
                rerank_sl[i, j] * agree_cr[i],
                legacy_sl[i, j] * agree_cl[i],
                float(j == compact_top[i]),
                float(j == rerank_top[i]),
                float(j == legacy_top[i]),
                agree_cr[i],
                agree_cl[i],
                agree_rl[i],
            ])
            labels.append(float(shortlist[i, j] == i))

    x = np.asarray(feats, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    reg = ADAPTIVE_RIDGE * np.eye(x.shape[1], dtype=np.float64)
    weights = np.linalg.solve(x.T @ x + reg, x.T @ y)
    return {
        "weights": weights.astype(np.float32),
        "use_compact_kappa": bool(compact_kappas is not None),
    }


def _apply_adaptive_linear(
    compact_sl: np.ndarray,
    rerank_sl: np.ndarray,
    legacy_sl: np.ndarray,
    shortlist: np.ndarray,
    model: dict[str, Any],
    compact_kappas: np.ndarray | None,
) -> np.ndarray:
    n, k = shortlist.shape
    compact_margin = _query_margins(compact_sl, np.tile(np.arange(k), (n, 1)))
    rerank_margin = _query_margins(rerank_sl, np.tile(np.arange(k), (n, 1)))
    legacy_margin = _query_margins(legacy_sl, np.tile(np.arange(k), (n, 1)))

    compact_top = np.argmax(compact_sl, axis=1)
    rerank_top = np.argmax(rerank_sl, axis=1)
    legacy_top = np.argmax(legacy_sl, axis=1)
    agree_cr = (compact_top == rerank_top).astype(np.float32)
    agree_cl = (compact_top == legacy_top).astype(np.float32)
    agree_rl = (rerank_top == legacy_top).astype(np.float32)
    if model.get("use_compact_kappa", False) and compact_kappas is not None:
        kappa_term = compact_kappas.astype(np.float32)
    else:
        kappa_term = np.zeros(n, dtype=np.float32)

    weights = np.asarray(model["weights"], dtype=np.float32)
    fused = np.zeros((n, k), dtype=np.float32)
    for i in range(n):
        row_feat = np.stack([
            np.ones(k, dtype=np.float32),
            compact_sl[i],
            rerank_sl[i],
            legacy_sl[i],
            compact_sl[i] * compact_margin[i],
            rerank_sl[i] * rerank_margin[i],
            legacy_sl[i] * legacy_margin[i],
            compact_sl[i] * kappa_term[i],
            rerank_sl[i] * agree_cr[i],
            legacy_sl[i] * agree_cl[i],
            (np.arange(k) == compact_top[i]).astype(np.float32),
            (np.arange(k) == rerank_top[i]).astype(np.float32),
            (np.arange(k) == legacy_top[i]).astype(np.float32),
            np.full(k, agree_cr[i], dtype=np.float32),
            np.full(k, agree_cl[i], dtype=np.float32),
            np.full(k, agree_rl[i], dtype=np.float32),
        ], axis=1)
        fused[i] = row_feat @ weights
    return fused


def _evaluate_setting(
    compact_scores: np.ndarray,
    rerank_scores: np.ndarray,
    legacy_scores: np.ndarray,
    compact_order: np.ndarray,
    compact_gt_rank: np.ndarray,
    shortlist_k: int,
    family: str,
    normalization: str,
    weights: tuple[float, float, float] | None,
    compact_kappas: np.ndarray | None = None,
    adaptive_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n = compact_scores.shape[0]
    shortlist = compact_order[:, :shortlist_k]
    row_idx = np.arange(n)[:, None]
    compact_sl = compact_scores[row_idx, shortlist]
    rerank_sl = rerank_scores[row_idx, shortlist]
    legacy_sl = legacy_scores[row_idx, shortlist]

    if family == "weighted":
        alpha, beta, gamma = weights
        fused = alpha * compact_sl + beta * rerank_sl + gamma * legacy_sl
    elif family == "normalized_weighted":
        alpha, beta, gamma = weights
        fused = (
            alpha * _normalize_shortlist_scores(compact_sl, normalization)
            + beta * _normalize_shortlist_scores(rerank_sl, normalization)
            + gamma * _normalize_shortlist_scores(legacy_sl, normalization)
        )
    elif family == "rrf":
        alpha, beta, gamma = weights
        fused = (
            alpha / (RRF_K + _rank_within_shortlist(compact_sl))
            + beta / (RRF_K + _rank_within_shortlist(rerank_sl))
            + gamma / (RRF_K + _rank_within_shortlist(legacy_sl))
        )
    elif family == "rank_average":
        alpha, beta, gamma = weights
        fused = -(
            alpha * _rank_within_shortlist(compact_sl)
            + beta * _rank_within_shortlist(rerank_sl)
            + gamma * _rank_within_shortlist(legacy_sl)
        )
    elif family == "adaptive_linear":
        fused = _apply_adaptive_linear(
            compact_sl=compact_sl,
            rerank_sl=rerank_sl,
            legacy_sl=legacy_sl,
            shortlist=shortlist,
            model=adaptive_model,
            compact_kappas=compact_kappas,
        )
    else:
        raise ValueError(f"Unknown fusion family: {family}")

    local_order = np.argsort(-fused, axis=1)
    reranked_shortlist = shortlist[row_idx, local_order]
    gt_rank = compact_gt_rank.copy()
    gt_ids = np.arange(n)
    hit_mask = np.any(shortlist == gt_ids[:, None], axis=1)
    if np.any(hit_mask):
        hit_rows = np.where(hit_mask)[0]
        local_gt_rank = np.argmax(
            reranked_shortlist[hit_rows] == hit_rows[:, None],
            axis=1,
        ) + 1
        gt_rank[hit_rows] = local_gt_rank.astype(np.int32)

    metrics = _metrics_from_gt_rank(gt_rank)
    result: dict[str, Any] = {
        "shortlist_k": int(shortlist_k),
        "fusion_family": family,
        "normalization": normalization,
        **metrics,
    }
    if weights is not None:
        result["alpha"] = float(weights[0])
        result["beta"] = float(weights[1])
        result["gamma"] = float(weights[2])
    else:
        result["alpha"] = None
        result["beta"] = None
        result["gamma"] = None
    if adaptive_model is not None:
        result["adaptive_use_compact_kappa"] = bool(adaptive_model.get("use_compact_kappa", False))
        result["adaptive_num_weights"] = int(len(adaptive_model.get("weights", [])))
    return result


def _evaluate_split(
    split_arrays: dict[str, np.ndarray],
    include_adaptive: bool,
    shared_has_kappa: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores = _build_split_scores(split_arrays)
    baselines = _compute_baselines(scores)

    compact_orders: dict[str, np.ndarray] = {}
    compact_gt_ranks: dict[str, np.ndarray] = {}
    for variant_name, score_name in [("raw_cosine", "compact_raw"), ("csls", "compact_csls")]:
        compact_orders[variant_name], compact_gt_ranks[variant_name] = _gt_rank_from_scores(scores[score_name])

    sweep_results: list[dict[str, Any]] = []
    compact_kappas = split_arrays.get("compact_kappas")

    for compact_variant in COMPACT_SCORE_VARIANTS:
        compact_scores = scores["compact_raw"] if compact_variant == "raw_cosine" else scores["compact_csls"]
        compact_order = compact_orders[compact_variant]
        compact_gt_rank = compact_gt_ranks[compact_variant]

        for legacy_variant in LEGACY_SCORE_VARIANTS:
            legacy_scores = scores["legacy_raw"] if legacy_variant == "raw_cosine" else scores["legacy_csls"]
            rerank_scores = scores["rerank"]

            for shortlist_k in SHORTLIST_KS:
                for weights in WEIGHT_TRIPLES:
                    for family in ["weighted", "rrf", "rank_average"]:
                        result = _evaluate_setting(
                            compact_scores=compact_scores,
                            rerank_scores=rerank_scores,
                            legacy_scores=legacy_scores,
                            compact_order=compact_order,
                            compact_gt_rank=compact_gt_rank,
                            shortlist_k=shortlist_k,
                            family=family,
                            normalization="none",
                            weights=weights,
                        )
                        result["compact_score_variant"] = compact_variant
                        result["legacy_score_variant"] = legacy_variant
                        sweep_results.append(result)

                for normalization in NORMALIZATION_MODES:
                    for weights in WEIGHT_TRIPLES:
                        result = _evaluate_setting(
                            compact_scores=compact_scores,
                            rerank_scores=rerank_scores,
                            legacy_scores=legacy_scores,
                            compact_order=compact_order,
                            compact_gt_rank=compact_gt_rank,
                            shortlist_k=shortlist_k,
                            family="normalized_weighted",
                            normalization=normalization,
                            weights=weights,
                        )
                        result["compact_score_variant"] = compact_variant
                        result["legacy_score_variant"] = legacy_variant
                        sweep_results.append(result)

                if include_adaptive:
                    row_idx = np.arange(compact_scores.shape[0])[:, None]
                    shortlist = compact_order[:, :shortlist_k]
                    compact_sl = _normalize_shortlist_scores(
                        compact_scores[row_idx, shortlist], "none"
                    )
                    rerank_sl = _normalize_shortlist_scores(
                        rerank_scores[row_idx, shortlist], "none"
                    )
                    legacy_sl = _normalize_shortlist_scores(
                        legacy_scores[row_idx, shortlist], "none"
                    )
                    use_kappa = compact_kappas if shared_has_kappa and compact_kappas is not None else None
                    adaptive_model = _fit_adaptive_linear(
                        compact_sl=compact_sl,
                        rerank_sl=rerank_sl,
                        legacy_sl=legacy_sl,
                        shortlist=shortlist,
                        compact_kappas=use_kappa,
                    )
                    result = _evaluate_setting(
                        compact_scores=compact_scores,
                        rerank_scores=rerank_scores,
                        legacy_scores=legacy_scores,
                        compact_order=compact_order,
                        compact_gt_rank=compact_gt_rank,
                        shortlist_k=shortlist_k,
                        family="adaptive_linear",
                        normalization="none",
                        weights=None,
                        compact_kappas=use_kappa,
                        adaptive_model=adaptive_model,
                    )
                    result["compact_score_variant"] = compact_variant
                    result["legacy_score_variant"] = legacy_variant
                    result["adaptive_model"] = {
                        "weights": [float(x) for x in adaptive_model["weights"]],
                        "use_compact_kappa": bool(adaptive_model["use_compact_kappa"]),
                    }
                    sweep_results.append(result)

    return baselines, sweep_results


def _apply_frozen_setting(
    split_arrays: dict[str, np.ndarray],
    setting: dict[str, Any],
    shared_has_kappa: bool,
) -> dict[str, Any]:
    scores = _build_split_scores(split_arrays)
    compact_variant = setting["compact_score_variant"]
    legacy_variant = setting["legacy_score_variant"]
    compact_scores = scores["compact_raw"] if compact_variant == "raw_cosine" else scores["compact_csls"]
    legacy_scores = scores["legacy_raw"] if legacy_variant == "raw_cosine" else scores["legacy_csls"]
    compact_order, compact_gt_rank = _gt_rank_from_scores(compact_scores)
    adaptive_model = setting.get("adaptive_model")
    compact_kappas = split_arrays.get("compact_kappas") if shared_has_kappa else None
    result = _evaluate_setting(
        compact_scores=compact_scores,
        rerank_scores=scores["rerank"],
        legacy_scores=legacy_scores,
        compact_order=compact_order,
        compact_gt_rank=compact_gt_rank,
        shortlist_k=int(setting["shortlist_k"]),
        family=str(setting["fusion_family"]),
        normalization=str(setting["normalization"]),
        weights=None if setting["fusion_family"] == "adaptive_linear" else (
            float(setting["alpha"]),
            float(setting["beta"]),
            float(setting["gamma"]),
        ),
        compact_kappas=compact_kappas,
        adaptive_model=adaptive_model,
    )
    result["compact_score_variant"] = compact_variant
    result["legacy_score_variant"] = legacy_variant
    if adaptive_model is not None:
        result["adaptive_model"] = adaptive_model
    return result


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _nearby_experiment_dirs(path: Path) -> list[str]:
    parent = path.parent
    if not parent.exists():
        return []
    return sorted(
        p.name for p in parent.iterdir()
        if p.is_dir()
    )[:20]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep tri-expert retrieval fusion")
    parser.add_argument("tri_results_dir", type=str, help="V32/V33 results dir, e.g. experimental_results/V32_pca_rerank_2048/subj01")
    parser.add_argument("legacy_results_dir", type=str, help="Legacy N1v28a results dir, e.g. experimental_results/N1v28a_dual_head/subj01")
    parser.add_argument("--include-adaptive", action="store_true", help="Include optional adaptive linear fusion fitted on VAL only")
    args = parser.parse_args()

    tri_results_dir = Path(args.tri_results_dir)
    legacy_results_dir = Path(args.legacy_results_dir)
    if not tri_results_dir.exists():
        nearby = _nearby_experiment_dirs(tri_results_dir)
        hint = f" Nearby experiment dirs: {nearby}" if nearby else ""
        raise FileNotFoundError(f"Tri-expert results directory not found: {tri_results_dir}.{hint}")
    if not legacy_results_dir.exists():
        nearby = _nearby_experiment_dirs(legacy_results_dir)
        hint = f" Nearby experiment dirs: {nearby}" if nearby else ""
        raise FileNotFoundError(f"Legacy results directory not found: {legacy_results_dir}.{hint}")
    tri_metrics_dir = tri_results_dir / "metrics"
    legacy_metrics_dir = legacy_results_dir / "metrics"
    if not tri_metrics_dir.exists():
        nearby = sorted(p.name for p in tri_results_dir.iterdir()) if tri_results_dir.exists() else []
        raise FileNotFoundError(
            f"Tri-expert metrics directory not found: {tri_metrics_dir}. "
            f"Contents of {tri_results_dir}: {nearby}"
        )
    if not legacy_metrics_dir.exists():
        nearby = sorted(p.name for p in legacy_results_dir.iterdir()) if legacy_results_dir.exists() else []
        raise FileNotFoundError(
            f"Legacy metrics directory not found: {legacy_metrics_dir}. "
            f"Contents of {legacy_results_dir}: {nearby}"
        )

    tri_val = _load_tri_split(tri_metrics_dir, "val")
    tri_shared = _load_tri_split(tri_metrics_dir, "shared1000")
    legacy_val = _load_legacy_split(
        legacy_metrics_dir,
        legacy_results_dir,
        "val",
        reference_split=tri_val,
    )
    legacy_shared = _load_legacy_split(
        legacy_metrics_dir,
        legacy_results_dir,
        "shared1000",
        reference_split=tri_shared,
    )

    _assert_split_disjointness(tri_val["nsd_ids"], tri_shared["nsd_ids"], "tri_results_dir")
    _assert_split_disjointness(legacy_val["nsd_ids"], legacy_shared["nsd_ids"], "legacy_results_dir")

    val_arrays = _align_common_ids(tri_val, legacy_val, "val")
    shared_arrays = _align_common_ids(tri_shared, legacy_shared, "shared1000")
    _assert_split_disjointness(val_arrays["nsd_ids"], shared_arrays["nsd_ids"], "aligned_tri_legacy")

    diagnostics_dir = tri_results_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    shared_has_kappa = (
        val_arrays.get("compact_kappas") is not None
        and shared_arrays.get("compact_kappas") is not None
    )
    if args.include_adaptive and not shared_has_kappa:
        logger.info("Adaptive fusion will ignore compact_kappa because it is not available on both VAL and SHARED1000")

    val_baselines, val_results = _evaluate_split(
        val_arrays,
        include_adaptive=args.include_adaptive,
        shared_has_kappa=shared_has_kappa,
    )
    best_val = _best_result(val_results)
    shared_best = _apply_frozen_setting(shared_arrays, best_val, shared_has_kappa=shared_has_kappa)
    shared_baselines, _ = _evaluate_split(
        shared_arrays,
        include_adaptive=False,
        shared_has_kappa=shared_has_kappa,
    )

    val_two_expert = _load_existing_two_expert_fusion(tri_metrics_dir, "val")
    shared_two_expert = _load_existing_two_expert_fusion(tri_metrics_dir, "shared1000")

    best_compact_csls = val_baselines["compact_csls"]["R@1"]
    best_legacy_csls = val_baselines["legacy_csls"]["R@1"]
    best_two_expert_val = (
        float(val_two_expert.get("fused", {}).get("fused_r@1", 0.0))
        if val_two_expert is not None else 0.0
    )
    best_two_expert_shared = (
        float(shared_two_expert.get("fused", {}).get("fused_r@1", 0.0))
        if shared_two_expert is not None else 0.0
    )

    report = {
        "search_space": {
            "shortlist_k": SHORTLIST_KS,
            "fusion_families": ["weighted", "normalized_weighted", "rrf", "rank_average"]
            + (["adaptive_linear"] if args.include_adaptive else []),
            "normalization_modes": NORMALIZATION_MODES,
            "weight_step": WEIGHT_STEP,
            "weight_triples_count": len(WEIGHT_TRIPLES),
            "compact_score_variants": COMPACT_SCORE_VARIANTS,
            "legacy_score_variants": LEGACY_SCORE_VARIANTS,
            "rerank_score_variant": "cosine",
        },
        "alignment": {
            "val_common_nsd_ids": int(len(val_arrays["nsd_ids"])),
            "shared1000_common_nsd_ids": int(len(shared_arrays["nsd_ids"])),
            "adaptive_compact_kappa_used": bool(shared_has_kappa),
        },
        "val": {
            "baselines": val_baselines,
            "existing_two_expert_fusion": val_two_expert,
            "best_setting": best_val,
            "all_results": val_results,
        },
        "shared1000": {
            "baselines": shared_baselines,
            "existing_two_expert_fusion": shared_two_expert,
            "frozen_best_setting": shared_best,
        },
    }

    json_path = diagnostics_dir / "tri_fusion_sweep.json"
    csv_path = diagnostics_dir / "tri_fusion_sweep_val.csv"
    _save_json(json_path, report)
    _write_csv(csv_path, [
        {k: v for k, v in row.items() if k != "adaptive_model"}
        for row in val_results
    ])

    val_metrics_payload = {
        "selected_on": "val",
        "baselines": val_baselines,
        "existing_two_expert_fusion": val_two_expert,
        "tri_fused_best": best_val,
        "gain_over_compact_csls": float(best_val["R@1"] - best_compact_csls),
        "gain_over_legacy_csls": float(best_val["R@1"] - best_legacy_csls),
        "gain_over_two_expert_fusion": float(best_val["R@1"] - best_two_expert_val),
    }
    shared_metrics_payload = {
        "selected_on": "val",
        "baselines": shared_baselines,
        "existing_two_expert_fusion": shared_two_expert,
        "tri_fused_frozen": shared_best,
        "gain_over_compact_csls": float(shared_best["R@1"] - shared_baselines["compact_csls"]["R@1"]),
        "gain_over_legacy_csls": float(shared_best["R@1"] - shared_baselines["legacy_csls"]["R@1"]),
        "gain_over_two_expert_fusion": float(shared_best["R@1"] - best_two_expert_shared),
    }
    _save_json(tri_metrics_dir / "val_tri_fused_metrics.json", val_metrics_payload)
    _save_json(tri_metrics_dir / "shared1000_tri_fused_metrics.json", shared_metrics_payload)

    print("=" * 76)
    print("TRI-EXPERT FUSION SWEEP SUMMARY")
    print("=" * 76)
    print(f"Tri results dir:     {tri_results_dir}")
    print(f"Legacy results dir:  {legacy_results_dir}")
    print("")
    print("VAL baselines")
    print(f"  Compact raw R@1:          {val_baselines['compact_raw']['R@1']:.1%}")
    print(f"  Compact CSLS R@1:         {val_baselines['compact_csls']['R@1']:.1%}")
    print(f"  Rerank-only R@1:          {val_baselines['rerank_only']['R@1']:.1%}")
    print(f"  Legacy raw R@1:           {val_baselines['legacy_raw']['R@1']:.1%}")
    print(f"  Legacy CSLS R@1:          {val_baselines['legacy_csls']['R@1']:.1%}")
    if val_two_expert is not None:
        print(f"  Current 2-expert R@1:     {best_two_expert_val:.1%}")
    print("")
    print("Best VAL tri-expert setting")
    print(f"  compact score:            {best_val['compact_score_variant']}")
    print(f"  legacy score:             {best_val['legacy_score_variant']}")
    print(f"  family:                   {best_val['fusion_family']}")
    print(f"  normalization:            {best_val['normalization']}")
    print(f"  shortlist_k:              {best_val['shortlist_k']}")
    if best_val["alpha"] is not None:
        print(f"  alpha / beta / gamma:     {best_val['alpha']} / {best_val['beta']} / {best_val['gamma']}")
    print(f"  VAL R@1 / R@5 / R@10:     {best_val['R@1']:.1%} / {best_val['R@5']:.1%} / {best_val['R@10']:.1%}")
    print(f"  VAL MedR / MRR:           {best_val['median_rank']:.1f} / {best_val['MRR']:.4f}")
    print("")
    print("Frozen SHARED1000 result")
    print(f"  SHARED R@1 / R@5 / R@10:  {shared_best['R@1']:.1%} / {shared_best['R@5']:.1%} / {shared_best['R@10']:.1%}")
    print(f"  SHARED MedR / MRR:        {shared_best['median_rank']:.1f} / {shared_best['MRR']:.4f}")
    print("")
    print("Gains")
    print(f"  Over compact CSLS (VAL):  {best_val['R@1'] - best_compact_csls:+.1%}")
    print(f"  Over compact CSLS (S1000):{shared_best['R@1'] - shared_baselines['compact_csls']['R@1']:+.1%}")
    print(f"  Over legacy CSLS (VAL):   {best_val['R@1'] - best_legacy_csls:+.1%}")
    print(f"  Over legacy CSLS (S1000): {shared_best['R@1'] - shared_baselines['legacy_csls']['R@1']:+.1%}")
    if val_two_expert is not None:
        print(f"  Over 2-expert (VAL):      {best_val['R@1'] - best_two_expert_val:+.1%}")
    if shared_two_expert is not None:
        print(f"  Over 2-expert (S1000):    {shared_best['R@1'] - best_two_expert_shared:+.1%}")
    print(f"  Over rerank-only (S1000): {shared_best['R@1'] - shared_baselines['rerank_only']['R@1']:+.1%}")
    print("")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
