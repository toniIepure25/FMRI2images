#!/usr/bin/env python3
"""Utilities for the frozen final best retrieval system."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sweep_tri_fusion_retrieval import (
    _align_common_ids,
    _assert_split_disjointness,
    _build_split_scores,
    _compute_baselines,
    _evaluate_setting,
    _gt_rank_from_scores,
    _load_existing_two_expert_fusion,
    _load_legacy_split,
    _load_tri_split,
    _normalize_shortlist_scores,
    _rank_within_shortlist,
)

logger = logging.getLogger(__name__)

DEFAULT_TRI_RESULTS_DIR = Path("experimental_results/V35_legacy_teacher_distill/subj01")
DEFAULT_LEGACY_RESULTS_DIR = Path("experimental_results/N1v28a_dual_head/subj01")
DEFAULT_OUTPUT_DIR = Path("final_outputs/best_system")

EXPECTED_FROZEN_TRI_CONFIG: dict[str, Any] = {
    "compact_score": "csls",
    "legacy_score": "csls",
    "family": "normalized_weighted",
    "normalization": "zscore",
    "shortlist_k": 150,
    "alpha": 0.3,
    "beta": 0.0,
    "gamma": 0.7,
}
EXPECTED_SHARED1000_R1 = 0.772


@dataclass(frozen=True)
class FrozenBestSystem:
    tri_results_dir: Path
    legacy_results_dir: Path
    tri_metrics_dir: Path
    legacy_metrics_dir: Path
    diagnostics_path: Path
    frozen_config: dict[str, Any]
    frozen_shared1000_r1: float | None


@dataclass
class SplitAnalysis:
    split: str
    nsd_ids: np.ndarray
    aligned: dict[str, np.ndarray]
    scores: dict[str, np.ndarray]
    baselines: dict[str, Any]
    fixed_two_expert: dict[str, Any] | None
    fixed_tri_expert: dict[str, Any]
    per_query: list[dict[str, Any]]


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _require_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")


def _extract_best_setting(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        payload.get("best_val"),
        payload.get("best_val_setting"),
        payload.get("selected_val_setting"),
    ]
    summary = payload.get("summary")
    if isinstance(summary, dict):
        candidates.extend([
            summary.get("best_val"),
            summary.get("best_val_setting"),
        ])
    val_block = payload.get("val")
    if isinstance(val_block, dict):
        candidates.extend([
            val_block.get("best_setting"),
            val_block.get("best_val"),
        ])
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    raise KeyError("Could not find a best VAL tri-fusion setting in tri_fusion_sweep.json")


def _normalize_frozen_setting(setting: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "compact_score": setting.get("compact_score") or setting.get("compact_score_variant"),
        "legacy_score": setting.get("legacy_score") or setting.get("legacy_score_variant"),
        "family": setting.get("family") or setting.get("fusion_family"),
        "normalization": setting.get("normalization", "none"),
        "shortlist_k": int(setting.get("shortlist_k")),
        "alpha": float(setting.get("alpha")),
        "beta": float(setting.get("beta")),
        "gamma": float(setting.get("gamma")),
    }
    missing = [k for k, v in normalized.items() if v is None]
    if missing:
        raise KeyError(f"Frozen tri-fusion setting is missing keys: {missing}")
    return normalized


def _extract_shared_r1(payload: dict[str, Any]) -> float | None:
    candidates = []
    shared_block = payload.get("shared1000")
    if isinstance(shared_block, dict):
        frozen = shared_block.get("frozen_best_setting")
        if isinstance(frozen, dict):
            candidates.append(frozen.get("R@1"))
    candidates.extend([
        payload.get("shared1000_r@1"),
        payload.get("shared_r@1"),
    ])
    for value in candidates:
        if value is None:
            continue
        return float(value)
    return None


def load_frozen_best_system(
    tri_results_dir: Path = DEFAULT_TRI_RESULTS_DIR,
    legacy_results_dir: Path = DEFAULT_LEGACY_RESULTS_DIR,
    *,
    strict: bool = True,
) -> FrozenBestSystem:
    tri_results_dir = Path(tri_results_dir)
    legacy_results_dir = Path(legacy_results_dir)
    tri_metrics_dir = tri_results_dir / "metrics"
    legacy_metrics_dir = legacy_results_dir / "metrics"
    diagnostics_path = tri_results_dir / "diagnostics" / "tri_fusion_sweep.json"

    _require_dir(tri_results_dir, "tri/main results dir")
    _require_dir(legacy_results_dir, "legacy results dir")
    _require_dir(tri_metrics_dir, "tri/main metrics dir")
    _require_dir(legacy_metrics_dir, "legacy metrics dir")
    if not diagnostics_path.exists():
        raise FileNotFoundError(f"Frozen tri_fusion_sweep.json not found: {diagnostics_path}")

    payload = _load_json(diagnostics_path)
    frozen_config = _normalize_frozen_setting(_extract_best_setting(payload))
    shared_r1 = _extract_shared_r1(payload)

    if strict and frozen_config != EXPECTED_FROZEN_TRI_CONFIG:
        raise RuntimeError(
            "Saved frozen tri-fusion setting does not match the approved production recipe.\n"
            f"Saved:    {frozen_config}\n"
            f"Expected: {EXPECTED_FROZEN_TRI_CONFIG}"
        )
    if strict and shared_r1 is not None and abs(shared_r1 - EXPECTED_SHARED1000_R1) > 1e-6:
        raise RuntimeError(
            "Saved frozen SHARED1000 R@1 does not match the approved production result.\n"
            f"Saved:    {shared_r1:.4f}\n"
            f"Expected: {EXPECTED_SHARED1000_R1:.4f}"
        )

    return FrozenBestSystem(
        tri_results_dir=tri_results_dir,
        legacy_results_dir=legacy_results_dir,
        tri_metrics_dir=tri_metrics_dir,
        legacy_metrics_dir=legacy_metrics_dir,
        diagnostics_path=diagnostics_path,
        frozen_config=frozen_config,
        frozen_shared1000_r1=shared_r1,
    )


def _compute_fixed_tri_per_query(
    aligned: dict[str, np.ndarray],
    frozen_config: dict[str, Any],
) -> list[dict[str, Any]]:
    scores = _build_split_scores(aligned)

    compact_scores = scores["compact_csls"] if frozen_config["compact_score"] == "csls" else scores["compact_raw"]
    legacy_scores = scores["legacy_csls"] if frozen_config["legacy_score"] == "csls" else scores["legacy_raw"]
    rerank_scores = scores["rerank"]

    compact_order, compact_gt_rank = _gt_rank_from_scores(compact_scores)
    legacy_order, legacy_gt_rank = _gt_rank_from_scores(legacy_scores)
    rerank_order, rerank_gt_rank = _gt_rank_from_scores(rerank_scores)

    n = compact_scores.shape[0]
    shortlist_k = min(int(frozen_config["shortlist_k"]), n)
    shortlist = compact_order[:, :shortlist_k]
    row_idx = np.arange(n)[:, None]
    compact_sl = compact_scores[row_idx, shortlist]
    rerank_sl = rerank_scores[row_idx, shortlist]
    legacy_sl = legacy_scores[row_idx, shortlist]

    family = str(frozen_config["family"])
    alpha = float(frozen_config["alpha"])
    beta = float(frozen_config["beta"])
    gamma = float(frozen_config["gamma"])
    normalization = str(frozen_config["normalization"])

    if family == "weighted":
        fused_scores = alpha * compact_sl + beta * rerank_sl + gamma * legacy_sl
    elif family == "normalized_weighted":
        fused_scores = (
            alpha * _normalize_shortlist_scores(compact_sl, normalization)
            + beta * _normalize_shortlist_scores(rerank_sl, normalization)
            + gamma * _normalize_shortlist_scores(legacy_sl, normalization)
        )
    elif family == "rrf":
        fused_scores = (
            alpha / (60.0 + _rank_within_shortlist(compact_sl))
            + beta / (60.0 + _rank_within_shortlist(rerank_sl))
            + gamma / (60.0 + _rank_within_shortlist(legacy_sl))
        )
    elif family == "rank_average":
        fused_scores = -(
            alpha * _rank_within_shortlist(compact_sl)
            + beta * _rank_within_shortlist(rerank_sl)
            + gamma * _rank_within_shortlist(legacy_sl)
        )
    else:
        raise ValueError(f"Unsupported frozen tri fusion family: {family}")

    local_order = np.argsort(-fused_scores, axis=1)
    fused_shortlist = shortlist[row_idx, local_order]
    fused_gt_rank = compact_gt_rank.copy()
    hit_rows = np.where(np.any(shortlist == np.arange(n)[:, None], axis=1))[0]
    if hit_rows.size > 0:
        local_gt_rank = np.argmax(
            fused_shortlist[hit_rows] == hit_rows[:, None],
            axis=1,
        ) + 1
        fused_gt_rank[hit_rows] = local_gt_rank.astype(np.int32)

    rows: list[dict[str, Any]] = []
    for i, nsd_id in enumerate(aligned["nsd_ids"].astype(np.int32).tolist()):
        fused_topk_idx = fused_shortlist[i, : min(5, fused_shortlist.shape[1])].astype(np.int32)
        fused_topk_nsd = aligned["nsd_ids"][fused_topk_idx].astype(np.int32).tolist()
        rows.append(
            {
                "row_index": i,
                "nsd_id": int(nsd_id),
                "compact_top1_index": int(compact_order[i, 0]),
                "compact_top1_nsd_id": int(aligned["nsd_ids"][compact_order[i, 0]]),
                "legacy_top1_index": int(legacy_order[i, 0]),
                "legacy_top1_nsd_id": int(aligned["nsd_ids"][legacy_order[i, 0]]),
                "rerank_top1_index": int(rerank_order[i, 0]),
                "rerank_top1_nsd_id": int(aligned["nsd_ids"][rerank_order[i, 0]]),
                "fused_top1_index": int(fused_shortlist[i, 0]),
                "fused_top1_nsd_id": int(aligned["nsd_ids"][fused_shortlist[i, 0]]),
                "fused_top5_indices": [int(x) for x in fused_topk_idx],
                "fused_top5_nsd_ids": [int(x) for x in fused_topk_nsd],
                "compact_gt_rank": int(compact_gt_rank[i]),
                "legacy_gt_rank": int(legacy_gt_rank[i]),
                "rerank_gt_rank": int(rerank_gt_rank[i]),
                "fused_gt_rank": int(fused_gt_rank[i]),
                "fused_fixed_case": bool(fused_gt_rank[i] < min(compact_gt_rank[i], legacy_gt_rank[i])),
                "fused_fixed_to_top1": bool(fused_gt_rank[i] == 1 and min(compact_gt_rank[i], legacy_gt_rank[i]) > 1),
            }
        )
    return rows


def analyze_split(system: FrozenBestSystem, split: str) -> SplitAnalysis:
    tri_split = _load_tri_split(system.tri_metrics_dir, split)
    legacy_split = _load_legacy_split(
        system.legacy_metrics_dir,
        system.legacy_results_dir,
        split,
        reference_split=tri_split,
    )
    aligned = _align_common_ids(tri_split, legacy_split, split)
    scores = _build_split_scores(aligned)
    baselines = _compute_baselines(scores)

    compact_key = "compact_csls" if system.frozen_config["compact_score"] == "csls" else "compact_raw"
    legacy_key = "legacy_csls" if system.frozen_config["legacy_score"] == "csls" else "legacy_raw"
    compact_score_matrix = scores[compact_key]
    legacy_score_matrix = scores[legacy_key]
    rerank_score_matrix = scores["rerank"]
    compact_order, compact_gt_rank = _gt_rank_from_scores(compact_score_matrix)

    fixed_tri = _evaluate_setting(
        compact_scores=compact_score_matrix,
        rerank_scores=rerank_score_matrix,
        legacy_scores=legacy_score_matrix,
        compact_order=compact_order,
        compact_gt_rank=compact_gt_rank,
        shortlist_k=int(system.frozen_config["shortlist_k"]),
        family=str(system.frozen_config["family"]),
        normalization=str(system.frozen_config["normalization"]),
        weights=(
            float(system.frozen_config["alpha"]),
            float(system.frozen_config["beta"]),
            float(system.frozen_config["gamma"]),
        ),
    )

    fixed_two_expert = _load_existing_two_expert_fusion(system.tri_metrics_dir, split)
    per_query = _compute_fixed_tri_per_query(aligned, system.frozen_config)
    return SplitAnalysis(
        split=split,
        nsd_ids=aligned["nsd_ids"],
        aligned=aligned,
        scores=scores,
        baselines=baselines,
        fixed_two_expert=fixed_two_expert,
        fixed_tri_expert=fixed_tri,
        per_query=per_query,
    )


def load_final_best_system_analysis(
    tri_results_dir: Path = DEFAULT_TRI_RESULTS_DIR,
    legacy_results_dir: Path = DEFAULT_LEGACY_RESULTS_DIR,
    *,
    strict: bool = True,
) -> tuple[FrozenBestSystem, SplitAnalysis, SplitAnalysis]:
    system = load_frozen_best_system(tri_results_dir, legacy_results_dir, strict=strict)
    val_analysis = analyze_split(system, "val")
    shared_analysis = analyze_split(system, "shared1000")
    _assert_split_disjointness(val_analysis.nsd_ids, shared_analysis.nsd_ids, "final_best_system")
    return system, val_analysis, shared_analysis


def split_summary_payload(system: FrozenBestSystem, analysis: SplitAnalysis) -> dict[str, Any]:
    compact_raw = analysis.baselines["compact_raw"]
    compact_csls = analysis.baselines["compact_csls"]
    rerank_only = analysis.baselines["rerank_only"]
    legacy_raw = analysis.baselines["legacy_raw"]
    legacy_csls = analysis.baselines["legacy_csls"]
    fixed_tri = analysis.fixed_tri_expert

    two_expert_fused = None
    if analysis.fixed_two_expert is not None:
        two_expert_fused = analysis.fixed_two_expert.get("fused", analysis.fixed_two_expert)

    tri_r1 = float(fixed_tri["R@1"])
    compact_csls_r1 = float(compact_csls["R@1"])
    legacy_csls_r1 = float(legacy_csls["R@1"])
    two_expert_r1 = (
        float(two_expert_fused.get("fused_r@1", two_expert_fused.get("R@1", 0.0)))
        if two_expert_fused is not None else None
    )

    payload: dict[str, Any] = {
        "split": analysis.split,
        "n_queries": int(len(analysis.nsd_ids)),
        "compact_raw": compact_raw,
        "compact_csls": compact_csls,
        "rerank_only": rerank_only,
        "legacy_raw": legacy_raw,
        "legacy_csls": legacy_csls,
        "fixed_tri_expert": fixed_tri,
        "gain_over_compact_csls": tri_r1 - compact_csls_r1,
        "gain_over_legacy_csls": tri_r1 - legacy_csls_r1,
        "gain_over_two_expert_fusion": None if two_expert_r1 is None else tri_r1 - two_expert_r1,
    }
    if analysis.fixed_two_expert is not None:
        payload["fixed_two_expert_fusion"] = analysis.fixed_two_expert
    return payload
