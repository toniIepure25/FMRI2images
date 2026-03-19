#!/usr/bin/env python3
"""Fit a learned tri-expert shortlist fusion gate on VAL only.

This wave keeps all three experts frozen:
    1. compact expert from the current triple-head run
    2. rerank expert from the current rerank head
    3. legacy N1v28a token-space expert

The gate is trained on shortlist-local candidate features from VAL only, then
the single best VAL setting is frozen and applied once to SHARED1000.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from sweep_tri_fusion_retrieval import (
    _align_common_ids,
    _assert_split_disjointness,
    _best_result,
    _build_split_scores,
    _compute_baselines,
    _gt_rank_from_scores,
    _load_existing_two_expert_fusion,
    _load_legacy_split,
    _load_tri_split,
    _metrics_from_gt_rank,
    _nearby_experiment_dirs,
    _rank_within_shortlist,
    _save_json,
    _write_csv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SHORTLIST_KS = [50, 100, 150]
COMPACT_SHORTLIST_VARIANTS = ["raw_cosine", "csls"]
MODEL_FAMILIES = ["linear_logistic", "shallow_mlp"]
DEFAULT_SEED = 42


class LinearLogisticGate(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class ShallowMLPGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_existing_fixed_tri_fusion(metrics_dir: Path, prefix: str) -> dict[str, Any] | None:
    path = metrics_dir / f"{prefix}_tri_fused_metrics.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _extract_fixed_tri_r1(payload: dict[str, Any] | None) -> float | None:
    if payload is None:
        return None
    tri = payload.get("tri_fused_best", payload.get("tri_fused_frozen", payload))
    return float(tri.get("R@1", 0.0))


def _top_margin(local_scores: np.ndarray, top_n: int) -> np.ndarray:
    if local_scores.shape[1] < 2:
        return np.zeros(local_scores.shape[0], dtype=np.float32)
    sorted_scores = np.sort(local_scores, axis=1)[:, ::-1]
    idx = min(max(top_n - 1, 1), sorted_scores.shape[1] - 1)
    return (sorted_scores[:, 0] - sorted_scores[:, idx]).astype(np.float32)


def _repeat_query_feature(values: np.ndarray, shortlist_k: int) -> np.ndarray:
    return np.repeat(values[:, None].astype(np.float32), shortlist_k, axis=1)


def _flatten_feature_map(feature_map: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    feature_names = list(feature_map.keys())
    x = np.stack([feature_map[name].reshape(-1) for name in feature_names], axis=1).astype(np.float32)
    return x, feature_names


def _shortlist_metrics_from_logits(
    logits: np.ndarray,
    shortlist: np.ndarray,
    compact_gt_rank: np.ndarray,
) -> dict[str, float]:
    row_idx = np.arange(shortlist.shape[0])[:, None]
    local_order = np.argsort(-logits, axis=1)
    reranked_shortlist = shortlist[row_idx, local_order]

    gt_rank = compact_gt_rank.copy()
    gt_ids = np.arange(shortlist.shape[0])
    hit_mask = np.any(shortlist == gt_ids[:, None], axis=1)
    if np.any(hit_mask):
        hit_rows = np.where(hit_mask)[0]
        local_gt_rank = (
            np.argmax(reranked_shortlist[hit_rows] == hit_rows[:, None], axis=1) + 1
        ).astype(np.int32)
        gt_rank[hit_rows] = local_gt_rank
    return _metrics_from_gt_rank(gt_rank)


def _serialize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {k: v.detach().cpu().numpy().tolist() for k, v in state_dict.items()}


def _deserialize_state_dict(serialized: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in serialized.items()}


def _build_model(model_family: str, input_dim: int) -> nn.Module:
    if model_family == "linear_logistic":
        return LinearLogisticGate(input_dim)
    if model_family == "shallow_mlp":
        return ShallowMLPGate(input_dim)
    raise ValueError(f"Unknown model_family: {model_family}")


def _feature_stats(x: np.ndarray) -> dict[str, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean, "std": std}


def _apply_feature_stats(x: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((x - stats["mean"]) / stats["std"]).astype(np.float32)


def _train_gate_model(
    x: np.ndarray,
    y: np.ndarray,
    shortlist: np.ndarray,
    compact_gt_rank: np.ndarray,
    model_family: str,
    seed: int,
) -> dict[str, Any]:
    _set_seed(seed)
    input_dim = x.shape[1]
    model = _build_model(model_family, input_dim)

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y.astype(np.float32))
    pos = max(float(y.sum()), 1.0)
    neg = max(float(y.shape[0] - y.sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if model_family == "linear_logistic":
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-2, weight_decay=1e-3)
        max_epochs = 300
        eval_every = 5
        patience = 30
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=2e-4)
        max_epochs = 400
        eval_every = 5
        patience = 40

    best_payload: dict[str, Any] | None = None
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % eval_every == 0 or epoch == max_epochs:
            model.eval()
            with torch.no_grad():
                eval_logits = model(x_t).detach().cpu().numpy().reshape(shortlist.shape)
            metrics = _shortlist_metrics_from_logits(eval_logits, shortlist, compact_gt_rank)
            row: dict[str, Any] = {
                "epoch": int(epoch),
                "train_bce": float(loss.item()),
                **metrics,
            }
            if best_payload is None or _best_result([best_payload["metrics"], row]) == row:
                best_payload = {
                    "epoch": int(epoch),
                    "train_bce": float(loss.item()),
                    "metrics": row,
                    "state_dict": copy.deepcopy(model.state_dict()),
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    assert best_payload is not None
    model.load_state_dict(best_payload["state_dict"])
    model.eval()
    return {
        "model_family": model_family,
        "best_epoch": int(best_payload["epoch"]),
        "best_train_bce": float(best_payload["train_bce"]),
        "best_metrics": best_payload["metrics"],
        "state_dict": _serialize_state_dict(model.state_dict()),
        "pos_weight": float(pos_weight.item()),
    }


def _predict_gate_logits(
    x: np.ndarray,
    model_family: str,
    state_dict: dict[str, Any],
) -> np.ndarray:
    model = _build_model(model_family, x.shape[1])
    model.load_state_dict(_deserialize_state_dict(state_dict))
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x.astype(np.float32))).detach().cpu().numpy()
    return logits.astype(np.float32)


def _build_gate_dataset(
    split_arrays: dict[str, np.ndarray],
    shortlist_k: int,
    compact_shortlist_variant: str,
) -> dict[str, Any]:
    scores = _build_split_scores(split_arrays)
    n = scores["compact_raw"].shape[0]
    compact_shortlist_scores = (
        scores["compact_raw"] if compact_shortlist_variant == "raw_cosine" else scores["compact_csls"]
    )
    compact_order, compact_gt_rank = _gt_rank_from_scores(compact_shortlist_scores)
    shortlist = compact_order[:, :shortlist_k]
    row_idx = np.arange(n)[:, None]

    local_scores = {
        "compact_raw": scores["compact_raw"][row_idx, shortlist].astype(np.float32),
        "compact_csls": scores["compact_csls"][row_idx, shortlist].astype(np.float32),
        "rerank": scores["rerank"][row_idx, shortlist].astype(np.float32),
        "legacy_raw": scores["legacy_raw"][row_idx, shortlist].astype(np.float32),
        "legacy_csls": scores["legacy_csls"][row_idx, shortlist].astype(np.float32),
    }
    local_ranks = {name: _rank_within_shortlist(vals).astype(np.float32) for name, vals in local_scores.items()}
    top1_margins = {name: _top_margin(vals, 2) for name, vals in local_scores.items()}
    top5_margins = {name: _top_margin(vals, 5) for name, vals in local_scores.items()}

    compact_top = np.argmax(local_scores["compact_csls"], axis=1)
    rerank_top = np.argmax(local_scores["rerank"], axis=1)
    legacy_top = np.argmax(local_scores["legacy_csls"], axis=1)
    agree_compact_rerank = (compact_top == rerank_top).astype(np.float32)
    agree_compact_legacy = (compact_top == legacy_top).astype(np.float32)
    agree_rerank_legacy = (rerank_top == legacy_top).astype(np.float32)

    compact_kappas = split_arrays.get("compact_kappas")
    compact_kappa_q = (
        compact_kappas.astype(np.float32).reshape(n, -1).mean(axis=1)
        if compact_kappas is not None else np.zeros(n, dtype=np.float32)
    )

    shortlist_pos = np.tile(np.arange(shortlist_k, dtype=np.float32), (n, 1))
    denom = float(max(shortlist_k - 1, 1))
    shortlist_pos_norm = shortlist_pos / denom

    feature_map: dict[str, np.ndarray] = {
        "compact_raw_score": local_scores["compact_raw"],
        "compact_csls_score": local_scores["compact_csls"],
        "rerank_score": local_scores["rerank"],
        "legacy_raw_score": local_scores["legacy_raw"],
        "legacy_csls_score": local_scores["legacy_csls"],
        "compact_raw_rank": local_ranks["compact_raw"],
        "compact_csls_rank": local_ranks["compact_csls"],
        "rerank_rank": local_ranks["rerank"],
        "legacy_raw_rank": local_ranks["legacy_raw"],
        "legacy_csls_rank": local_ranks["legacy_csls"],
        "compact_raw_minus_rerank": local_scores["compact_raw"] - local_scores["rerank"],
        "compact_csls_minus_rerank": local_scores["compact_csls"] - local_scores["rerank"],
        "compact_raw_minus_legacy_raw": local_scores["compact_raw"] - local_scores["legacy_raw"],
        "compact_csls_minus_legacy_csls": local_scores["compact_csls"] - local_scores["legacy_csls"],
        "rerank_minus_legacy_raw": local_scores["rerank"] - local_scores["legacy_raw"],
        "rerank_minus_legacy_csls": local_scores["rerank"] - local_scores["legacy_csls"],
        "compact_shortlist_position_norm": shortlist_pos_norm,
        "compact_raw_top1_margin": _repeat_query_feature(top1_margins["compact_raw"], shortlist_k),
        "compact_raw_top5_margin": _repeat_query_feature(top5_margins["compact_raw"], shortlist_k),
        "compact_csls_top1_margin": _repeat_query_feature(top1_margins["compact_csls"], shortlist_k),
        "compact_csls_top5_margin": _repeat_query_feature(top5_margins["compact_csls"], shortlist_k),
        "rerank_top1_margin": _repeat_query_feature(top1_margins["rerank"], shortlist_k),
        "rerank_top5_margin": _repeat_query_feature(top5_margins["rerank"], shortlist_k),
        "legacy_raw_top1_margin": _repeat_query_feature(top1_margins["legacy_raw"], shortlist_k),
        "legacy_raw_top5_margin": _repeat_query_feature(top5_margins["legacy_raw"], shortlist_k),
        "legacy_csls_top1_margin": _repeat_query_feature(top1_margins["legacy_csls"], shortlist_k),
        "legacy_csls_top5_margin": _repeat_query_feature(top5_margins["legacy_csls"], shortlist_k),
        "agree_compact_rerank_top1": _repeat_query_feature(agree_compact_rerank, shortlist_k),
        "agree_compact_legacy_top1": _repeat_query_feature(agree_compact_legacy, shortlist_k),
        "agree_rerank_legacy_top1": _repeat_query_feature(agree_rerank_legacy, shortlist_k),
        "is_compact_top1": (local_ranks["compact_csls"] == 1).astype(np.float32),
        "is_rerank_top1": (local_ranks["rerank"] == 1).astype(np.float32),
        "is_legacy_top1": (local_ranks["legacy_csls"] == 1).astype(np.float32),
        "compact_kappa_query": _repeat_query_feature(compact_kappa_q, shortlist_k),
    }
    x, feature_names = _flatten_feature_map(feature_map)
    y = (shortlist == np.arange(n)[:, None]).astype(np.float32).reshape(-1)

    return {
        "scores": scores,
        "shortlist": shortlist,
        "compact_gt_rank": compact_gt_rank.astype(np.int32),
        "shortlist_recall": float(np.mean(compact_gt_rank <= shortlist_k)),
        "x": x,
        "y": y,
        "feature_names": feature_names,
    }


def _fit_setting(
    split_arrays: dict[str, np.ndarray],
    shortlist_k: int,
    compact_shortlist_variant: str,
    model_family: str,
    seed: int,
) -> dict[str, Any]:
    dataset = _build_gate_dataset(split_arrays, shortlist_k, compact_shortlist_variant)
    stats = _feature_stats(dataset["x"])
    x_scaled = _apply_feature_stats(dataset["x"], stats)
    train_result = _train_gate_model(
        x=x_scaled,
        y=dataset["y"],
        shortlist=dataset["shortlist"],
        compact_gt_rank=dataset["compact_gt_rank"],
        model_family=model_family,
        seed=seed,
    )
    best_metrics = dict(train_result["best_metrics"])
    result: dict[str, Any] = {
        "model_family": model_family,
        "shortlist_k": int(shortlist_k),
        "compact_shortlist_variant": compact_shortlist_variant,
        "shortlist_recall": float(dataset["shortlist_recall"]),
        "feature_dim": int(x_scaled.shape[1]),
        "feature_names": dataset["feature_names"],
        "scaler_mean": stats["mean"].reshape(-1).astype(float).tolist(),
        "scaler_std": stats["std"].reshape(-1).astype(float).tolist(),
        "model_state_dict": train_result["state_dict"],
        "best_epoch": int(train_result["best_epoch"]),
        "best_train_bce": float(train_result["best_train_bce"]),
        "train_pos_weight": float(train_result["pos_weight"]),
        **best_metrics,
    }
    return result


def _apply_frozen_gate(
    split_arrays: dict[str, np.ndarray],
    frozen_setting: dict[str, Any],
) -> dict[str, Any]:
    dataset = _build_gate_dataset(
        split_arrays,
        shortlist_k=int(frozen_setting["shortlist_k"]),
        compact_shortlist_variant=str(frozen_setting["compact_shortlist_variant"]),
    )
    stats = {
        "mean": np.asarray(frozen_setting["scaler_mean"], dtype=np.float32)[None, :],
        "std": np.asarray(frozen_setting["scaler_std"], dtype=np.float32)[None, :],
    }
    x_scaled = _apply_feature_stats(dataset["x"], stats)
    logits = _predict_gate_logits(
        x=x_scaled,
        model_family=str(frozen_setting["model_family"]),
        state_dict=frozen_setting["model_state_dict"],
    ).reshape(dataset["shortlist"].shape)
    metrics = _shortlist_metrics_from_logits(
        logits,
        dataset["shortlist"],
        dataset["compact_gt_rank"],
    )
    result = {
        "model_family": str(frozen_setting["model_family"]),
        "shortlist_k": int(frozen_setting["shortlist_k"]),
        "compact_shortlist_variant": str(frozen_setting["compact_shortlist_variant"]),
        "shortlist_recall": float(dataset["shortlist_recall"]),
        "feature_dim": int(dataset["x"].shape[1]),
        **metrics,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit learned tri-fusion gate on VAL only")
    parser.add_argument(
        "tri_results_dir",
        type=str,
        help="Current triple-head results dir, e.g. experimental_results/V33b_shortlist_teacher_distill_preinit/subj01",
    )
    parser.add_argument(
        "legacy_results_dir",
        type=str,
        help="Legacy N1v28a results dir, e.g. experimental_results/N1v28a_dual_head/subj01",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    tri_results_dir = Path(args.tri_results_dir)
    legacy_results_dir = Path(args.legacy_results_dir)
    if not tri_results_dir.exists():
        nearby = _nearby_experiment_dirs(tri_results_dir)
        hint = f" Nearby experiment dirs: {nearby}" if nearby else ""
        raise FileNotFoundError(f"Tri results directory not found: {tri_results_dir}.{hint}")
    if not legacy_results_dir.exists():
        nearby = _nearby_experiment_dirs(legacy_results_dir)
        hint = f" Nearby experiment dirs: {nearby}" if nearby else ""
        raise FileNotFoundError(f"Legacy results directory not found: {legacy_results_dir}.{hint}")

    tri_metrics_dir = tri_results_dir / "metrics"
    legacy_metrics_dir = legacy_results_dir / "metrics"
    if not tri_metrics_dir.exists():
        raise FileNotFoundError(f"Tri metrics directory not found: {tri_metrics_dir}")
    if not legacy_metrics_dir.exists():
        raise FileNotFoundError(f"Legacy metrics directory not found: {legacy_metrics_dir}")

    tri_val = _load_tri_split(tri_metrics_dir, "val")
    tri_shared = _load_tri_split(tri_metrics_dir, "shared1000")
    legacy_val = _load_legacy_split(legacy_metrics_dir, legacy_results_dir, "val", reference_split=tri_val)
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
    if len(val_arrays["nsd_ids"]) != len(tri_val["nsd_ids"]) or len(val_arrays["nsd_ids"]) != len(legacy_val["nsd_ids"]):
        raise ValueError(
            "VAL nsd_id alignment is not exact after recovery; refusing to fit tri-fusion gate "
            f"(tri={len(tri_val['nsd_ids'])}, legacy={len(legacy_val['nsd_ids'])}, common={len(val_arrays['nsd_ids'])})"
        )
    if len(shared_arrays["nsd_ids"]) != len(tri_shared["nsd_ids"]) or len(shared_arrays["nsd_ids"]) != len(legacy_shared["nsd_ids"]):
        raise ValueError(
            "SHARED1000 nsd_id alignment is not exact after recovery; refusing to fit tri-fusion gate "
            f"(tri={len(tri_shared['nsd_ids'])}, legacy={len(legacy_shared['nsd_ids'])}, common={len(shared_arrays['nsd_ids'])})"
        )

    diagnostics_dir = tri_results_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    val_scores = _build_split_scores(val_arrays)
    shared_scores = _build_split_scores(shared_arrays)
    val_baselines = _compute_baselines(val_scores)
    shared_baselines = _compute_baselines(shared_scores)

    val_two_expert = _load_existing_two_expert_fusion(tri_metrics_dir, "val")
    shared_two_expert = _load_existing_two_expert_fusion(tri_metrics_dir, "shared1000")
    val_fixed_tri = _load_existing_fixed_tri_fusion(tri_metrics_dir, "val")
    shared_fixed_tri = _load_existing_fixed_tri_fusion(tri_metrics_dir, "shared1000")

    rows: list[dict[str, Any]] = []
    for shortlist_k in SHORTLIST_KS:
        for compact_shortlist_variant in COMPACT_SHORTLIST_VARIANTS:
            for model_family in MODEL_FAMILIES:
                logger.info(
                    "Fitting gate: model=%s shortlist_k=%d compact_shortlist_variant=%s",
                    model_family,
                    shortlist_k,
                    compact_shortlist_variant,
                )
                row = _fit_setting(
                    split_arrays=val_arrays,
                    shortlist_k=shortlist_k,
                    compact_shortlist_variant=compact_shortlist_variant,
                    model_family=model_family,
                    seed=args.seed,
                )
                rows.append(row)

    best_val = _best_result(rows)
    shared_best = _apply_frozen_gate(shared_arrays, best_val)

    best_two_expert_val = (
        float(val_two_expert.get("fused", {}).get("fused_r@1", 0.0))
        if val_two_expert is not None else 0.0
    )
    best_two_expert_shared = (
        float(shared_two_expert.get("fused", {}).get("fused_r@1", 0.0))
        if shared_two_expert is not None else 0.0
    )
    best_fixed_tri_val = _extract_fixed_tri_r1(val_fixed_tri)
    best_fixed_tri_shared = _extract_fixed_tri_r1(shared_fixed_tri)

    report = {
        "search_space": {
            "shortlist_k": SHORTLIST_KS,
            "compact_shortlist_variants": COMPACT_SHORTLIST_VARIANTS,
            "model_families": MODEL_FAMILIES,
            "seed": int(args.seed),
        },
        "alignment": {
            "val_common_nsd_ids": int(len(val_arrays["nsd_ids"])),
            "shared1000_common_nsd_ids": int(len(shared_arrays["nsd_ids"])),
            "strict_val_alignment": bool(len(val_arrays["nsd_ids"]) == len(tri_val["nsd_ids"]) == len(legacy_val["nsd_ids"])),
            "strict_shared_alignment": bool(len(shared_arrays["nsd_ids"]) == len(tri_shared["nsd_ids"]) == len(legacy_shared["nsd_ids"])),
        },
        "val": {
            "baselines": val_baselines,
            "existing_two_expert_fusion": val_two_expert,
            "existing_fixed_tri_fusion": val_fixed_tri,
            "all_results": rows,
            "best_setting": best_val,
        },
        "shared1000": {
            "baselines": shared_baselines,
            "existing_two_expert_fusion": shared_two_expert,
            "existing_fixed_tri_fusion": shared_fixed_tri,
            "frozen_best_setting": shared_best,
        },
    }

    summary_path = diagnostics_dir / "tri_fusion_gate_summary.json"
    csv_path = diagnostics_dir / "tri_fusion_gate_val.csv"
    _save_json(summary_path, report)
    _write_csv(
        csv_path,
        [
            {k: v for k, v in row.items() if k not in {"model_state_dict", "feature_names", "scaler_mean", "scaler_std"}}
            for row in rows
        ],
    )

    val_metrics_payload = {
        "selected_on": "val",
        "baselines": val_baselines,
        "existing_two_expert_fusion": val_two_expert,
        "existing_fixed_tri_fusion": val_fixed_tri,
        "tri_gated_best": best_val,
        "gain_over_compact_csls": float(best_val["R@1"] - val_baselines["compact_csls"]["R@1"]),
        "gain_over_legacy_csls": float(best_val["R@1"] - val_baselines["legacy_csls"]["R@1"]),
        "gain_over_two_expert_fusion": float(best_val["R@1"] - best_two_expert_val),
        "gain_over_fixed_tri_fusion": (
            None if best_fixed_tri_val is None else float(best_val["R@1"] - best_fixed_tri_val)
        ),
    }
    shared_metrics_payload = {
        "selected_on": "val",
        "baselines": shared_baselines,
        "existing_two_expert_fusion": shared_two_expert,
        "existing_fixed_tri_fusion": shared_fixed_tri,
        "tri_gated_frozen": shared_best,
        "gain_over_compact_csls": float(shared_best["R@1"] - shared_baselines["compact_csls"]["R@1"]),
        "gain_over_legacy_csls": float(shared_best["R@1"] - shared_baselines["legacy_csls"]["R@1"]),
        "gain_over_two_expert_fusion": float(shared_best["R@1"] - best_two_expert_shared),
        "gain_over_fixed_tri_fusion": (
            None if best_fixed_tri_shared is None else float(shared_best["R@1"] - best_fixed_tri_shared)
        ),
    }
    _save_json(tri_metrics_dir / "val_tri_gated_metrics.json", val_metrics_payload)
    _save_json(tri_metrics_dir / "shared1000_tri_gated_metrics.json", shared_metrics_payload)

    print("=" * 78)
    print("LEARNED TRI-FUSION GATE SUMMARY")
    print("=" * 78)
    print(f"Tri results dir:    {tri_results_dir}")
    print(f"Legacy results dir: {legacy_results_dir}")
    print("")
    print("Best VAL gate")
    print(f"  model_family:             {best_val['model_family']}")
    print(f"  shortlist_k:              {best_val['shortlist_k']}")
    print(f"  compact shortlist score:  {best_val['compact_shortlist_variant']}")
    print(f"  best_epoch:               {best_val['best_epoch']}")
    print(f"  feature_dim:              {best_val['feature_dim']}")
    print(f"  VAL R@1 / R@5 / R@10:     {best_val['R@1']:.1%} / {best_val['R@5']:.1%} / {best_val['R@10']:.1%}")
    print(f"  VAL MedR / MRR:           {best_val['median_rank']:.1f} / {best_val['MRR']:.4f}")
    print("")
    print("Frozen SHARED1000 gate")
    print(f"  SHARED R@1 / R@5 / R@10:  {shared_best['R@1']:.1%} / {shared_best['R@5']:.1%} / {shared_best['R@10']:.1%}")
    print(f"  SHARED MedR / MRR:        {shared_best['median_rank']:.1f} / {shared_best['MRR']:.4f}")
    print("")
    print("Gains")
    print(f"  Over compact CSLS (VAL):  {val_metrics_payload['gain_over_compact_csls']:+.1%}")
    print(f"  Over compact CSLS (S1000):{shared_metrics_payload['gain_over_compact_csls']:+.1%}")
    if val_metrics_payload["gain_over_fixed_tri_fusion"] is not None:
        print(f"  Over fixed tri (VAL):     {val_metrics_payload['gain_over_fixed_tri_fusion']:+.1%}")
    else:
        print("  Over fixed tri (VAL):     n/a (fixed tri metrics missing)")
    if shared_metrics_payload["gain_over_fixed_tri_fusion"] is not None:
        print(f"  Over fixed tri (S1000):   {shared_metrics_payload['gain_over_fixed_tri_fusion']:+.1%}")
    else:
        print("  Over fixed tri (S1000):   n/a (fixed tri metrics missing)")
    print("")
    print(f"Saved JSON: {summary_path}")
    print(f"Saved CSV:  {csv_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
