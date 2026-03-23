#!/usr/bin/env python3
"""Diagnose whether a multi-hypothesis compact vMF head is actually using its components.

This utility is aimed at V42-style runs where compact component arrays are saved
alongside the usual validation/shared1000 outputs.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

import numpy as np
import yaml

_EMBEDDING_EVAL_PATH = SRC_ROOT / "fmri2img" / "eval" / "embedding_eval.py"
_spec = importlib.util.spec_from_file_location("embedding_eval_local", _EMBEDDING_EVAL_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load embedding_eval module from {_EMBEDDING_EVAL_PATH}")
_embedding_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_embedding_eval)
compute_mixture_component_diagnostics = _embedding_eval.compute_mixture_component_diagnostics
compute_mixture_vmf_retrieval_metrics = _embedding_eval.compute_mixture_vmf_retrieval_metrics
compute_retrieval_metrics = _embedding_eval.compute_retrieval_metrics
compute_retrieval_metrics_csls = _embedding_eval.compute_retrieval_metrics_csls

logger = logging.getLogger(__name__)


def _load_array(metrics_dir: Path, name: str) -> Optional[np.ndarray]:
    path = metrics_dir / name
    if not path.exists():
        return None
    return np.load(path)


def _load_split(metrics_dir: Path, split: str) -> Optional[Dict[str, np.ndarray]]:
    mu = _load_array(metrics_dir, f"{split}_predictions_compact_component_mu.npy")
    kappa = _load_array(metrics_dir, f"{split}_predictions_compact_component_kappa.npy")
    if mu is None or kappa is None:
        return None
    compact_preds = _load_array(metrics_dir, f"{split}_predictions_compact.npy")
    compact_gts = _load_array(metrics_dir, f"{split}_ground_truth_compact.npy")
    logits = _load_array(metrics_dir, f"{split}_predictions_compact_component_logits.npy")
    if compact_preds is None or compact_gts is None:
        raise FileNotFoundError(
            f"Missing compact predictions/ground truth for split={split} in {metrics_dir}"
        )
    return {
        "component_mu": mu,
        "component_kappa": kappa,
        "component_logits": logits,
        "compact_preds": compact_preds,
        "compact_gts": compact_gts,
    }


def _per_component_report(component_mu: np.ndarray, ground_truth: np.ndarray) -> list[Dict[str, float]]:
    rows = []
    for idx in range(component_mu.shape[1]):
        mu_i = component_mu[:, idx, :]
        ret = compute_retrieval_metrics(mu_i, ground_truth, ks=(1, 5, 10), normalize=True)
        csls = compute_retrieval_metrics_csls(mu_i, ground_truth, ks=(1, 5, 10), normalize=True, csls_k=10)
        rows.append({
            "component": idx,
            "r@1": float(ret["top1_accuracy"]),
            "r@5": float(ret["top5_accuracy"]),
            "r@10": float(ret["top10_accuracy"]),
            "mrr": float(ret["mrr"]),
            "csls_r@1": float(csls["top1_accuracy"]),
        })
    return rows


def _top1_disagreement_rate(component_mu: np.ndarray, ground_truth: np.ndarray) -> float:
    comp = component_mu / np.clip(np.linalg.norm(component_mu, axis=-1, keepdims=True), 1e-8, None)
    gts = ground_truth / np.clip(np.linalg.norm(ground_truth, axis=-1, keepdims=True), 1e-8, None)
    sims = np.einsum("nmd,kd->nmk", comp, gts)
    top1 = np.argmax(sims, axis=-1)
    anchor = top1[:, :1]
    disagree = np.any(top1 != anchor, axis=1)
    return float(disagree.mean())


def _analyze_split(metrics_dir: Path, split: str) -> Optional[Dict[str, Any]]:
    payload = _load_split(metrics_dir, split)
    if payload is None:
        logger.info("No compact component arrays found for split=%s", split)
        return None

    mu = payload["component_mu"]
    kappa = payload["component_kappa"]
    logits = payload["component_logits"]
    compact_preds = payload["compact_preds"]
    compact_gts = payload["compact_gts"]

    compact_raw = compute_retrieval_metrics(compact_preds, compact_gts, ks=(1, 5, 10), normalize=True)
    compact_csls = compute_retrieval_metrics_csls(compact_preds, compact_gts, ks=(1, 5, 10), normalize=True, csls_k=10)
    mixture = compute_mixture_vmf_retrieval_metrics(mu, kappa, compact_gts, component_logits=logits, ks=(1, 5, 10), normalize=True)
    component_diag = compute_mixture_component_diagnostics(mu, kappa, component_logits=logits)
    component_rows = _per_component_report(mu, compact_gts)

    return {
        "n_images": int(compact_preds.shape[0]),
        "component_shape": list(mu.shape),
        "compact_raw": {
            "r@1": float(compact_raw["top1_accuracy"]),
            "r@5": float(compact_raw["top5_accuracy"]),
            "r@10": float(compact_raw["top10_accuracy"]),
            "mrr": float(compact_raw["mrr"]),
        },
        "compact_csls": {
            "r@1": float(compact_csls["top1_accuracy"]),
            "r@5": float(compact_csls["top5_accuracy"]),
            "r@10": float(compact_csls["top10_accuracy"]),
            "mrr": float(compact_csls["mrr"]),
        },
        "mixture": {
            "r@1": float(mixture["top1_accuracy"]),
            "r@5": float(mixture["top5_accuracy"]),
            "r@10": float(mixture["top10_accuracy"]),
            "mrr": float(mixture["mrr"]),
        },
        "mixture_minus_compact_raw_r@1": float(mixture["top1_accuracy"] - compact_raw["top1_accuracy"]),
        "mixture_minus_compact_csls_r@1": float(mixture["top1_accuracy"] - compact_csls["top1_accuracy"]),
        "component_top1_disagreement_rate": _top1_disagreement_rate(mu, compact_gts),
        "component_diagnostics": {k: float(v) for k, v in component_diag.items()},
        "per_component_retrieval": component_rows,
    }


def _load_config(results_dir: Path) -> Dict[str, Any]:
    config_path = results_dir / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _load_summary(metrics_dir: Path) -> Dict[str, Any]:
    path = metrics_dir / "summary.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _load_history(metrics_dir: Path) -> list[dict[str, str]]:
    path = metrics_dir / "training_log.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _maybe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_observed(history: list[dict[str, str]], key: str) -> Optional[Dict[str, float]]:
    points = []
    for row in history:
        val = _maybe_float(row.get(key))
        if val is None:
            continue
        points.append({"epoch": int(row["epoch"]), "value": float(val)})
    if not points:
        return None
    return max(points, key=lambda item: item["value"])


def _analyze_checkpoint_behavior(results_dir: Path, metrics_dir: Path) -> Dict[str, Any]:
    config = _load_config(results_dir)
    history = _load_history(metrics_dir)
    summary = _load_summary(metrics_dir)
    training_cfg = config.get("training", {}) if isinstance(config, dict) else {}
    eval_cfg = config.get("evaluation", {}) if isinstance(config, dict) else {}
    checkpoint_metric = eval_cfg.get("checkpoint_metric") or training_cfg.get("checkpoint_metric") or "r@1"
    min_delta = float(training_cfg.get("early_stop_min_delta", 0.001))
    saved_best_epoch = summary.get("best_epoch")
    saved_best_metric = _maybe_float(summary.get("best_checkpoint_metric_value"))
    observed_ckpt = _best_observed(history, f"val_{checkpoint_metric}")
    observed_r1 = _best_observed(history, "val_r@1")

    blocked_by_delta = False
    if observed_ckpt and saved_best_metric is not None:
        blocked_by_delta = observed_ckpt["value"] > saved_best_metric and (observed_ckpt["value"] - saved_best_metric) <= min_delta

    return {
        "checkpoint_metric": checkpoint_metric,
        "early_stop_min_delta": min_delta,
        "saved_best_epoch": saved_best_epoch,
        "saved_best_metric": saved_best_metric,
        "observed_best_checkpoint_metric": observed_ckpt,
        "observed_best_r@1": observed_r1,
        "saved_vs_observed_gap": None if not observed_ckpt or saved_best_metric is None else float(observed_ckpt["value"] - saved_best_metric),
        "improvement_was_blocked_by_min_delta": bool(blocked_by_delta),
    }


def _root_causes(report: Dict[str, Any]) -> list[str]:
    findings: list[str] = []
    ckpt = report.get("checkpoint_behavior", {})
    if ckpt.get("improvement_was_blocked_by_min_delta"):
        findings.append(
            "The checkpoint metric improved later in training, but the gain was smaller than early_stop_min_delta, so the run kept the earlier checkpoint."
        )
    for split_name in ("val", "shared1000"):
        split = report.get("splits", {}).get(split_name)
        if not split:
            continue
        diag = split.get("component_diagnostics", {})
        if (
            diag.get("component_pairwise_cos_mean", 0.0) > 0.999
            and diag.get("component_weight_entropy_norm_mean", 0.0) > 0.99
            and diag.get("component_kappa_across_component_std_mean", 1.0) < 1e-3
        ):
            findings.append(
                f"{split_name}: the mixture head collapsed completely — component directions are identical, weights are uniform, and kappas are effectively identical."
            )
        if abs(split.get("mixture_minus_compact_raw_r@1", 0.0)) < 1e-4:
            findings.append(
                f"{split_name}: mixture retrieval is indistinguishable from raw compact retrieval, so the additional hypotheses are not contributing new ranking signal."
            )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose mixture-vMF compact-head collapse")
    parser.add_argument("results_dir", type=Path, help="Experiment subject directory (e.g. experimental_results/V42.../subj01)")
    args = parser.parse_args()

    results_dir = args.results_dir
    metrics_dir = results_dir / "metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    diagnostics_dir = results_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "results_dir": str(results_dir),
        "checkpoint_behavior": _analyze_checkpoint_behavior(results_dir, metrics_dir),
        "splits": {},
    }
    for split in ("val", "shared1000"):
        split_report = _analyze_split(metrics_dir, split)
        if split_report is not None:
            report["splits"][split] = split_report
    report["root_causes"] = _root_causes(report)

    out_path = diagnostics_dir / "mixture_vmf_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 78)
    print("MIXTURE-vMF DIAGNOSTICS")
    print("=" * 78)
    ckpt = report["checkpoint_behavior"]
    print(f"Checkpoint metric: {ckpt['checkpoint_metric']}  |  early_stop_min_delta={ckpt['early_stop_min_delta']}")
    print(f"Saved best epoch/value: {ckpt['saved_best_epoch']} / {ckpt['saved_best_metric']}")
    print(f"Observed best checkpoint metric: {ckpt['observed_best_checkpoint_metric']}")
    print(f"Observed best compact R@1: {ckpt['observed_best_r@1']}")
    for split_name, split_report in report["splits"].items():
        diag = split_report["component_diagnostics"]
        print("-" * 78)
        print(split_name)
        print(f"  compact raw R@1      : {split_report['compact_raw']['r@1']:.4f}")
        print(f"  compact CSLS R@1     : {split_report['compact_csls']['r@1']:.4f}")
        print(f"  mixture R@1          : {split_report['mixture']['r@1']:.4f}")
        print(f"  pairwise cos mean    : {diag['component_pairwise_cos_mean']:.6f}")
        print(f"  weight entropy norm  : {diag['component_weight_entropy_norm_mean']:.6f}")
        print(f"  top weight mean      : {diag['component_top_weight_mean']:.6f}")
        print(f"  kappa across std mean: {diag['component_kappa_across_component_std_mean']:.6f}")
        print(f"  top1 disagreement    : {split_report['component_top1_disagreement_rate']:.6f}")
    if report['root_causes']:
        print("-" * 78)
        print("Root causes")
        for item in report['root_causes']:
            print(f"  - {item}")
    print(f"Saved JSON: {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
