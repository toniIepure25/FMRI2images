#!/usr/bin/env python3
"""Promotion gate for the all-8 screening ladder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_summary(results_dir: Path) -> Dict[str, Any]:
    path = results_dir / "metrics" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary.json: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid summary payload: {path}")
    return data


def _load_training_rows(results_dir: Path) -> List[Dict[str, str]]:
    path = results_dir / "metrics" / "training_log.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing training_log.csv: {path}")
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _parse_float(row: Dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None or value == "":
            continue
        return float(value)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-dir", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--metric-name", default=None)
    parser.add_argument("--require-mixture", action="store_true")
    parser.add_argument("--max-pairwise-cos", type=float, default=0.98)
    parser.add_argument("--min-top-weight", type=float, default=0.35)
    parser.add_argument("--max-top-weight", type=float, default=0.85)
    parser.add_argument("--min-kappa-std", type=float, default=0.05)
    args = parser.parse_args()

    current_dir = Path(args.current_dir)
    baseline_dir = Path(args.baseline_dir)

    current_summary = _load_summary(current_dir)
    baseline_summary = _load_summary(baseline_dir)

    metric_name = args.metric_name or current_summary.get("checkpoint_metric") or baseline_summary.get("checkpoint_metric")
    if not metric_name:
        raise ValueError("Could not determine checkpoint metric name")

    current_best = float(current_summary.get("best_checkpoint_metric_value", float("-inf")))
    baseline_best = float(baseline_summary.get("best_checkpoint_metric_value", float("-inf")))

    payload: Dict[str, Any] = {
        "metric_name": metric_name,
        "current_best": current_best,
        "baseline_best": baseline_best,
        "beat_baseline": current_best > baseline_best,
    }
    if current_best <= baseline_best:
        print(json.dumps(payload, indent=2))
        return 1

    if args.require_mixture:
        rows = _load_training_rows(current_dir)
        metric_col = f"val_{metric_name}"
        valid_rows = [row for row in rows if row.get(metric_col) not in (None, "")]
        if not valid_rows:
            payload["mixture_gate"] = "missing_metric_rows"
            print(json.dumps(payload, indent=2))
            return 1
        best_row = max(valid_rows, key=lambda row: float(row[metric_col]))
        pairwise_cos = _parse_float(best_row, "val_component_pairwise_cos_mean", "val_mixture_pairwise_cos")
        top_weight = _parse_float(best_row, "val_component_top_weight_mean", "val_mixture_top_weight_mean")
        kappa_std = _parse_float(
            best_row,
            "val_component_kappa_across_component_std_mean",
            "val_mixture_kappa_std_mean",
        )
        payload.update(
            {
                "mixture_pairwise_cos": pairwise_cos,
                "mixture_top_weight": top_weight,
                "mixture_kappa_std": kappa_std,
            }
        )
        if pairwise_cos is None or top_weight is None or kappa_std is None:
            payload["mixture_gate"] = "missing_diagnostics"
            print(json.dumps(payload, indent=2))
            return 1
        mixture_ok = (
            pairwise_cos < args.max_pairwise_cos
            and args.min_top_weight <= top_weight <= args.max_top_weight
            and kappa_std > args.min_kappa_std
        )
        payload["mixture_gate"] = mixture_ok
        if not mixture_ok:
            print(json.dumps(payload, indent=2))
            return 1

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
