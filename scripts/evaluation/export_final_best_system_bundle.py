#!/usr/bin/env python3
"""Export the final metrics bundle for the frozen best retrieval system."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from final_best_system import (
    DEFAULT_LEGACY_RESULTS_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRI_RESULTS_DIR,
    EXPECTED_FROZEN_TRI_CONFIG,
    EXPECTED_SHARED1000_R1,
    load_final_best_system_analysis,
    split_summary_payload,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _scalar_row(split: str, system_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": split,
        "system": system_name,
        "R@1": metrics.get("R@1", metrics.get("fused_r@1", "")),
        "R@5": metrics.get("R@5", metrics.get("fused_r@5", "")),
        "R@10": metrics.get("R@10", metrics.get("fused_r@10", "")),
        "MedR": metrics.get("median_rank", metrics.get("fused_median_rank", "")),
        "MRR": metrics.get("MRR", metrics.get("fused_mrr", "")),
    }


def _flatten_summary_rows(split_payload: dict[str, Any]) -> list[dict[str, Any]]:
    split = split_payload["split"]
    rows = [
        _scalar_row(split, "compact_raw", split_payload["compact_raw"]),
        _scalar_row(split, "compact_csls", split_payload["compact_csls"]),
        _scalar_row(split, "rerank_only", split_payload["rerank_only"]),
        _scalar_row(split, "legacy_raw", split_payload["legacy_raw"]),
        _scalar_row(split, "legacy_csls", split_payload["legacy_csls"]),
        _scalar_row(split, "fixed_tri_expert", split_payload["fixed_tri_expert"]),
    ]
    if "fixed_two_expert_fusion" in split_payload:
        rows.append(_scalar_row(split, "fixed_two_expert_fusion", split_payload["fixed_two_expert_fusion"].get("fused", {})))
    tri_row = rows[-1] if rows else {}
    for row in rows:
        row["gain_over_compact_csls"] = split_payload.get("gain_over_compact_csls", "") if row["system"] == "fixed_tri_expert" else ""
        row["gain_over_legacy_csls"] = split_payload.get("gain_over_legacy_csls", "") if row["system"] == "fixed_tri_expert" else ""
        row["gain_over_two_expert_fusion"] = split_payload.get("gain_over_two_expert_fusion", "") if row["system"] == "fixed_tri_expert" else ""
    return rows


def _json_safe_per_query(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe_rows: list[dict[str, Any]] = []
    for row in rows:
        safe: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (list, tuple)):
                safe[key] = json.dumps(value)
            else:
                safe[key] = value
        safe_rows.append(safe)
    return safe_rows


def _write_readme(output_dir: Path, tri_results_dir: Path, legacy_results_dir: Path) -> None:
    text = f"""# Final Best System

This directory contains the frozen final retrieval system outputs for thesis/report/presentation.

## Best System

The production retrieval system is the fixed tri-expert fusion built from:

- tri/main results dir: `{tri_results_dir}`
- legacy results dir: `{legacy_results_dir}`

Frozen fusion recipe:

- compact score: `{EXPECTED_FROZEN_TRI_CONFIG['compact_score']}`
- legacy score: `{EXPECTED_FROZEN_TRI_CONFIG['legacy_score']}`
- family: `{EXPECTED_FROZEN_TRI_CONFIG['family']}`
- normalization: `{EXPECTED_FROZEN_TRI_CONFIG['normalization']}`
- shortlist_k: `{EXPECTED_FROZEN_TRI_CONFIG['shortlist_k']}`
- alpha / beta / gamma: `{EXPECTED_FROZEN_TRI_CONFIG['alpha']} / {EXPECTED_FROZEN_TRI_CONFIG['beta']} / {EXPECTED_FROZEN_TRI_CONFIG['gamma']}`
- expected SHARED1000 R@1: `{EXPECTED_SHARED1000_R1:.1%}`

## Final Files

- `final_metrics_summary.json`: compact final metrics bundle for VAL and SHARED1000
- `final_metrics_summary.csv`: report-friendly table of the main systems/metrics
- `per_query_val_predictions.csv`: per-query frozen best-system predictions on VAL
- `per_query_shared1000_predictions.csv`: per-query frozen best-system predictions on SHARED1000

## Qualitatives And Reconstructions

- `qualitatives/val/`
- `qualitatives/shared1000/`
- `qualitatives/contact_sheets/`
- `reconstructions/val/`
- `reconstructions/shared1000/`

## Regeneration

Metrics bundle:

```bash
python3 scripts/evaluation/export_final_best_system_bundle.py \
  --tri-results-dir {tri_results_dir} \
  --legacy-results-dir {legacy_results_dir}
```

Qualitatives + retrieval reconstructions:

```bash
python3 scripts/evaluation/export_final_qualitatives.py \
  --tri-results-dir {tri_results_dir} \
  --legacy-results-dir {legacy_results_dir} \
  --stimuli-hdf5 /path/to/nsd_stimuli.hdf5
```

## Notes

- This package freezes the validated fixed tri-expert system exactly.
- No new training is introduced by these export scripts.
- Retrieval-based reconstruction uses the fused top-1 retrieved image as the final reconstruction.
- Optional generative export is intentionally not part of the default pipeline.
"""
    path = output_dir / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tri-results-dir", type=Path, default=DEFAULT_TRI_RESULTS_DIR)
    parser.add_argument("--legacy-results-dir", type=Path, default=DEFAULT_LEGACY_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-config-drift", action="store_true")
    args = parser.parse_args()

    system, val_analysis, shared_analysis = load_final_best_system_analysis(
        tri_results_dir=args.tri_results_dir,
        legacy_results_dir=args.legacy_results_dir,
        strict=not args.allow_config_drift,
    )

    val_summary = split_summary_payload(system, val_analysis)
    shared_summary = split_summary_payload(system, shared_analysis)

    payload = {
        "system": {
            "tri_results_dir": str(system.tri_results_dir),
            "legacy_results_dir": str(system.legacy_results_dir),
            "frozen_setting_source": str(system.diagnostics_path),
            "frozen_tri_fusion_config": system.frozen_config,
            "expected_shared1000_r@1": EXPECTED_SHARED1000_R1,
            "saved_shared1000_r@1": system.frozen_shared1000_r1,
        },
        "val": val_summary,
        "shared1000": shared_summary,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "final_metrics_summary.json", payload)

    summary_rows = _flatten_summary_rows(val_summary) + _flatten_summary_rows(shared_summary)
    _write_csv(output_dir / "final_metrics_summary.csv", summary_rows)
    _write_csv(output_dir / "per_query_val_predictions.csv", _json_safe_per_query(val_analysis.per_query))
    _write_csv(output_dir / "per_query_shared1000_predictions.csv", _json_safe_per_query(shared_analysis.per_query))
    _write_readme(output_dir, system.tri_results_dir, system.legacy_results_dir)

    logger.info("Saved final metrics bundle to %s", output_dir)
    logger.info("VAL fixed tri R@1: %.1f%%", float(val_summary["fixed_tri_expert"]["R@1"]) * 100)
    logger.info("SHARED1000 fixed tri R@1: %.1f%%", float(shared_summary["fixed_tri_expert"]["R@1"]) * 100)


if __name__ == "__main__":
    main()
