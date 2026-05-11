#!/usr/bin/env python3
"""Integrate real artifacts from K8s pod extraction into pipeline_cases.json.

Reads the manifest produced by the pod extraction (pipeline_real_artifacts_manifest.json)
and merges real fMRI previews, CLIP previews, and corrected metrics into the
existing pipeline_cases.json.

Run from demo_thesis/::

    python3 scripts/integrate_real_artifacts.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


def _demo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def main() -> None:
    demo = _demo_root()
    manifest_path = demo / "public" / "data" / "pipeline_real_artifacts_manifest.json"
    cases_path = demo / "public" / "data" / "pipeline_cases.json"

    if not manifest_path.is_file():
        print(f"Error: manifest not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)
    if not cases_path.is_file():
        print(f"Error: pipeline_cases.json not found at {cases_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    with open(cases_path, encoding="utf-8") as f:
        cases: list[dict[str, Any]] = json.load(f)

    real_cases = manifest.get("cases", {})
    updated = 0
    stats = {"fmri_real": 0, "clip_real": 0, "metrics_updated": 0}

    for case in cases:
        case_id = str(case.get("id", ""))
        real = real_cases.get(case_id)
        if real is None:
            continue

        updated += 1

        # Replace fMRI preview with real binned activation
        fmri_data = real.get("fmriPreview", {})
        if fmri_data.get("available") and fmri_data.get("values"):
            case["fmriPreview"] = fmri_data["values"]
            case["fmriPreviewMeta"] = {
                "kind": "replay",
                "source": fmri_data.get("source", "Real ROI feature vector"),
                "stats": fmri_data.get("stats"),
            }
            stats["fmri_real"] += 1
        else:
            case["fmriPreviewMeta"] = {"kind": "unknown", "source": None}

        # Add CLIP embedding preview (ground-truth)
        clip_data = real.get("clipPreview", {})
        if clip_data.get("available") and clip_data.get("values"):
            case["clipPreview"] = clip_data["values"]
            case["clipPreviewMeta"] = {
                "kind": "replay",
                "source": clip_data.get("source", "Ground-truth CLIP embedding"),
                "note": clip_data.get("note", ""),
            }
            stats["clip_real"] += 1
        else:
            case["clipPreview"] = None
            case["clipPreviewMeta"] = {"kind": "unknown", "source": None}

        # Update metrics with real experiment values where available
        real_metrics = real.get("realMetrics", {})
        m = case.get("metrics", {})

        dpixcorr = _safe_float(real_metrics.get("diffusion_pixcorr"))
        dssim = _safe_float(real_metrics.get("diffusion_ssim"))

        if dpixcorr is not None:
            m["pixcorr"] = round(dpixcorr, 6)
            stats["metrics_updated"] += 1
        if dssim is not None:
            m["ssim"] = round(dssim, 6)

        # Update provenance metadata
        case.setdefault("assetProvenance", {})
        case["assetProvenance"]["fmriPreview"] = (
            "replay" if fmri_data.get("available") else "unknown"
        )
        case["assetProvenance"]["clipPreview"] = (
            "replay" if clip_data.get("available") else "unknown"
        )

        case.setdefault("metricProvenance", {})
        case["metricProvenance"]["pixcorr"] = (
            "replay" if dpixcorr is not None else "unknown"
        )
        case["metricProvenance"]["ssim"] = (
            "replay" if dssim is not None else "unknown"
        )

        # Remove placeholder ROI scores — replaced by honest unavailable state
        if "roiScores" in case:
            case["roiScores"] = []
            case.setdefault("assetProvenance", {})["roiScores"] = "unknown"

    # Write back
    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Updated {updated}/{len(cases)} cases from real manifest")
    print(f"  fMRI real previews: {stats['fmri_real']}")
    print(f"  CLIP real previews: {stats['clip_real']}")
    print(f"  Metrics corrected:  {stats['metrics_updated']}")
    print(f"Written to {cases_path}")


if __name__ == "__main__":
    main()
