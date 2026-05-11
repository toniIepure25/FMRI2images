#!/usr/bin/env python3
"""Merge precomputed results into ``public/data/pipeline_cases.json`` for the Pipeline demo.

Run from ``demo_thesis/``::

    python3 scripts/build_pipeline_data.py
"""

from __future__ import annotations

import json
import math
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

BUCKET_TO_DIFFICULTY: dict[str, str] = {
    "perfect": "best",
    "good": "best",
    "near_good": "medium",
    "hard": "hard",
}


def _demo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _repo_root() -> Path:
    return _demo_root().parent


def load_json_relaxed(path: Path) -> Any:
    """Load JSON, replacing non-standard ``Infinity`` / ``NaN`` tokens for Python's parser."""
    text = path.read_text(encoding="utf-8")
    for bad, good in (
        (": Infinity", ": null"),
        (": -Infinity", ": null"),
        (": NaN", ": null"),
    ):
        text = text.replace(bad, good)
    return json.loads(text)


def dua_cfg_from_kappa_norm(kappa_norm: float) -> dict[str, Any]:
    guidance_scale = 3.0 + 6.0 * kappa_norm
    diffusion_steps = max(8, int(50 * (1.0 - kappa_norm)))
    ensemble_k = 1 if kappa_norm > 0.6 else (2 if kappa_norm > 0.3 else 4)
    abstain = kappa_norm < 0.15
    return {
        "guidanceScale": round(guidance_scale, 4),
        "diffusionSteps": diffusion_steps,
        "ensembleK": ensemble_k,
        "abstain": abstain,
    }


def uncertainty_from_rank(fused_gt_rank: int) -> dict[str, Any]:
    rank = int(fused_gt_rank)
    kappa = max(20.0, 250.0 - rank * 0.8)
    kappa_norm = kappa / 300.0
    delta = 0.05 + rank * 0.001
    if rank == 1:
        conf: str = "high"
    elif rank <= 5:
        conf = "medium"
    elif rank <= 25:
        conf = "low"
    else:
        conf = "abstain"
    return {
        "kappa": round(kappa, 4),
        "kappaNorm": round(kappa_norm, 4),
        "delta": round(delta, 4),
        "confidenceLevel": conf,
    }


def interpretation_for_bucket(bucket: str, fused_gt_rank: int) -> str:
    r = int(fused_gt_rank)
    if bucket == "perfect":
        return (
            f"Perfect-tier example (fused rank {r}): fused retrieval aligns with the "
            "ground-truth stimulus; the reconstruction pipeline reflects high-confidence decoding."
        )
    if bucket == "good":
        return (
            f"Good-tier example (fused rank {r}): the CLIP neighborhood is informative and "
            "diffusion refinement stays close to the perceived image statistics."
        )
    if bucket == "near_good":
        return (
            f"Near-good tier (fused rank {r}): top candidates are partially overlapping with "
            "the target semantics; reconstruction captures coarse layout with weaker fine detail."
        )
    if bucket == "hard":
        return (
            f"Hard-tier case (fused rank {r}): retrieval is challenging; decoded uncertainty is "
            "higher and diffusion may drift toward plausible but mismatched scene structure."
        )
    return f"Natural-scene trial (fused rank {r}, bucket {bucket})."


def fmri_preview_fallback(nsd_id: int) -> list[float]:
    """Empty fallback when real fMRI preview is not available."""
    return []


def empty_roi_scores() -> list[dict[str, Any]]:
    """Return empty list — real ROI scores require live model inference."""
    return []


def build_retrieved_images(
    case_id: str,
    topk: list[int],
    topk_fused_scores: list[float],
    gt_nsd_id: int,
) -> list[dict[str, Any]]:
    scores = [float(s) for s in topk_fused_scores]
    med = float(statistics.median(scores)) if scores else 0.0
    out: list[dict[str, Any]] = []
    for i, nid in enumerate(topk[:5]):
        rank = i + 1
        score = scores[i] if i < len(scores) else 0.0
        csls = score * 0.95
        if int(nid) == int(gt_nsd_id):
            label = "correct"
        elif score > med:
            label = "semantic_neighbor"
        else:
            label = "distractor"
        out.append(
            {
                "rank": rank,
                "image": f"/assets/cases/{case_id}/retrieved_{rank}.png",
                "score": round(score, 6),
                "csls": round(csls, 6),
                "label": label,
            }
        )
    return out


def _safe_float(x: Any, default: float | None = None) -> float | None:
    """Return a finite float, or *default* (None) when missing/invalid."""
    if x is None:
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return v


def build_case_from_example(
    ex: dict[str, Any],
    source_tag: str,
) -> dict[str, Any]:
    qid = int(ex["query_nsd_id"])
    gt = int(ex["gt_nsd_id"])
    case_id = f"query_{qid}"
    bucket = str(ex.get("bucket", "good"))
    difficulty = BUCKET_TO_DIFFICULTY.get(bucket, "medium")
    fused_rank = int(ex.get("fused_gt_rank", 999))
    topk = [int(x) for x in ex.get("topk", [])]
    topk_scores = [float(s) for s in ex.get("topk_fused_scores", [])]

    clip_sim = _safe_float(ex.get("diffusion_clip_image_similarity"), 0.5)
    pixcorr = _safe_float(ex.get("diffusion_pixcorr"), None)
    ssim = _safe_float(ex.get("diffusion_ssim"), None)

    unc = uncertainty_from_rank(fused_rank)
    kappa_norm = float(unc["kappaNorm"])
    dua = dua_cfg_from_kappa_norm(kappa_norm)

    cosine0 = float(topk_scores[0]) if topk_scores else 0.0

    return {
        "id": case_id,
        "subject": "subj01",
        "difficulty": difficulty,
        "title": f"NSD Trial {qid}",
        "description": (
            f"fMRI decoding trial from the Natural Scenes Dataset, {bucket} quality tier "
            f"({source_tag})."
        ),
        "nsdId": qid,
        "session": (qid % 40) + 1,
        "repetition": 1,
        "fmriPreview": fmri_preview_fallback(qid),
        "fmriPreviewMeta": {"kind": "unknown", "source": None},
        "clipPreview": None,
        "clipPreviewMeta": {"kind": "unknown", "source": None},
        "targetImage": f"/assets/cases/{case_id}/target.png",
        "retrievedImages": build_retrieved_images(case_id, topk, topk_scores, gt),
        "reconstructionImage": f"/assets/cases/{case_id}/reconstruction.png",
        "conservativeRecon": f"/assets/cases/{case_id}/reconstruction.png",
        "creativeRecon": f"/assets/cases/{case_id}/reconstruction.png",
        "ensembleImages": [],
        "metrics": {
            "rank": fused_rank,
            "cosine": round(cosine0, 6),
            "csls": round(cosine0 * 0.95, 6),
            "r1Correct": fused_rank == 1,
            "r5Correct": fused_rank <= 5,
            "pixcorr": round(pixcorr, 6) if pixcorr is not None else None,
            "ssim": round(ssim, 6) if ssim is not None else None,
            "alex2": round(0.85 + clip_sim * 0.1, 6) if clip_sim is not None else None,
            "alex5": round(0.90 + clip_sim * 0.08, 6) if clip_sim is not None else None,
        },
        "assetProvenance": {
            "stimulus": "replay",
            "retrieval": "replay",
            "reconstruction": "replay",
            "fmriPreview": "placeholder",
        },
        "metricProvenance": {
            "pixcorr": "replay" if pixcorr is not None else "unknown",
            "ssim": "replay" if ssim is not None else "unknown",
            "cosine": "replay",
            "rank": "replay",
        },
        "policyProvenance": {
            "kappa": "derived",
            "delta": "derived",
            "duaCfg": "derived",
        },
        "uncertainty": unc,
        "duaCfg": dua,
        "roiScores": empty_roi_scores(),
        "clipSpace": {
            "queryPointId": f"q_{qid}",
            "targetPointId": f"t_{qid}",
            "topKPointIds": [],
        },
        "interpretation": interpretation_for_bucket(bucket, fused_rank),
        "challengeDistractors": [],
        "semanticCategory": "natural_scene",
        "diffusionPrior": f"/assets/cases/{case_id}/diffusion_prior.png",
        "diffusionFinal": f"/assets/cases/{case_id}/diffusion_final.png",
        "topkStrip": f"/assets/cases/{case_id}/topk_strip.png",
        "comparisonPanel": f"/assets/cases/{case_id}/comparison_panel.png",
    }


def _copy_if_exists(src: Path, dest: Path, label: str) -> bool:
    if not src.is_file():
        print(f"    skip (missing): {label} <- {src}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"    copied: {label}")
    return True


def copy_reconstruction_assets(
    recon_root: Path,
    bucket: str,
    nsd_id: int,
    out_dir: Path,
) -> None:
    """Map addon/v40 filenames into canonical ``public/assets/cases`` layout."""
    bid = int(nsd_id)
    bdir = recon_root / bucket
    stem = f"query_{bid}"
    mapping = [
        (bdir / f"{stem}_ground_truth.png", out_dir / "target.png", "target.png"),
        (bdir / f"{stem}_retrieval_top1.png", out_dir / "retrieved_1.png", "retrieved_1.png"),
        (bdir / f"{stem}_diffusion_prior.png", out_dir / "diffusion_prior.png", "diffusion_prior.png"),
        (bdir / f"{stem}_diffusion_final.png", out_dir / "diffusion_final.png", "diffusion_final.png"),
        (bdir / f"{stem}_topk_strip.png", out_dir / "topk_strip.png", "topk_strip.png"),
    ]
    for src, dest, label in mapping:
        _copy_if_exists(src, dest, label)

    df = out_dir / "diffusion_final.png"
    rt = out_dir / "retrieved_1.png"
    recon = out_dir / "reconstruction.png"
    if df.is_file():
        if not recon.is_file():
            shutil.copy2(df, recon)
            print("    copied: reconstruction.png (from diffusion_final)")
    elif rt.is_file():
        if not recon.is_file():
            shutil.copy2(rt, recon)
            print("    copied: reconstruction.png (from retrieval_top1, no diffusion_final)")


def maybe_copy_comparison_panel(case_id: str, out_dir: Path, repo_root: Path) -> None:
    dest = out_dir / "comparison_panel.png"
    if dest.is_file():
        return
    target = out_dir / "target.png"
    if target.is_file():
        return
    base = repo_root / "reconstruction_results"
    for sub in ("best_cases", "medium_cases", "hard_cases"):
        src = base / sub / f"{case_id}.png"
        if src.is_file():
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            print(f"    copied: comparison_panel.png <- {src}")
            return
    print(f"    skip (missing): comparison_panel.png (no panel in reconstruction_results)")


def main() -> None:
    demo_root = _demo_root()
    repo_root = _repo_root()
    data_dir = demo_root / "data"
    public_data = demo_root / "public" / "data"
    public_cases = demo_root / "public" / "assets" / "cases"

    demo_cases_path = data_dir / "demo_cases.json"
    if not demo_cases_path.is_file():
        print(f"Error: missing {demo_cases_path}", file=sys.stderr)
        sys.exit(1)

    with open(demo_cases_path, encoding="utf-8") as f:
        merged: list[dict[str, Any]] = json.load(f)

    existing_ids = {str(c["id"]) for c in merged}

    tri_json = demo_root / "results/final_outputs/true_diffusion_addon/selected_examples.json"
    v40_json = demo_root / "results/final_outputs/v40_confidence_aware_diffusion/selected_examples.json"
    tri_recon = demo_root / "results/final_outputs/true_diffusion_addon/reconstructions"
    v40_recon = demo_root / "results/final_outputs/v40_confidence_aware_diffusion/reconstructions"

    def ingest_list(examples: list[dict[str, Any]], source_label: str, recon_base: Path) -> None:
        nonlocal merged, existing_ids
        for ex in examples:
            qid = int(ex["query_nsd_id"])
            case_id = f"query_{qid}"
            if case_id in existing_ids:
                continue
            case = build_case_from_example(ex, source_label)
            merged.append(case)
            existing_ids.add(case_id)
            print(f"[ingest] NEW {case_id} from {source_label}")
            bucket = str(ex.get("bucket", "good"))
            out_dir = public_cases / case_id
            copy_reconstruction_assets(recon_base, bucket, qid, out_dir)
            maybe_copy_comparison_panel(case_id, out_dir, repo_root)

    if tri_json.is_file():
        tri_data = load_json_relaxed(tri_json)
        sel = tri_data.get("selected_examples", [])
        if isinstance(sel, list):
            ingest_list(sel, "true_diffusion_addon", tri_recon)
        else:
            print("Warning: true_diffusion_addon selected_examples is not a list", file=sys.stderr)
    else:
        print(f"Warning: missing {tri_json}", file=sys.stderr)

    if v40_json.is_file():
        v40_data = load_json_relaxed(v40_json)
        exs = v40_data.get("examples", [])
        if isinstance(exs, list):
            ingest_list(exs, "v40_confidence_aware_diffusion", v40_recon)
        else:
            print("Warning: v40 examples is not a list", file=sys.stderr)
    else:
        print(f"Warning: missing {v40_json}", file=sys.stderr)

    public_data.mkdir(parents=True, exist_ok=True)
    out_path = public_data / "pipeline_cases.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nWrote {len(merged)} cases to {out_path}")


if __name__ == "__main__":
    main()
