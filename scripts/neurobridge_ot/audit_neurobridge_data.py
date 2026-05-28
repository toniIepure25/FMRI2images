#!/usr/bin/env python3
"""
NeuroBridge-OT Data Audit
============================

Verify data availability, ROI mappings, target caches, subject data,
and teacher artifacts before running experiments.

Usage:
    python scripts/neurobridge_ot/audit_neurobridge_data.py \
        --subjects subj01 subj02 subj05 subj07 \
        --check-teachers
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def check_subject_data(subject: str, cache_root: Path, index_root: Path) -> dict:
    """Check data availability for a subject."""
    result = {"subject": subject, "status": "ok", "issues": []}

    fmri_path = cache_root / f"subject={subject}" / "fmri_features.npy"
    index_path = index_root / f"subject={subject}" / "index.parquet"

    if not fmri_path.exists():
        result["issues"].append(f"fMRI features not found: {fmri_path}")
        result["status"] = "missing_data"
    else:
        fmri = np.load(fmri_path, mmap_mode="r")
        result["n_trials"] = fmri.shape[0]
        result["n_voxels"] = fmri.shape[1]
        result["fmri_dtype"] = str(fmri.dtype)
        result["fmri_size_mb"] = fmri.nbytes / 1e6

    if not index_path.exists():
        result["issues"].append(f"Index parquet not found: {index_path}")
        result["status"] = "missing_data"
    else:
        import pandas as pd
        df = pd.read_parquet(index_path)
        result["index_rows"] = len(df)
        result["columns"] = list(df.columns)
        result["has_shared1000"] = "shared1000" in df.columns
        result["has_nsdId"] = "nsdId" in df.columns
        if "shared1000" in df.columns:
            result["n_shared1000"] = int(df["shared1000"].fillna(False).sum())
        result["unique_nsd_ids"] = int(df["nsdId"].nunique()) if "nsdId" in df.columns else 0

    return result


def check_roi_indices(subject: str, nsd_root: Path) -> dict:
    """Check ROI index availability."""
    result = {"subject": subject, "status": "ok", "rois": {}}

    try:
        from fmri2img.data.roi_utils import build_roi_index
        roi_dims, roi_indices = build_roi_index(subject)
        result["n_rois"] = len(roi_dims)
        for name, n_vox in roi_dims.items():
            result["rois"][name] = n_vox
        result["total_voxels"] = sum(roi_dims.values())
    except ImportError:
        result["status"] = "roi_utils_not_importable"
        result["issues"] = ["Cannot import build_roi_index — install package or check NSD_DATA_ROOT"]
    except Exception as e:
        result["status"] = "roi_build_failed"
        result["issues"] = [str(e)]

    return result


def check_clip_cache(clip_path: Path) -> dict:
    """Check CLIP embedding cache."""
    result = {"status": "ok"}
    if not clip_path.exists():
        result["status"] = "not_found"
        result["path"] = str(clip_path)
        return result

    import pandas as pd
    df = pd.read_parquet(clip_path)
    result["n_entries"] = len(df)
    result["columns"] = list(df.columns)[:10]
    result["path"] = str(clip_path)
    return result


def check_teacher_artifacts(output_root: Path, subjects: list) -> dict:
    """Check teacher model artifacts availability."""
    result = {"teachers": {}}
    teacher_models = ["V61a_finetune_difflr", "V62a_cls_retrieval_768d", "V66a_roi_pretrain"]

    for model_name in teacher_models:
        result["teachers"][model_name] = {}
        for subj in subjects:
            ckpt = output_root / model_name / subj / "checkpoints" / "best.pt"
            result["teachers"][model_name][subj] = {
                "checkpoint_exists": ckpt.exists(),
                "path": str(ckpt),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description="NeuroBridge-OT Data Audit")
    parser.add_argument("--subjects", nargs="+",
                        default=["subj01", "subj02", "subj05", "subj07"])
    parser.add_argument("--check-teachers", action="store_true")
    parser.add_argument("--check-rois", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cache_root = Path(os.environ.get("CACHE_ROOT", "cache")) / "preextracted"
    index_root = Path("data/indices/nsd_index")
    nsd_root = Path(os.environ.get("NSD_DATA_ROOT", "data/nsd"))
    output_root = Path(os.environ.get("OUTPUT_ROOT", "experimental_results"))

    logger.info("=" * 70)
    logger.info("NeuroBridge-OT Data Audit")
    logger.info("=" * 70)
    logger.info("Cache root: %s", cache_root)
    logger.info("Index root: %s", index_root)
    logger.info("NSD root: %s", nsd_root)

    audit = {"subjects": {}, "clip_cache": {}, "environment": {}}

    # Check environment
    audit["environment"] = {
        "CACHE_ROOT": os.environ.get("CACHE_ROOT", "NOT SET"),
        "NSD_DATA_ROOT": os.environ.get("NSD_DATA_ROOT", "NOT SET"),
        "OUTPUT_ROOT": os.environ.get("OUTPUT_ROOT", "NOT SET"),
        "CUDA_AVAILABLE": str(__import__("torch").cuda.is_available()),
    }

    # Check subjects
    for subj in args.subjects:
        logger.info("\nChecking %s...", subj)
        subj_result = check_subject_data(subj, cache_root, index_root)
        audit["subjects"][subj] = subj_result

        if subj_result["issues"]:
            for issue in subj_result["issues"]:
                logger.warning("  [%s] %s", subj, issue)
        else:
            logger.info("  OK: %d trials, %d voxels",
                        subj_result.get("n_trials", 0), subj_result.get("n_voxels", 0))

    # Check CLIP cache
    for path_name in ["outputs/clip_cache/clip_multilayer.parquet",
                      "outputs/clip_cache/clip.parquet"]:
        clip_path = Path(path_name)
        audit["clip_cache"][path_name] = check_clip_cache(clip_path)
        if audit["clip_cache"][path_name]["status"] == "ok":
            logger.info("\nCLIP cache: %s — %d entries",
                        path_name, audit["clip_cache"][path_name]["n_entries"])

    # Check ROI indices
    if args.check_rois:
        audit["roi_indices"] = {}
        for subj in args.subjects:
            audit["roi_indices"][subj] = check_roi_indices(subj, nsd_root)
            if audit["roi_indices"][subj]["status"] == "ok":
                logger.info("\n  ROI indices %s: %d ROIs, %d total voxels",
                            subj, audit["roi_indices"][subj]["n_rois"],
                            audit["roi_indices"][subj]["total_voxels"])

    # Check teachers
    if args.check_teachers:
        audit["teachers"] = check_teacher_artifacts(output_root, args.subjects)
        logger.info("\nTeacher artifacts:")
        for model, subj_map in audit["teachers"]["teachers"].items():
            available = sum(1 for s in subj_map.values() if s["checkpoint_exists"])
            logger.info("  %s: %d/%d subjects available", model, available, len(subj_map))

    # Summary
    n_ok = sum(1 for s in audit["subjects"].values() if s["status"] == "ok")
    n_total = len(audit["subjects"])
    logger.info("\n" + "=" * 70)
    logger.info("AUDIT SUMMARY: %d/%d subjects ready", n_ok, n_total)

    if n_ok < n_total:
        logger.warning("Some subjects have missing data. Run 'make preextract' first.")

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("experimental_results/neurobridge_ot_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    logger.info("Audit saved to %s", out_path)


if __name__ == "__main__":
    main()
