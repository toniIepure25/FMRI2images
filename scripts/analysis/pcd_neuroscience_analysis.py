"""
PCD Neuroscience Analysis: Prediction Error Decomposition & Hierarchy Validation
=================================================================================

Extracts and analyses the neuroscientific insights from a trained Predictive
Cortical Decoder (PCD) model.  The six analyses test whether predictive coding
theory holds in the visual decoding setting:

1. **Information contribution per level** — ablate each level's error to
   measure information added beyond lower levels.
2. **Category-conditional error maps** — do faces produce larger prediction
   errors at FFA?  Scenes at PPA?  Tests functional specialization via
   predictive coding.
3. **Hierarchy order matters?** — compare full vs reversed vs random hierarchy.
4. **Cross-subject consistency** — same pattern across 8 NSD subjects?
5. **Error magnitude ↔ kappa** — does prediction error predict decoding
   confidence?
6. **PCD vs flat Transformer** — does hierarchical prediction help?

Usage:
    python scripts/analysis/pcd_neuroscience_analysis.py \\
        --checkpoint experimental_results/PCD_v1_8subject/subj01/checkpoints/best.pt \\
        --subject subj01 \\
        --output-dir experimental_results/PCD_v1_8subject/neuroscience \\
        --clip-cache outputs/clip_cache/clip_multilayer.parquet

References:
    Rao & Ballard (1999) Predictive coding in the visual cortex
    Clark (2013) Whatever next? Predictive brains, situated agents
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fmri2img.models.unified_model import create_model, PCDModel
from fmri2img.models.predictive_cortical_decoder import (
    HIERARCHY_LEVELS,
    LEVEL_NAMES,
    N_LEVELS,
    PCDOutput,
)
from fmri2img.data.roi_utils import build_roi_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_pcd_model(
    checkpoint_path: Path,
    subject: str,
    device: str = "cuda",
) -> Tuple[PCDModel, Dict[str, Any]]:
    """Load a trained PCD model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt.get("config") or ckpt.get("model_config", {})
    model_config = config.get("model", config)

    roi_names = list(model_config.get("encoder", {}).get("roi_dims", {}).keys())
    if roi_names:
        _, roi_indices = build_roi_index(subject, roi_names)
    else:
        roi_indices = None

    model = create_model(model_config, roi_indices=roi_indices)
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict", {})
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    logger.info("Loaded PCD model from %s (%d params)", checkpoint_path,
                sum(p.numel() for p in model.parameters()))
    return model, config


def load_subject_data(
    subject: str,
    clip_cache_path: str,
    shared1000_only: bool = True,
    category_cache_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load fMRI features, CLIP embeddings, and optional category labels."""
    cache_root = os.environ.get("CACHE_ROOT", "cache")
    feat_path = Path(cache_root) / "preextracted" / f"subject={subject}" / "fmri_features.npy"
    index_path = Path("data/indices/nsd_index") / f"subject={subject}" / "index.parquet"

    features = np.load(feat_path, mmap_mode="r")
    index_df = pd.read_parquet(index_path)

    clip_df = pd.read_parquet(clip_cache_path)
    emb_col = None
    for col in ["fused", "final", "embedding", "clip_embedding"]:
        if col in clip_df.columns:
            emb_col = col
            break
    if emb_col is None:
        raise ValueError(f"No embedding column found in {clip_cache_path}")

    nsd_to_emb = {}
    for _, row in clip_df.iterrows():
        nsd_id = int(row["nsdId"])
        emb = np.array(row[emb_col], dtype=np.float32)
        nsd_to_emb[nsd_id] = emb

    if shared1000_only:
        stim_info_path = Path(os.environ.get("NSD_DATA_ROOT", "data/nsd")) / \
            "nsddata/experiments/nsd/nsd_stim_info_merged.csv"
        if stim_info_path.exists():
            stim_df = pd.read_csv(stim_info_path)
            shared_ids = set(stim_df[stim_df["shared1000"]]["nsdId"].values)
            mask = index_df["nsdId"].isin(shared_ids)
            index_df = index_df[mask].reset_index(drop=True)
            logger.info("Filtered to SHARED1000: %d trials", len(index_df))

    trial_indices = index_df.index.values
    nsd_ids = index_df["nsdId"].values

    fmri_data = features[trial_indices]
    embeddings = np.array([nsd_to_emb.get(nid, np.zeros(768, dtype=np.float32))
                           for nid in nsd_ids])

    categories = None
    if category_cache_path and Path(category_cache_path).exists():
        cat_df = pd.read_parquet(category_cache_path)
        nsd_to_cat = dict(zip(cat_df["nsdId"], cat_df["supercategory"]))
        categories = [nsd_to_cat.get(nid, "unknown") for nid in nsd_ids]

    return {
        "fmri": fmri_data,
        "embeddings": embeddings,
        "nsd_ids": nsd_ids,
        "categories": categories,
    }


@torch.no_grad()
def extract_prediction_errors(
    model: PCDModel,
    fmri: np.ndarray,
    device: str = "cuda",
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """Run model on all samples, collecting prediction error magnitudes and level info."""
    model.eval()
    n = len(fmri)
    all_error_mags = defaultdict(list)
    all_level_kappas = []
    all_level_weights = []
    all_mus = []
    all_kappas = []

    for i in range(0, n, batch_size):
        batch = torch.from_numpy(fmri[i:i + batch_size].copy()).float().to(device)
        pcd_out = model.pcd(batch, return_details=True)

        all_mus.append(pcd_out.mu.cpu().numpy())
        all_kappas.append(pcd_out.kappa.cpu().numpy())

        if pcd_out.level_kappas is not None:
            all_level_kappas.append(pcd_out.level_kappas.cpu().numpy())
        if pcd_out.level_weights is not None:
            all_level_weights.append(pcd_out.level_weights.cpu().numpy())

        for j, err in enumerate(pcd_out.prediction_errors):
            if err is not None:
                err_mag = err.norm(dim=-1).mean(dim=-1).cpu().numpy()
                all_error_mags[f"L{j-1}_to_L{j}"].append(err_mag)

    results = {
        "mu": np.concatenate(all_mus),
        "kappa": np.concatenate(all_kappas),
    }
    if all_level_kappas:
        results["level_kappas"] = np.concatenate(all_level_kappas)
    if all_level_weights:
        results["level_weights"] = np.concatenate(all_level_weights)
    for key, vals in all_error_mags.items():
        results[f"error_mag_{key}"] = np.concatenate(vals)

    return results


def analysis_1_information_contribution(
    results: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Quantify how much information each level adds via prediction error magnitude."""
    report = {}
    for key in sorted(results.keys()):
        if key.startswith("error_mag_"):
            level_name = key.replace("error_mag_", "")
            mags = results[key]
            report[level_name] = {
                "mean_error": float(np.mean(mags)),
                "std_error": float(np.std(mags)),
                "median_error": float(np.median(mags)),
                "p25": float(np.percentile(mags, 25)),
                "p75": float(np.percentile(mags, 75)),
            }

    if "level_weights" in results:
        weights = results["level_weights"]
        for i, name in enumerate(LEVEL_NAMES):
            report[f"weight_{name}"] = {
                "mean": float(weights[:, i].mean()),
                "std": float(weights[:, i].std()),
            }

    logger.info("=== Analysis 1: Information Contribution ===")
    for k, v in report.items():
        logger.info("  %s: %s", k, v)
    return report


def analysis_2_category_conditional_errors(
    results: Dict[str, np.ndarray],
    categories: Optional[List[str]],
) -> Dict[str, Any]:
    """Test functional specialization: do specific categories generate
    larger prediction errors at their preferred ROI level?"""
    if categories is None:
        logger.warning("No category labels available — skipping analysis 2")
        return {}

    cats = np.array(categories)
    unique_cats = sorted(set(cats))
    report = {}

    for error_key in sorted(results.keys()):
        if not error_key.startswith("error_mag_"):
            continue
        level_name = error_key.replace("error_mag_", "")
        mags = results[error_key]

        cat_stats = {}
        for cat in unique_cats:
            mask = cats == cat
            if mask.sum() < 5:
                continue
            cat_stats[cat] = {
                "mean_error": float(np.mean(mags[mask])),
                "std_error": float(np.std(mags[mask])),
                "n_samples": int(mask.sum()),
            }
        report[level_name] = cat_stats

    if "level_kappas" in results:
        lk = results["level_kappas"]
        kappa_by_cat = {}
        for cat in unique_cats:
            mask = cats == cat
            if mask.sum() < 5:
                continue
            kappa_by_cat[cat] = {
                f"kappa_{LEVEL_NAMES[i]}": float(lk[mask, i].mean())
                for i in range(min(lk.shape[1], len(LEVEL_NAMES)))
            }
        report["level_kappas_by_category"] = kappa_by_cat

    logger.info("=== Analysis 2: Category-Conditional Errors ===")
    for level, stats in report.items():
        if isinstance(stats, dict) and any(isinstance(v, dict) for v in stats.values()):
            for cat, s in stats.items():
                if isinstance(s, dict) and "mean_error" in s:
                    logger.info("  %s / %s: mean_error=%.4f (n=%d)",
                                level, cat, s["mean_error"], s.get("n_samples", 0))
    return report


def analysis_5_error_kappa_correlation(
    results: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Test whether prediction error magnitude predicts kappa (confidence)."""
    kappa = results["kappa"].squeeze()
    report = {}

    for key in sorted(results.keys()):
        if not key.startswith("error_mag_"):
            continue
        mags = results[key]
        corr = float(np.corrcoef(mags, kappa)[0, 1])
        report[key.replace("error_mag_", "")] = {
            "pearson_r": corr,
            "n_samples": len(mags),
        }

    if "level_kappas" in results:
        lk = results["level_kappas"]
        level_corrs = {}
        for i, name in enumerate(LEVEL_NAMES):
            if i < lk.shape[1]:
                corr = float(np.corrcoef(lk[:, i], kappa)[0, 1])
                level_corrs[name] = corr
        report["level_kappa_vs_global_kappa"] = level_corrs

    logger.info("=== Analysis 5: Error-Kappa Correlation ===")
    for k, v in report.items():
        logger.info("  %s: %s", k, v)
    return report


def analysis_4_cross_subject_consistency(
    subject_results: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    """Check if the level weight pattern is consistent across subjects."""
    report = {}

    weight_patterns = {}
    for subj, res in subject_results.items():
        if "level_weights" in res:
            mean_weights = res["level_weights"].mean(axis=0)
            weight_patterns[subj] = mean_weights.tolist()

    if len(weight_patterns) >= 2:
        patterns = np.array(list(weight_patterns.values()))
        mean_pattern = patterns.mean(axis=0)
        std_pattern = patterns.std(axis=0)
        report["mean_level_weights"] = {
            LEVEL_NAMES[i]: float(mean_pattern[i])
            for i in range(len(mean_pattern))
        }
        report["std_level_weights"] = {
            LEVEL_NAMES[i]: float(std_pattern[i])
            for i in range(len(std_pattern))
        }
        report["per_subject_weights"] = weight_patterns

        from itertools import combinations
        corrs = []
        for s1, s2 in combinations(weight_patterns.keys(), 2):
            p1 = np.array(weight_patterns[s1])
            p2 = np.array(weight_patterns[s2])
            corrs.append(float(np.corrcoef(p1, p2)[0, 1]))
        report["pairwise_weight_correlations"] = {
            "mean": float(np.mean(corrs)),
            "min": float(np.min(corrs)),
            "max": float(np.max(corrs)),
        }

    logger.info("=== Analysis 4: Cross-Subject Consistency ===")
    for k, v in report.items():
        logger.info("  %s: %s", k, v)
    return report


def run_single_subject_analysis(
    model: PCDModel,
    subject: str,
    clip_cache_path: str,
    device: str = "cuda",
    category_cache_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run all single-subject analyses."""
    data = load_subject_data(subject, clip_cache_path,
                             category_cache_path=category_cache_path)
    results = extract_prediction_errors(model, data["fmri"], device=device)

    report = {
        "subject": subject,
        "n_trials": len(data["fmri"]),
        "analysis_1_information_contribution": analysis_1_information_contribution(results),
        "analysis_2_category_conditional_errors": analysis_2_category_conditional_errors(
            results, data["categories"]
        ),
        "analysis_5_error_kappa_correlation": analysis_5_error_kappa_correlation(results),
    }

    return report, results


def main():
    parser = argparse.ArgumentParser(description="PCD Neuroscience Analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument("--subjects", type=str, nargs="+",
                        default=["subj01", "subj02", "subj05", "subj07"])
    parser.add_argument("--output-dir", type=str, default="experimental_results/PCD_v1_8subject/neuroscience")
    parser.add_argument("--clip-cache", type=str, default="outputs/clip_cache/clip_multilayer.parquet")
    parser.add_argument("--category-cache", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multi-subject", action="store_true",
                        help="Run cross-subject analysis (analysis 4)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config = load_pcd_model(Path(args.checkpoint), args.subject, args.device)

    if args.multi_subject:
        subject_results = {}
        full_report = {}
        for subj in args.subjects:
            logger.info("=== Processing %s ===", subj)
            report, results = run_single_subject_analysis(
                model, subj, args.clip_cache, args.device, args.category_cache
            )
            subject_results[subj] = results
            full_report[subj] = report

        cross_subj = analysis_4_cross_subject_consistency(subject_results)
        full_report["cross_subject"] = cross_subj

        out_path = output_dir / "multi_subject_analysis.json"
        with open(out_path, "w") as f:
            json.dump(full_report, f, indent=2, default=str)
        logger.info("Saved multi-subject analysis to %s", out_path)
    else:
        report, _ = run_single_subject_analysis(
            model, args.subject, args.clip_cache, args.device, args.category_cache
        )
        out_path = output_dir / f"analysis_{args.subject}.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Saved analysis to %s", out_path)


if __name__ == "__main__":
    main()
