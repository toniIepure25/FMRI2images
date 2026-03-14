#!/usr/bin/env python3
"""
Embedding Diagnostics — Hubness, Similarity, and Failure Analysis

Loads validation predictions from a trained model's output directory and
produces a structured diagnostic report.  Designed to answer: "What blocks
better R@1?"

Usage:
    python scripts/evaluation/diagnose_embeddings.py \
        --results-dir experimental_results/N1v14_vmf_nce/subj01

Output (in results-dir/diagnostics/):
    report.json          — structured JSON with all metrics
    hubness_histogram.png        — k-occurrence distribution
    similarity_distribution.png  — positive vs negative pair cosine sims
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _load_embeddings(results_dir: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load predictions, ground-truth, and optionally kappa from results dir."""
    metrics_dir = results_dir / "metrics"
    if not metrics_dir.exists():
        metrics_dir = results_dir

    preds_path = None
    gts_path = None
    kappa_path = None

    for candidate_dir in [metrics_dir, results_dir]:
        for name in ["val_predictions.npy", "predictions.npy", "val_preds.npy"]:
            p = candidate_dir / name
            if p.exists():
                preds_path = p
                break
        for name in ["val_ground_truth.npy", "ground_truth.npy", "val_gts.npy"]:
            p = candidate_dir / name
            if p.exists():
                gts_path = p
                break
        for name in ["val_kappas.npy", "kappas.npy"]:
            p = candidate_dir / name
            if p.exists():
                kappa_path = p
                break

    if preds_path is None or gts_path is None:
        raise FileNotFoundError(
            f"Cannot find prediction/ground-truth .npy files in {results_dir}. "
            "Expected val_predictions.npy and val_ground_truth.npy in metrics/ or root."
        )

    preds = np.load(preds_path)
    gts = np.load(gts_path)
    kappas = np.load(kappa_path) if kappa_path else None

    logger.info("Loaded preds %s, gts %s, kappas %s",
                preds.shape, gts.shape, kappas.shape if kappas is not None else None)
    return preds, gts, kappas


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return x / norms


def compute_hubness_analysis(preds: np.ndarray, gts: np.ndarray, k: int = 10) -> Dict:
    """Compute hubness metrics on the gallery (ground-truth) side."""
    sim = preds @ gts.T  # (N, M)
    N, M = sim.shape

    top_k_indices = np.argsort(-sim, axis=1)[:, :k]  # (N, k)
    k_occ = np.zeros(M, dtype=int)
    for row in top_k_indices:
        for idx in row:
            k_occ[idx] += 1

    expected = N * k / M
    from scipy.stats import skew
    skewness = float(skew(k_occ))
    hub_threshold = 2 * expected
    hub_fraction = float(np.mean(k_occ > hub_threshold))
    antihub_fraction = float(np.mean(k_occ == 0))

    hub_indices = np.where(k_occ > hub_threshold)[0]
    antihub_indices = np.where(k_occ == 0)[0]

    return {
        "k": k,
        "gallery_size": M,
        "num_queries": N,
        "expected_k_occ": round(expected, 2),
        "skewness": round(skewness, 4),
        "hub_fraction": round(hub_fraction, 4),
        "antihub_fraction": round(antihub_fraction, 4),
        "num_hubs": int(len(hub_indices)),
        "num_antihubs": int(len(antihub_indices)),
        "k_occ_mean": round(float(k_occ.mean()), 2),
        "k_occ_std": round(float(k_occ.std()), 2),
        "k_occ_max": int(k_occ.max()),
        "top_hubs": [int(i) for i in hub_indices[np.argsort(-k_occ[hub_indices])[:10]]],
        "k_occurrence": k_occ.tolist(),
    }


def compute_similarity_analysis(preds: np.ndarray, gts: np.ndarray) -> Dict:
    """Analyze positive vs negative pair similarity distributions."""
    sim = preds @ gts.T
    N = sim.shape[0]

    pos_sims = np.array([sim[i, i] for i in range(N)])

    mask = ~np.eye(N, dtype=bool)
    neg_sims = sim[mask]

    neg_sample = neg_sims[np.random.RandomState(42).choice(
        len(neg_sims), min(50000, len(neg_sims)), replace=False
    )]

    separability = float(pos_sims.mean() - neg_sample.mean())
    overlap_threshold = neg_sample.mean() + neg_sample.std()
    overlap_fraction = float(np.mean(pos_sims < overlap_threshold))

    return {
        "positive_sim_mean": round(float(pos_sims.mean()), 4),
        "positive_sim_std": round(float(pos_sims.std()), 4),
        "positive_sim_min": round(float(pos_sims.min()), 4),
        "positive_sim_median": round(float(np.median(pos_sims)), 4),
        "negative_sim_mean": round(float(neg_sample.mean()), 4),
        "negative_sim_std": round(float(neg_sample.std()), 4),
        "negative_sim_max": round(float(neg_sample.max()), 4),
        "separability": round(separability, 4),
        "overlap_fraction": round(overlap_fraction, 4),
        "positive_sims": pos_sims.tolist(),
        "negative_sims_sample": neg_sample[:5000].tolist(),
    }


def compute_retrieval_gap(preds: np.ndarray, gts: np.ndarray, csls_k: int = 10) -> Dict:
    """Compute raw R@1 vs CSLS R@1 and quantify the gap."""
    from fmri2img.eval.embedding_eval import (
        compute_retrieval_metrics,
        compute_retrieval_metrics_csls,
    )

    raw = compute_retrieval_metrics(preds, gts, ks=(1, 5, 10), normalize=False)
    csls = compute_retrieval_metrics_csls(preds, gts, ks=(1, 5, 10),
                                           normalize=False, csls_k=csls_k)

    return {
        "raw_r@1": round(raw["top1_accuracy"], 4),
        "raw_r@5": round(raw["top5_accuracy"], 4),
        "raw_r@10": round(raw["top10_accuracy"], 4),
        "raw_mrr": round(raw.get("mrr", 0), 4),
        "raw_median_rank": raw.get("median_rank", -1),
        "csls_r@1": round(csls["top1_accuracy"], 4),
        "csls_r@5": round(csls["top5_accuracy"], 4),
        "csls_r@10": round(csls["top10_accuracy"], 4),
        "csls_mrr": round(csls.get("mrr", 0), 4),
        "csls_median_rank": csls.get("median_rank", -1),
        "gap_r@1": round(csls["top1_accuracy"] - raw["top1_accuracy"], 4),
        "gap_r@5": round(csls["top5_accuracy"] - raw["top5_accuracy"], 4),
    }


def compute_kappa_analysis(
    preds: np.ndarray, gts: np.ndarray, kappas: np.ndarray, n_bins: int = 5
) -> Dict:
    """Bin predictions by kappa and compute per-bin R@1."""
    kappas_flat = kappas.squeeze()
    sim = preds @ gts.T
    N = sim.shape[0]

    ranks = np.array([
        np.where(np.argsort(-sim[i]) == i)[0][0] + 1 for i in range(N)
    ])
    correct = ranks == 1

    bin_edges = np.percentile(kappas_flat, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-6

    bins = []
    for b in range(n_bins):
        mask = (kappas_flat >= bin_edges[b]) & (kappas_flat < bin_edges[b + 1])
        count = int(mask.sum())
        if count > 0:
            bins.append({
                "bin": b,
                "kappa_range": [round(float(bin_edges[b]), 2), round(float(bin_edges[b + 1]), 2)],
                "count": count,
                "r@1": round(float(correct[mask].mean()), 4),
                "mean_rank": round(float(ranks[mask].mean()), 1),
                "kappa_mean": round(float(kappas_flat[mask].mean()), 2),
            })

    return {
        "n_bins": n_bins,
        "kappa_global_mean": round(float(kappas_flat.mean()), 2),
        "kappa_global_std": round(float(kappas_flat.std()), 2),
        "kappa_global_min": round(float(kappas_flat.min()), 2),
        "kappa_global_max": round(float(kappas_flat.max()), 2),
        "per_bin": bins,
    }


def compute_failure_analysis(preds: np.ndarray, gts: np.ndarray, top_n: int = 20) -> Dict:
    """Identify the hardest samples (lowest cosine sim to ground truth)."""
    pos_sims = np.array([preds[i] @ gts[i] for i in range(len(preds))])
    worst_indices = np.argsort(pos_sims)[:top_n]

    sim = preds @ gts.T
    failures = []
    for idx in worst_indices:
        rank = int(np.where(np.argsort(-sim[idx]) == idx)[0][0]) + 1
        top3_gallery = np.argsort(-sim[idx])[:3].tolist()
        failures.append({
            "sample_idx": int(idx),
            "cosine_sim": round(float(pos_sims[idx]), 4),
            "rank": rank,
            "top3_retrieved": top3_gallery,
        })

    return {
        "top_n": top_n,
        "failures": failures,
    }


def plot_hubness_histogram(k_occ: list, output_path: Path) -> None:
    """Plot k-occurrence distribution."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        k_occ_arr = np.array(k_occ)
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.hist(k_occ_arr, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(k_occ_arr.mean(), color="red", linestyle="--",
                   label=f"Mean: {k_occ_arr.mean():.1f}")
        ax.set_xlabel("k-occurrence (times appearing in top-k)")
        ax.set_ylabel("Gallery items")
        ax.set_title("Hubness: k-Occurrence Distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("Saved hubness histogram to %s", output_path)
    except ImportError:
        logger.warning("matplotlib not available; skipping hubness plot")


def plot_similarity_distribution(pos_sims: list, neg_sims: list, output_path: Path) -> None:
    """Plot positive vs negative similarity distributions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.hist(neg_sims, bins=100, alpha=0.5, label="Negative pairs", density=True)
        ax.hist(pos_sims, bins=50, alpha=0.7, label="Positive pairs", density=True)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.set_title("Positive vs Negative Pair Similarity")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("Saved similarity distribution to %s", output_path)
    except ImportError:
        logger.warning("matplotlib not available; skipping similarity plot")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding diagnostics")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Path to experiment output directory")
    parser.add_argument("--csls-k", type=int, default=10,
                        help="k for CSLS and hubness analysis")
    parser.add_argument("--kappa-bins", type=int, default=5,
                        help="Number of bins for kappa analysis")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error("Results directory not found: %s", results_dir)
        sys.exit(1)

    diag_dir = results_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    preds, gts, kappas = _load_embeddings(results_dir)

    preds = _normalize(preds)
    gts = _normalize(gts)

    report: Dict = {}

    logger.info("Computing hubness analysis...")
    hubness = compute_hubness_analysis(preds, gts, k=args.csls_k)
    k_occ = hubness.pop("k_occurrence")
    report["hubness"] = hubness

    logger.info("Computing similarity analysis...")
    sim_analysis = compute_similarity_analysis(preds, gts)
    pos_sims = sim_analysis.pop("positive_sims")
    neg_sims = sim_analysis.pop("negative_sims_sample")
    report["similarity"] = sim_analysis

    logger.info("Computing retrieval gap (raw vs CSLS)...")
    report["retrieval_gap"] = compute_retrieval_gap(preds, gts, csls_k=args.csls_k)

    if kappas is not None:
        logger.info("Computing kappa analysis...")
        report["kappa"] = compute_kappa_analysis(preds, gts, kappas, n_bins=args.kappa_bins)

    logger.info("Computing failure analysis...")
    report["failures"] = compute_failure_analysis(preds, gts)

    # --- V30: kappa calibration (when kappas available) ---
    _kappa_calib = None
    if kappas is not None:
        try:
            from fmri2img.eval.kappa_diagnostics import compute_kappa_calibration_report
            logger.info("Computing kappa calibration report...")
            metrics_dir = results_dir / "metrics"
            _rich_p_path = metrics_dir / "val_predictions_rich.npy"
            _rich_g_path = metrics_dir / "val_ground_truth_rich.npy"
            _rp = np.load(_rich_p_path) if _rich_p_path.exists() else None
            _rg = np.load(_rich_g_path) if _rich_g_path.exists() else None
            _kappa_calib = compute_kappa_calibration_report(
                preds, gts, kappas, n_bins=args.kappa_bins,
                rich_preds=_rp, rich_gts=_rg,
            )
            report["kappa_calibration"] = _kappa_calib
            _kc_path = diag_dir / "kappa_calibration.json"
            with open(_kc_path, "w") as f:
                json.dump(_kappa_calib, f, indent=2)
            logger.info("Saved kappa calibration to %s", _kc_path)
        except Exception as e:
            logger.warning("Kappa calibration failed: %s", e)

    # --- V30: two-stage retrieval (prefer rerank head, fall back to rich/regression) ---
    _two_stage = None
    metrics_dir = results_dir / "metrics"
    _compact_p = metrics_dir / "val_predictions_compact.npy"
    _compact_g = metrics_dir / "val_ground_truth_compact.npy"
    # V30d+: prefer rerank head outputs over regression head
    _rerank_p = metrics_dir / "val_predictions_rerank.npy"
    _rerank_g = metrics_dir / "val_ground_truth_rerank.npy"
    _rich_p = metrics_dir / "val_predictions_rich.npy"
    _rich_g = metrics_dir / "val_ground_truth_rich.npy"
    if _compact_p.exists() and _compact_g.exists():
        # Choose rerank files if available, else fall back to rich
        if _rerank_p.exists() and _rerank_g.exists():
            _stage2_p, _stage2_g = _rerank_p, _rerank_g
            _stage2_source = "rerank_head"
        elif _rich_p.exists() and _rich_g.exists():
            _stage2_p, _stage2_g = _rich_p, _rich_g
            _stage2_source = "regression_head"
        else:
            _stage2_p, _stage2_g = None, None
            _stage2_source = None

        if _stage2_p is not None:
            try:
                from fmri2img.eval.two_stage_retrieval import two_stage_metrics
                logger.info("Computing two-stage retrieval metrics (source: %s)...", _stage2_source)
                _cp = np.load(_compact_p)
                _cg = np.load(_compact_g)
                _rp = np.load(_stage2_p)
                _rg = np.load(_stage2_g)
                _two_stage = two_stage_metrics(_cp, _cg, _rp, _rg,
                                               shortlist_k=100, ks=(1, 5, 10))
                _two_stage["stage2_source"] = _stage2_source
                report["two_stage_retrieval"] = _two_stage
                _ts_path = diag_dir / "two_stage_retrieval.json"
                with open(_ts_path, "w") as f:
                    json.dump(_two_stage, f, indent=2)
                logger.info("Saved two-stage retrieval report to %s", _ts_path)
            except Exception as e:
                logger.warning("Two-stage retrieval failed: %s", e)

    report_path = diag_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved diagnostic report to %s", report_path)

    plot_hubness_histogram(k_occ, diag_dir / "hubness_histogram.png")
    plot_similarity_distribution(pos_sims, neg_sims, diag_dir / "similarity_distribution.png")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"  Gallery size:      {hubness['gallery_size']}")
    print(f"  Raw R@1:           {report['retrieval_gap']['raw_r@1']:.1%}")
    print(f"  CSLS R@1:          {report['retrieval_gap']['csls_r@1']:.1%}")
    print(f"  Hubness gap:       {report['retrieval_gap']['gap_r@1']:.1%}")
    print(f"  Skewness (k-occ):  {hubness['skewness']:.2f}")
    print(f"  Hub fraction:      {hubness['hub_fraction']:.1%}")
    print(f"  Antihub fraction:  {hubness['antihub_fraction']:.1%}")
    print(f"  Pos sim (mean):    {sim_analysis['positive_sim_mean']:.4f}")
    print(f"  Neg sim (mean):    {sim_analysis['negative_sim_mean']:.4f}")
    print(f"  Separability:      {sim_analysis['separability']:.4f}")
    if kappas is not None:
        ka = report["kappa"]
        print(f"  Kappa (mean/std):  {ka['kappa_global_mean']:.1f} +/- {ka['kappa_global_std']:.1f}")
    if _kappa_calib is not None:
        print(f"  Kappa-rank rho:    {_kappa_calib['spearman_kappa_rank']['rho']:.3f}")
        print(f"  Kappa-correct rho: {_kappa_calib['spearman_kappa_correct']['rho']:.3f}")
    if _two_stage is not None:
        _ts_r = _two_stage.get("reranked", {})
        _ts_c = _two_stage.get("compact_raw", {})
        _ts_rich = _two_stage.get("rich_only", {})
        _ts_oracle = _two_stage.get("oracle_rerank", {})
        _ts_diag = _two_stage.get("rich_diagnostics", {})
        print(f"  Compact raw R@1:   {_ts_c.get('compact_r@1', 0):.1%}")
        print(f"  Reranked R@1:      {_ts_r.get('reranked_r@1', 0):.1%}")
        print(f"  Rich-only R@1:     {_ts_rich.get('rich_r@1', 0):.1%}")
        print(f"  Oracle rerank R@1: {_ts_oracle.get('oracle_r@1', 0):.1%}")
        print(f"  Rerank gain (pp):  {_two_stage.get('rerank_gain_over_compact_raw', 0)*100:.1f}")
        if _ts_diag:
            print(f"  Rich separability: {_ts_diag.get('separability', 0):.2f}")
            print(f"  Rich pos/neg cos:  {_ts_diag.get('pos_cosine_mean', 0):.4f} / "
                  f"{_ts_diag.get('neg_cosine_mean', 0):.4f}")
    print(f"\n  Report: {report_path}")
    print(f"  Plots:  {diag_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
