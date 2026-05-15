#!/usr/bin/env python3
"""CLI entry point for the Neural Manifold Alignment analysis framework.

Usage
-----
  # Full analysis from a config file
  python scripts/run_manifold_analysis.py --config configs/manifold_analysis.yaml

  # Auto-discover artifacts from an experiment directory
  python scripts/run_manifold_analysis.py --experiment-dir experimental_results/N1v30d/subj01

  # Dry run — validate artifacts and list runnable modules
  python scripts/run_manifold_analysis.py --config configs/manifold_analysis.yaml --dry-run

  # Run only specific modules
  python scripts/run_manifold_analysis.py --config configs/manifold_analysis.yaml \\
      --modules retrieval hubness rsa
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.manifold.claim_tests import evaluate_all_claims, save_claim_tests
from analysis.manifold.config import ManifoldAnalysisConfig, load_config
from analysis.manifold.counterfactual import (
    compute_counterfactual_edits,
    save_counterfactual_results,
)
from analysis.manifold.data_loading import AnalysisData, discover_artifacts, load_analysis_data
from analysis.manifold.figures import setup_figure_style
from analysis.manifold.hubness import compute_hub_counts, compute_hubness_metrics, save_hubness_results
from analysis.manifold.interpolation import run_interpolation, save_interpolation_results
from analysis.manifold.manifold_density import run_manifold_density, save_manifold_density_results
from analysis.manifold.neighborhood import (
    compute_neighborhood_overlap,
    compute_trustworthiness_continuity,
    save_neighborhood_results,
)
from analysis.manifold.nmas import compute_nmas, save_nmas_results
from analysis.manifold.reliability import run_reliability_analysis, save_reliability_results
from analysis.manifold.repeat_stability import run_repeat_stability, save_repeat_stability_results
from analysis.manifold.report import generate_report
from analysis.manifold.retrieval_metrics import (
    compute_retrieval_comparison,
    save_retrieval_results,
)
from analysis.manifold.roi_axes import run_roi_axes, save_roi_axes_results
from analysis.manifold.rsa import compute_rsa, save_rsa_results
from analysis.manifold.semantic_axes import run_semantic_axes, save_semantic_axes_results
from analysis.manifold.semantic_probes import run_semantic_probes, save_semantic_probe_results
from analysis.manifold.uncertainty import run_uncertainty_analysis, save_uncertainty_results
from analysis.manifold.vector_field import run_vector_field, save_vector_field_results

logger = logging.getLogger("manifold_analysis")


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_output_dir(config: ManifoldAnalysisConfig) -> Path:
    od = config.output.get("output_dir")
    if od:
        return Path(od)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _REPO_ROOT / "reports" / "manifold_analysis" / ts


def _resolve_target_indices(data: AnalysisData, config: ManifoldAnalysisConfig) -> Any:
    """Return gallery indices for retrieval.

    Diagonal fallback is reserved for explicit diagnostic runs. Production
    manifold reports must provide valid target-gallery mapping.
    """
    if data.target_gallery_indices is not None:
        return data.target_gallery_indices
    n = data.n_samples
    g = data.n_gallery
    allow_fallback = bool(config.analysis.get("allow_diagonal_fallback", False))
    if n == g and allow_fallback:
        logger.warning("No target_gallery_indices; using explicit diagonal fallback")
        return np.arange(n)
    raise ValueError(
        f"No target_gallery_indices available for N={n}, gallery={g}. "
        "Provide target_ids_path and gallery ids, use gallery_mode='target_ids', "
        "or set allow_diagonal_fallback=true only for diagnostics."
    )


def run_dry(config: ManifoldAnalysisConfig, data: AnalysisData) -> None:
    """Validate artifacts and list which modules can run."""
    print("\n=== DRY RUN ===\n")
    print(f"Samples (N): {data.n_samples}")
    print(f"Gallery (G): {data.n_gallery}")
    print(f"Embedding dim: {data.embedding_dim}")
    print(f"\nData availability:")
    for k, v in data.availability.items():
        print(f"  {k}: {v}")
    print(f"\nModules enabled in config:")
    for mod, enabled in config.modules.items():
        can_run = True
        reason = ""
        if mod == "repeat_stability" and not data.availability.get("has_per_trial"):
            can_run = False
            reason = "per-trial predictions missing"
        if mod == "uncertainty" and not data.availability.get("has_kappa"):
            can_run = False
            reason = "kappa values missing"
        if mod == "roi_axes":
            can_run = False
            reason = "requires model inference capability"
        status = "READY" if enabled and can_run else "SKIP"
        line = f"  {mod}: {status}"
        if not can_run and enabled:
            line += f" ({reason})"
        if not enabled:
            line += " (disabled)"
        print(line)
    print()


def run_analysis(
    config: ManifoldAnalysisConfig,
    data: AnalysisData,
    output_dir: Path,
    module_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Orchestrate all enabled analysis modules."""
    setup_figure_style()
    import numpy as np

    save_fmts = config.output.get("save_formats", ["png"])
    all_metrics: Dict[str, Any] = {}
    all_figures: List[str] = []
    ks = config.analysis.get("top_k", [1, 5, 10])
    csls_k = config.analysis.get("csls_k", 10)
    seed = config.analysis.get("seed", 42)
    bootstrap_n = int(config.analysis.get("bootstrap_n", 500) or 0)
    np.random.seed(seed)

    def _enabled(mod: str) -> bool:
        if module_filter:
            return mod in module_filter
        return config.module_enabled(mod)

    target_indices = _resolve_target_indices(data, config)
    correct: Optional[np.ndarray] = None
    cos_ranks: Optional[np.ndarray] = None
    csl_ranks: Optional[np.ndarray] = None
    csls_margins: Optional[np.ndarray] = None
    csl_entropy: Optional[np.ndarray] = None
    cos_scores: Optional[np.ndarray] = None
    density_arr: Optional[np.ndarray] = None

    # ---- 1. Retrieval ----
    if _enabled("retrieval"):
        logger.info("=== Module: Retrieval ===")
        ret = compute_retrieval_comparison(
            data.z_pred, data.gallery, target_indices, ks, csls_k,
            n_bootstrap=bootstrap_n, seed=seed,
        )
        res = save_retrieval_results(ret, output_dir, save_fmts)
        for prefix, sub in [("cosine_", ret["cosine"]["metrics"]),
                            ("csls_", ret["csls"]["metrics"])]:
            for k, v in sub.items():
                all_metrics[f"{prefix}{k}"] = v
        all_figures.extend(res["figures"])
        correct = ret["csls"]["per_trial"]["correct"].values
        cos_ranks = ret["cosine"]["ranks"]
        csl_ranks = ret["csls"]["ranks"]
        csls_margins = ret["csls"]["per_trial"]["csls_margin"].values
        csl_entropy = ret["csls"]["per_trial"]["entropy"].values
        cos_scores = ret["cosine"]["cosine_scores"]
    else:
        correct = np.zeros(data.n_samples, dtype=bool)

    # ---- 2. Hubness ----
    if _enabled("hubness") and cos_scores is not None:
        logger.info("=== Module: Hubness ===")
        cos_counts = compute_hub_counts(cos_scores, k=min(10, data.n_gallery))
        from analysis.manifold.utils import csls_scores as csls_fn
        csl_score_mat = csls_fn(data.z_pred, data.gallery, k=csls_k,
                                cosine_scores=cos_scores)
        csl_counts = compute_hub_counts(csl_score_mat, k=min(10, data.n_gallery))
        hub_metrics = compute_hubness_metrics(cos_counts, csl_counts)
        res = save_hubness_results(cos_counts, csl_counts, hub_metrics, output_dir, save_fmts)
        all_metrics.update(hub_metrics)
        all_figures.extend(res["figures"])

    # ---- 3. RSA ----
    if _enabled("rsa"):
        logger.info("=== Module: RSA ===")
        rsa_results = compute_rsa(
            data.z_pred, data.z_target,
            max_samples=config.analysis.get("rsa_max_samples", 1500),
            seed=seed,
            n_bootstrap=bootstrap_n,
        )
        res = save_rsa_results(rsa_results, output_dir, save_fmts)
        for k, v in res["metrics"].items():
            all_metrics[f"rsa_{k}"] = v
        all_figures.extend(res["figures"])

    # ---- 4. Neighbourhood ----
    if _enabled("neighborhood"):
        logger.info("=== Module: Neighbourhood ===")
        nhood_ks = config.analysis.get("neighborhood_ks", [5, 10, 20, 50, 100])
        nhood = compute_neighborhood_overlap(
            data.z_pred, data.z_target, data.gallery, nhood_ks,
            n_bootstrap=bootstrap_n, seed=seed,
        )
        tc = compute_trustworthiness_continuity(data.z_pred, data.z_target)
        nhood.update(tc)
        res = save_neighborhood_results(
            nhood, correct, csl_ranks if csl_ranks is not None else np.zeros(data.n_samples),
            output_dir, save_fmts,
        )
        all_metrics.update(res["metrics"])
        all_figures.extend(res["figures"])

    # ---- 5. Repeat stability ----
    if _enabled("repeat_stability"):
        logger.info("=== Module: Repeat Stability ===")
        rep = run_repeat_stability(data.per_trial_z_pred, data.per_trial_nsd_ids, seed)
        res = save_repeat_stability_results(rep, output_dir, save_fmts)
        all_metrics["repeat_stability_status"] = rep.get("status", "unavailable")
        for k, v in res["metrics"].items():
            if k != "status":
                all_metrics[k] = v
        all_figures.extend(res["figures"])

    # ---- 6. Uncertainty ----
    if _enabled("uncertainty"):
        logger.info("=== Module: Uncertainty ===")
        unc = run_uncertainty_analysis(
            data.kappa, correct,
            margins=csls_margins,
            entropy=csl_entropy,
            n_bins=config.analysis.get("calibration_n_bins", 10),
        )
        res = save_uncertainty_results(unc, data.kappa, correct, csls_margins,
                                       output_dir, save_fmts)
        all_metrics["uncertainty_status"] = unc.get("status", "unavailable")
        for k, v in res["metrics"].items():
            if k not in ("status", "calibration"):
                all_metrics[k] = v
        all_figures.extend(res["figures"])

    # ---- 7. Manifold density ----
    if _enabled("manifold_density"):
        logger.info("=== Module: Manifold Density ===")
        md = run_manifold_density(
            data.z_pred, data.gallery, correct,
            kappa=data.kappa,
            k=config.analysis.get("manifold_density_k", 20),
        )
        density_arr = md.get("_density")
        res = save_manifold_density_results(md, correct, data.kappa, output_dir, save_fmts)
        all_metrics.update(res["metrics"])
        all_figures.extend(res["figures"])

    # ---- 8. Reliability ----
    if _enabled("reliability"):
        logger.info("=== Module: Reliability ===")
        hub_scores = None
        if "cosine_skewness" in all_metrics and cos_scores is not None:
            from analysis.manifold.hubness import compute_hub_counts as _hc
            hc = _hc(cos_scores, k=min(10, data.n_gallery))
            hub_scores = hc[target_indices] if target_indices is not None else None

        rel = run_reliability_analysis(
            correct,
            kappa=data.kappa,
            csls_margin=csls_margins,
            manifold_density=density_arr,
            topk_entropy=csl_entropy,
            hubness_score=hub_scores,
            weights=config.reliability_weights,
            coverages=config.analysis.get("coverage_levels",
                                          [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
            n_bootstrap=bootstrap_n,
            seed=seed,
        )
        res = save_reliability_results(rel, correct, output_dir, save_fmts)
        all_metrics.update({k: v for k, v in res["metrics"].items() if k != "status"})
        all_figures.extend(res["figures"])

    # ---- 9. Semantic probes ----
    if _enabled("semantic_probes"):
        logger.info("=== Module: Semantic Probes ===")
        sp_cfg = config.semantic_probes
        sp = run_semantic_probes(
            data.z_pred, data.z_target,
            text_embeddings=data.text_embeddings,
            concepts=data.text_concepts or config.text_probes.get("concepts"),
            temperature=sp_cfg.get("temperature", 0.07),
            agreement_ks=sp_cfg.get("top_k", [1, 3, 5, 10]),
            retrieval_correct=correct,
            max_examples=sp_cfg.get("max_examples", 12),
            text_cache_metadata=data.text_probe_metadata,
            text_embeddings_path=config.artifacts.get("text_embeddings_path"),
        )
        res = save_semantic_probe_results(sp, output_dir, save_fmts)
        all_metrics["semantic_probes_status"] = sp.get("status", "unavailable")
        if sp.get("reason"):
            all_metrics["semantic_probes_reason"] = sp.get("reason")
        if sp.get("text_embeddings_path"):
            all_metrics["semantic_probes_text_embeddings_path"] = sp.get("text_embeddings_path")
        if sp.get("text_cache"):
            all_metrics["semantic_probes_text_cache"] = sp.get("text_cache")
        all_metrics.update({k: v for k, v in res["metrics"].items()
                           if isinstance(v, (int, float)) and k != "status"})
        all_figures.extend(res["figures"])

    # ---- 10. Semantic axes ----
    axes_arr = data.semantic_axes
    axis_names = data.semantic_axis_names
    if _enabled("semantic_axes"):
        logger.info("=== Module: Semantic Axes ===")
        sax_cfg = config.semantic_axes
        sa = run_semantic_axes(
            data.z_pred, data.z_target,
            axes=axes_arr,
            axis_names=axis_names,
            axis_definitions=sax_cfg.get("axes"),
            text_embeddings=data.text_embeddings,
            text_concepts=data.text_concepts,
            retrieval_correct=correct,
            max_examples=sax_cfg.get("max_examples", 12),
        )
        computed_axes = sa.get("_axes")
        if sa.get("status") == "computed" and computed_axes is not None:
            axes_arr = computed_axes
            axis_names = sa.get("axis_names", axis_names)
        res = save_semantic_axes_results(sa, output_dir, save_fmts)
        all_metrics["semantic_axes_status"] = res["metrics"].get("status", "unavailable")
        if res["metrics"].get("reason"):
            all_metrics["semantic_axes_reason"] = res["metrics"].get("reason")
        all_metrics.update({
            k: v
            for k, v in res["metrics"].items()
            if isinstance(v, (int, float, str)) and k != "status"
        })
        all_figures.extend(res["figures"])

    # ---- 11. Vector field ----
    if _enabled("vector_field"):
        logger.info("=== Module: Vector Field ===")
        vf = run_vector_field(data.z_pred, data.z_target, correct)
        res = save_vector_field_results(vf, correct, output_dir, save_fmts)
        all_metrics.update(res["metrics"])
        all_figures.extend(res["figures"])

    # ---- 12. Counterfactuals ----
    if _enabled("counterfactuals"):
        logger.info("=== Module: Counterfactuals ===")
        cfg_cf = config.counterfactual
        confidence = csls_margins
        cf = compute_counterfactual_edits(
            data.z_pred,
            axes_arr,
            axis_names,
            data.gallery,
            alphas=cfg_cf.get("alphas", [-2, -1, -0.5, 0, 0.5, 1, 2]),
            text_embeddings=data.text_embeddings,
            text_concepts=data.text_concepts,
            z_target=data.z_target,
            target_indices=target_indices,
            correct=correct,
            confidence=confidence,
            max_cases=cfg_cf.get("max_cases", cfg_cf.get("n_cases", 24)),
            seed=seed,
        )
        res = save_counterfactual_results(cf, output_dir, save_fmts)
        all_metrics["counterfactual_status"] = res["metrics"].get("status", "unavailable")
        if res["metrics"].get("reason"):
            all_metrics["counterfactual_reason"] = res["metrics"].get("reason")
        all_metrics.update({
            k: v
            for k, v in res["metrics"].items()
            if isinstance(v, (int, float, str)) and k != "status"
        })
        all_figures.extend(res["figures"])

    # ---- 13. Interpolation ----
    if _enabled("interpolation"):
        logger.info("=== Module: Interpolation ===")
        cfg_in = config.interpolation
        interp = run_interpolation(
            data.z_pred, data.gallery, correct,
            text_embeddings=data.text_embeddings,
            text_concepts=data.text_concepts,
            n_steps=cfg_in.get("steps", cfg_in.get("n_steps", 21)),
            n_pairs=cfg_in.get("max_pairs", cfg_in.get("n_pairs", 20)),
            seed=seed,
            pair_strategy=cfg_in.get("pair_strategy", "mixed"),
        )
        res = save_interpolation_results(interp, output_dir, save_fmts)
        all_metrics["interpolation_status"] = res["metrics"].get("status", "unavailable")
        if res["metrics"].get("reason"):
            all_metrics["interpolation_reason"] = res["metrics"].get("reason")
        all_metrics.update({
            k: v
            for k, v in res["metrics"].items()
            if isinstance(v, (int, float, str)) and k != "status"
        })
        all_figures.extend(res["figures"])

    # ---- 14. ROI axes ----
    if _enabled("roi_axes"):
        logger.info("=== Module: ROI Axes ===")
        roi = run_roi_axes()
        res = save_roi_axes_results(roi, output_dir, save_fmts)
        all_metrics["roi_axis_status"] = roi.get("status", "unavailable")
        all_figures.extend(res["figures"])

    # ---- 15. NMAS ----
    if _enabled("nmas"):
        logger.info("=== Module: NMAS ===")
        nmas_components: Dict[str, Optional[float]] = {
            "retrieval": all_metrics.get("csls_R@1"),
            "neighborhood": all_metrics.get("overlap@10_mean"),
            "rsa": all_metrics.get("rsa_spearman_rho"),
            "repeat_stability": all_metrics.get("same_different_auroc"),
            "on_manifold": all_metrics.get("density_mean"),
            "calibration": 1.0 - all_metrics.get("ece", 1.0) if all_metrics.get("ece") is not None else None,
        }
        nmas_result = compute_nmas(nmas_components)
        res = save_nmas_results(nmas_result, output_dir, save_fmts)
        all_metrics["nmas"] = nmas_result["nmas"]
        all_metrics["nmas_components"] = nmas_result["components"]
        all_figures.extend(res["figures"])

    # ---- 16. Claim tests ----
    logger.info("=== Evaluating Scientific Claims ===")
    claims = evaluate_all_claims(all_metrics)
    save_claim_tests(claims, output_dir / "claim_tests.json")

    # ---- 17. Report ----
    logger.info("=== Generating Report ===")
    generate_report(
        output_dir, all_metrics, claims, all_figures,
        config_summary=config.raw,
        experiment_id=data.experiment_id,
        subject=data.subject,
    )

    supported = sum(1 for c in claims if c.status == "supported")
    partial = sum(1 for c in claims if c.status == "partially_supported")
    unavail = sum(1 for c in claims if c.status == "unavailable")
    logger.info(
        "Analysis complete. Claims: %d supported, %d partial, %d unavailable. "
        "Report: %s/summary.md",
        supported, partial, unavail, output_dir,
    )

    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neural Manifold Alignment analysis framework"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to manifold_analysis.yaml")
    parser.add_argument("--experiment-dir", type=str, default=None,
                        help="Auto-discover artifacts from experiment directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate artifacts and list runnable modules")
    parser.add_argument("--modules", nargs="+", default=None,
                        help="Run only these modules")
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    _setup_logging(args.log_level)

    overrides: Dict[str, Any] = {}
    if args.experiment_dir:
        discovered = discover_artifacts(args.experiment_dir)
        overrides["artifacts"] = discovered
    if args.output_dir:
        overrides.setdefault("output", {})["output_dir"] = args.output_dir
    if args.subject:
        overrides.setdefault("artifacts", {})["subject"] = args.subject
    if args.seed:
        overrides.setdefault("analysis", {})["seed"] = args.seed

    config = load_config(args.config, overrides)
    output_dir = _build_output_dir(config)

    try:
        data = load_analysis_data(config)
    except FileNotFoundError as e:
        logger.error("Cannot load data: %s", e)
        if args.dry_run:
            print(f"\nERROR: {e}")
        sys.exit(1)

    if args.dry_run:
        run_dry(config, data)
        return

    run_analysis(config, data, output_dir, module_filter=args.modules)


if __name__ == "__main__":
    main()
