"""Markdown report and combined metrics generator."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from analysis.manifold.claim_tests import ClaimTest

logger = logging.getLogger(__name__)


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def generate_report(
    output_dir: Path,
    all_metrics: Dict[str, Any],
    claims: List[ClaimTest],
    figures: List[str],
    config_summary: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    subject: Optional[str] = None,
) -> Path:
    """Generate ``summary.md``, ``metrics.json``, and index files.

    Args:
        output_dir: Report root directory.
        all_metrics: Combined metrics from all modules.
        claims: Evaluated claim tests.
        figures: List of figure paths.
        config_summary: Config dict for provenance.
        experiment_id: Identifier for this experiment.
        subject: NSD subject identifier.

    Returns:
        Path to the generated ``summary.md``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    with open(output_dir / "figures_index.json", "w") as f:
        json.dump({"figures": figures}, f, indent=2)

    if config_summary is not None:
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_summary, f, indent=2, default=str)

    lines: List[str] = []
    lines.append("# Neural Manifold Alignment Analysis Report\n")
    lines.append(f"**Generated:** {now}  ")
    if experiment_id:
        lines.append(f"**Experiment:** {experiment_id}  ")
    if subject:
        lines.append(f"**Subject:** {subject}  ")
    lines.append("")

    # Executive summary
    lines.append("## Executive Summary\n")
    cos_r1 = all_metrics.get("cosine_R@1")
    csl_r1 = all_metrics.get("csls_R@1")
    nmas = all_metrics.get("nmas")
    rsa = all_metrics.get("rsa_spearman_rho")
    lines.append(
        f"This report evaluates a neural decoder mapping fMRI activity into "
        f"CLIP ViT-L/14 embedding space.  "
        f"Cosine R@1 = {_fmt(cos_r1)}, CSLS R@1 = {_fmt(csl_r1)}.  "
        f"RSA Spearman rho = {_fmt(rsa)}.  "
        f"NMAS (aggregate) = {_fmt(nmas)}.\n"
    )

    # Retrieval
    lines.append("## Retrieval Metrics\n")
    lines.append("*Ranks are reported 1-indexed (rank 1 means the correct image was top-ranked).*\n")
    lines.append("| Metric | Cosine | CSLS |")
    lines.append("|--------|--------|------|")
    for k_suffix in ["R@1", "R@5", "R@10", "median_rank", "mrr"]:
        cv = all_metrics.get(f"cosine_{k_suffix}")
        sv = all_metrics.get(f"csls_{k_suffix}")
        lines.append(f"| {k_suffix} | {_fmt(cv)} | {_fmt(sv)} |")
    lines.append("")

    # Hubness
    if all_metrics.get("cosine_skewness") is not None:
        lines.append("## Hubness Analysis\n")
        lines.append("| Metric | Cosine | CSLS |")
        lines.append("|--------|--------|------|")
        for m in ["skewness", "gini", "max_hub"]:
            cv = all_metrics.get(f"cosine_{m}")
            sv = all_metrics.get(f"csls_{m}")
            lines.append(f"| {m} | {_fmt(cv)} | {_fmt(sv)} |")
        lines.append("")

    # RSA
    if rsa is not None:
        lines.append("## Representational Similarity Analysis\n")
        lines.append(f"- Spearman rho: {_fmt(rsa)}")
        lines.append(f"- Pearson r: {_fmt(all_metrics.get('rsa_pearson_rho'))}")
        lines.append(f"- N samples: {_fmt(all_metrics.get('rsa_n_samples'))}\n")

    # Neighbourhood
    ov10 = all_metrics.get("overlap@10_mean")
    if ov10 is not None:
        lines.append("## Neighbourhood Preservation\n")
        for k in [5, 10, 20, 50, 100]:
            v = all_metrics.get(f"overlap@{k}_mean")
            if v is not None:
                base = all_metrics.get(f"overlap@{k}_random_baseline")
                lift = all_metrics.get(f"overlap@{k}_lift_over_random")
                lines.append(
                    f"- Overlap@{k}: {_fmt(v)} "
                    f"(random baseline {_fmt(base)}, lift {_fmt(lift)}x)"
                )
        lines.append("")

    # Repeat stability
    rs_status = all_metrics.get("repeat_stability_status")
    if rs_status == "computed":
        lines.append("## Repeat Stability\n")
        lines.append(f"- Within-image mean similarity: {_fmt(all_metrics.get('within_mean'))}")
        lines.append(f"- Between-image mean similarity: {_fmt(all_metrics.get('between_mean'))}")
        lines.append(f"- Same/different AUROC: {_fmt(all_metrics.get('same_different_auroc'))}")
        lines.append(f"- Stability ratio: {_fmt(all_metrics.get('stability_ratio'))}\n")
    elif rs_status == "unavailable":
        lines.append("## Repeat Stability\n")
        lines.append("*Skipped — per-trial predictions not available.*\n")

    # Uncertainty
    u_status = all_metrics.get("uncertainty_status")
    if u_status == "computed":
        lines.append("## Uncertainty (vMF Kappa)\n")
        lines.append(f"- Kappa mean: {_fmt(all_metrics.get('kappa_mean'))}")
        lines.append(f"- Kappa AUROC: {_fmt(all_metrics.get('kappa_auroc'))}")
        lines.append(f"- ECE: {_fmt(all_metrics.get('ece'))}")
        lines.append(f"- Point-biserial r: {_fmt(all_metrics.get('kappa_correct_rpb'))}\n")
    elif u_status == "unavailable":
        lines.append("## Uncertainty\n")
        lines.append("*Skipped — kappa values not available.*\n")

    # Manifold density
    dm = all_metrics.get("density_mean")
    if dm is not None:
        lines.append("## Manifold Density\n")
        lines.append(f"- Mean density: {_fmt(dm)}")
        lines.append(f"- Correct mean: {_fmt(all_metrics.get('density_correct_mean'))}")
        lines.append(f"- Incorrect mean: {_fmt(all_metrics.get('density_incorrect_mean'))}\n")

    # Reliability
    if all_metrics.get("aurc") is not None:
        lines.append("## Reliability-Aware Selective Decoding\n")
        lines.append(f"- AURC: {_fmt(all_metrics.get('aurc'))}")
        lines.append(f"- Components used: {all_metrics.get('used_components', '—')}\n")

    # Semantic probes
    sp_status = all_metrics.get("semantic_probes_status")
    if sp_status == "computed":
        lines.append("## Neural Semantic Probe: Language-Space Interpretation\n")
        lines.append(
            "This probe interprets decoded visual-stimulus-related fMRI embeddings "
            "relative to CLIP text concepts. It does not decode thoughts, dreams, "
            "or consciousness.\n"
        )
        cache = all_metrics.get("semantic_probes_text_cache", {}) or {}
        cache_path = all_metrics.get("semantic_probes_text_embeddings_path")
        model_name = cache.get("model_name") or cache.get("model")
        pretrained_name = cache.get("pretrained_name") or cache.get("pretrained")
        lines.append(f"- Text embedding cache: `{cache_path or cache.get('path', '—')}`")
        lines.append(f"- Text model: {_fmt(model_name)} / {_fmt(pretrained_name)}")
        lines.append(f"- Number of concepts: {_fmt(all_metrics.get('n_concepts'))}")
        lines.append(f"- Temperature: {_fmt(all_metrics.get('temperature'))}")
        lines.append(f"- Top-1 concept agreement: {_fmt(all_metrics.get('top1_agreement'))}")
        lines.append(f"- Agreement@3: {_fmt(all_metrics.get('agreement@3'))}")
        lines.append(f"- Agreement@5: {_fmt(all_metrics.get('agreement@5'))}")
        lines.append(f"- Agreement@10: {_fmt(all_metrics.get('agreement@10'))}")
        lines.append(f"- Score-vector Spearman: {_fmt(all_metrics.get('score_vector_spearman'))}")
        lines.append(f"- Score-vector Pearson: {_fmt(all_metrics.get('score_vector_pearson'))}")
        lines.append(f"- Mean JS divergence: {_fmt(all_metrics.get('js_divergence_mean'))}")
        lines.append(f"- Mean KL divergence: {_fmt(all_metrics.get('kl_divergence_mean'))}")
        lines.append(f"- Predicted entropy mean: {_fmt(all_metrics.get('pred_entropy_mean'))}")
        lines.append(f"- Target entropy mean: {_fmt(all_metrics.get('target_entropy_mean'))}")
        c09 = next((c for c in claims if c.claim_id == "C09"), None)
        if c09 is not None:
            lines.append(f"- C09 status: **{c09.status}** — {c09.explanation}")
        examples_path = output_dir / "semantic_probe_examples.json"
        if examples_path.exists():
            try:
                examples = json.loads(examples_path.read_text())[:5]
                if examples:
                    lines.append("\nExample decoded-vs-target top concepts:")
                    for ex in examples:
                        pred = ", ".join(
                            c["concept"] for c in ex.get("predicted_top_concepts", [])[:3]
                        )
                        target = ", ".join(
                            c["concept"] for c in ex.get("target_top_concepts", [])[:3]
                        )
                        rank = ex.get("target_top_concept_rank_in_prediction", "—")
                        lines.append(
                            f"- sample {ex.get('sample_index')}: decoded [{pred}] vs "
                            f"target [{target}], target top-concept rank {rank}"
                        )
            except Exception as exc:  # pragma: no cover - report should never fail on examples
                logger.warning("Could not read semantic probe examples: %s", exc)
        lines.append("")
    elif sp_status == "unavailable":
        lines.append("## Neural Semantic Probe: Language-Space Interpretation\n")
        lines.append(
            "*Skipped — CLIP text embedding cache unavailable. Build a real cache "
            "with `scripts/build/build_text_probe_embeddings.py` and set "
            "`artifacts.text_embeddings_path`. No text embeddings were fabricated.*\n"
        )
        reason = all_metrics.get("semantic_probes_reason")
        if reason:
            lines.append(f"- Reason: {reason}\n")

    # Semantic axes
    ax_status = all_metrics.get("semantic_axes_status")
    mp = all_metrics.get("mean_preservation")
    if ax_status == "computed" or mp is not None:
        lines.append("## Semantic Axis Analysis: Interpretable Concept Directions in CLIP Space\n")
        lines.append(
            "Semantic axes are CLIP text-space directions such as "
            "`text('animal') - text('object')`. They probe whether decoded "
            "visual-stimulus-related fMRI embeddings preserve interpretable "
            "concept dimensions. These axes are CLIP-space constructs, not "
            "literal neural concepts.\n"
        )
        lines.append(f"- Axes built: {_fmt(all_metrics.get('n_axes_built'))}")
        lines.append(f"- Axes unavailable: {_fmt(all_metrics.get('n_axes_unavailable'))}")
        lines.append(f"- Mean axis Pearson: {_fmt(all_metrics.get('mean_axis_pearson'))}")
        lines.append(f"- Mean axis Spearman: {_fmt(all_metrics.get('mean_axis_spearman'))}")
        lines.append(f"- Mean absolute axis error: {_fmt(all_metrics.get('mean_axis_mae'))}")
        lines.append(f"- Mean sign agreement: {_fmt(all_metrics.get('mean_sign_agreement'))}")
        lines.append(f"- Mean preservation: {_fmt(mp)}")
        lines.append(f"- Best preserved axis: {_fmt(all_metrics.get('best_axis'))}")
        lines.append(f"- Weakest axis: {_fmt(all_metrics.get('weakest_axis'))}")
        c10 = next((c for c in claims if c.claim_id == "C10"), None)
        if c10 is not None:
            lines.append(f"- C10 status: **{c10.status}** — {c10.explanation}")
        axis_summary = output_dir / "semantic_axes_axis_summary.csv"
        if axis_summary.exists():
            try:
                import csv

                rows = list(csv.DictReader(axis_summary.open()))[:10]
                if rows:
                    lines.append("\nAxis summary:")
                    lines.append("| Axis | Pearson | Spearman | MAE | Sign agreement | Preservation |")
                    lines.append("|---|---:|---:|---:|---:|---:|")
                    for row in rows:
                        lines.append(
                            f"| {row.get('axis')} | {_fmt(float(row.get('pearson_r', 'nan')))} "
                            f"| {_fmt(float(row.get('spearman_r', 'nan')))} "
                            f"| {_fmt(float(row.get('mae', 'nan')))} "
                            f"| {_fmt(float(row.get('sign_agreement', 'nan')))} "
                            f"| {_fmt(float(row.get('preservation_score', 'nan')))} |"
                        )
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not read semantic axes summary: %s", exc)
        lines.append("")
    elif ax_status == "unavailable":
        lines.append("## Semantic Axis Analysis: Interpretable Concept Directions in CLIP Space\n")
        lines.append(
            "*Skipped — semantic axes require a real CLIP text embedding cache or "
            "precomputed axis vectors. No axes were fabricated.*\n"
        )
        reason = all_metrics.get("semantic_axes_reason")
        if reason:
            lines.append(f"- Reason: {reason}\n")

    # Counterfactual latent editing
    cf_status = all_metrics.get("counterfactual_status")
    if cf_status == "computed":
        lines.append("## Counterfactual Latent Editing\n")
        lines.append(
            "This module edits decoded embeddings along CLIP semantic axes and "
            "observes retrieval/text-probe changes. It is not neural manipulation "
            "or causal brain control; it is a counterfactual analysis of decoded "
            "CLIP latent vectors.\n"
        )
        lines.append(f"- Cases: {_fmt(all_metrics.get('n_cases'))}")
        lines.append(f"- Axes: {_fmt(all_metrics.get('n_axes'))}")
        lines.append(f"- Mean robustness margin: {_fmt(all_metrics.get('robustness_margin_mean'))}")
        lines.append(f"- Median robustness margin: {_fmt(all_metrics.get('robustness_margin_median'))}")
        lines.append(f"- Transition rate: {_fmt(all_metrics.get('transition_rate'))}")
        lines.append(f"- Transitions observed: {_fmt(all_metrics.get('n_transitions_observed'))}")
        c12 = next((c for c in claims if c.claim_id == "C12"), None)
        if c12 is not None:
            lines.append(f"- C12 status: **{c12.status}** — {c12.explanation}")
        lines.append("")
    elif cf_status == "unavailable":
        lines.append("## Counterfactual Latent Editing\n")
        lines.append(
            "*Skipped — counterfactual latent editing requires semantic axes and "
            "CLIP text probes. Missing prerequisites were not fabricated.*\n"
        )
        reason = all_metrics.get("counterfactual_reason")
        if reason:
            lines.append(f"- Reason: {reason}\n")

    # Interpolation
    interp_status = all_metrics.get("interpolation_status")
    if interp_status == "computed":
        lines.append("## Walking on the Decoded Semantic Manifold\n")
        lines.append(
            "This module performs spherical interpolation between decoded CLIP "
            "embeddings and tracks text-probe distributions along the path. It "
            "is latent-space interpolation analysis, not a real neural trajectory.\n"
        )
        lines.append(f"- Pairs: {_fmt(all_metrics.get('n_pairs'))}")
        lines.append(f"- Steps per path: {_fmt(all_metrics.get('n_steps'))}")
        lines.append(f"- Mean smoothness: {_fmt(all_metrics.get('mean_smoothness'))}")
        lines.append(f"- Median smoothness: {_fmt(all_metrics.get('median_smoothness'))}")
        lines.append(f"- Mean semantic velocity: {_fmt(all_metrics.get('mean_semantic_velocity'))}")
        lines.append(f"- Abrupt transition rate: {_fmt(all_metrics.get('abrupt_transition_rate'))}")
        lines.append(f"- Top-concept transition count: {_fmt(all_metrics.get('top_concept_transition_count'))}")
        c13 = next((c for c in claims if c.claim_id == "C13"), None)
        if c13 is not None:
            lines.append(f"- C13 status: **{c13.status}** — {c13.explanation}")
        lines.append("")
    elif interp_status == "unavailable":
        lines.append("## Walking on the Decoded Semantic Manifold\n")
        lines.append(
            "*Skipped — interpolation trajectories require CLIP text probes for "
            "semantic trajectory metrics. Missing text embeddings were not fabricated.*\n"
        )
        reason = all_metrics.get("interpolation_reason")
        if reason:
            lines.append(f"- Reason: {reason}\n")

    # NMAS
    if nmas is not None:
        lines.append("## NMAS (Neural-Manifold Alignment Score)\n")
        lines.append(f"- **NMAS = {_fmt(nmas)}**")
        comps = all_metrics.get("nmas_components", {})
        if comps:
            for k, v in comps.items():
                lines.append(f"  - {k}: {_fmt(v)}")
        lines.append("")

    # Claim tests
    lines.append("## Scientific Claim Tests\n")
    lines.append("| ID | Claim | Status | Explanation |")
    lines.append("|----|-------|--------|-------------|")
    for c in claims:
        lines.append(f"| {c.claim_id} | {c.claim_text[:60]}… | **{c.status}** | {c.explanation} |")
    lines.append("")

    # Figures
    if figures:
        lines.append("## Generated Figures\n")
        for fp in sorted(figures):
            name = Path(fp).stem
            lines.append(f"- `{fp}`")
        lines.append("")

    # Scientific honesty
    lines.append("## Scientific Honesty Statement\n")
    lines.append(
        "All metrics in this report were computed from saved model artifacts "
        "(predicted embeddings, target CLIP embeddings, optional kappa values). "
        "No metrics were fabricated.  Modules for which required data was "
        "unavailable are marked as *unavailable* with explicit explanations.  "
        "Claim statuses are determined by transparent threshold comparisons "
        "documented in the claim_tests.json file.\n"
    )

    # Limitations
    lines.append("## Limitations\n")
    lines.append(
        "- Semantic text probes depend on CLIP text encoder availability.\n"
        "- Repeat stability requires per-trial (not image-averaged) predictions.\n"
        "- ROI ablation requires model inference capability at analysis time.\n"
        "- NMAS thresholds are heuristic and should be validated on held-out data.\n"
        "- Calibration analysis uses normalised kappa as a confidence proxy, "
        "not a true probability.\n"
    )

    md_path = output_dir / "summary.md"
    md_path.write_text("\n".join(lines))
    logger.info("Report written to %s", md_path)
    return md_path
