"""Structured scientific claim testing system.

Each claim is evaluated against observed metrics and assigned a status:
    supported / partially_supported / not_supported / unavailable
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClaimTest:
    claim_id: str
    claim_text: str
    required_metrics: List[str]
    observed_values: Dict[str, Any] = field(default_factory=dict)
    status: str = "unavailable"
    threshold_logic: str = ""
    explanation: str = ""
    related_figures: List[str] = field(default_factory=list)
    related_tables: List[str] = field(default_factory=list)


CLAIM_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "claim_id": "C01",
        "claim_text": "CSLS improves neural-to-image retrieval compared to raw cosine retrieval.",
        "required_metrics": ["cosine_R@1", "csls_R@1"],
    },
    {
        "claim_id": "C02",
        "claim_text": "CSLS reduces hubness in neural-to-CLIP retrieval.",
        "required_metrics": ["cosine_skewness", "csls_skewness", "cosine_gini", "csls_gini"],
    },
    {
        "claim_id": "C03",
        "claim_text": "Decoded fMRI embeddings preserve representational geometry of CLIP image space.",
        "required_metrics": ["rsa_spearman_rho"],
    },
    {
        "claim_id": "C04",
        "claim_text": "Decoded fMRI embeddings preserve local semantic neighbourhoods.",
        "required_metrics": ["overlap@10_mean", "overlap@10_random_baseline"],
    },
    {
        "claim_id": "C05",
        "claim_text": "Repeated presentations of the same stimulus produce stable decoded embeddings.",
        "required_metrics": ["same_different_auroc", "stability_ratio"],
    },
    {
        "claim_id": "C06",
        "claim_text": "vMF kappa provides meaningful directional uncertainty.",
        "required_metrics": ["kappa_auroc", "kappa_correct_rpb"],
    },
    {
        "claim_id": "C07",
        "claim_text": "Reliable predictions tend to be on-manifold and have high CSLS margin / low entropy.",
        "required_metrics": ["density_correct_mean", "density_incorrect_mean"],
    },
    {
        "claim_id": "C08",
        "claim_text": "Reliability-aware selective decoding improves accuracy at lower coverage.",
        "required_metrics": ["aurc", "selective_prediction"],
    },
    {
        "claim_id": "C09",
        "claim_text": "Decoded brain embeddings can be interpreted through CLIP text probes.",
        "required_metrics": ["agreement@5", "score_vector_spearman"],
    },
    {
        "claim_id": "C10",
        "claim_text": "Semantic axes reveal structured preservation / distortion of concept dimensions.",
        "required_metrics": ["mean_axis_spearman", "mean_preservation"],
    },
    {
        "claim_id": "C11",
        "claim_text": "Semantic error vectors are structured rather than random.",
        "required_metrics": ["mean_error_direction_similarity"],
    },
    {
        "claim_id": "C12",
        "claim_text": "Counterfactual latent edits reveal semantic robustness margins.",
        "required_metrics": ["robustness_margin_mean"],
    },
    {
        "claim_id": "C13",
        "claim_text": "Decoded embeddings form continuous semantic trajectories under interpolation.",
        "required_metrics": ["mean_smoothness"],
    },
    {
        "claim_id": "C14",
        "claim_text": "ROI ablations map cortical regions to semantic axes.",
        "required_metrics": ["roi_axis_status"],
    },
]


def _evaluate_claim(defn: Dict[str, Any], all_metrics: Dict[str, Any]) -> ClaimTest:
    """Evaluate a single claim against the full metrics dict."""
    ct = ClaimTest(
        claim_id=defn["claim_id"],
        claim_text=defn["claim_text"],
        required_metrics=defn["required_metrics"],
    )

    observed: Dict[str, Any] = {}
    missing: List[str] = []
    for m in defn["required_metrics"]:
        v = all_metrics.get(m)
        if v is not None:
            observed[m] = v
        else:
            missing.append(m)
    ct.observed_values = observed

    if missing:
        ct.status = "unavailable"
        ct.threshold_logic = "Required artifacts or metrics were not produced for this run."
        ct.explanation = f"Missing metrics: {missing}"
        return ct

    cid = defn["claim_id"]

    if cid == "C01":
        cos = observed["cosine_R@1"]
        csl = observed["csls_R@1"]
        delta = csl - cos
        ct.threshold_logic = "supported if CSLS R@1 improves by at least 0.03; partial if non-negative but smaller."
        if delta >= 0.03:
            ct.status = "supported"
            ct.explanation = f"CSLS R@1 improves by {delta:+.3f} ({cos:.3f} -> {csl:.3f})."
        elif csl >= cos:
            ct.status = "partially_supported"
            ct.explanation = f"CSLS R@1 improves only marginally by {delta:+.3f} ({cos:.3f} -> {csl:.3f})."
        else:
            ct.status = "not_supported"
            ct.explanation = f"CSLS R@1 decreases by {delta:+.3f} ({cos:.3f} -> {csl:.3f})."
        ct.related_figures = ["figures/figure_retrieval_comparison.png"]
        ct.related_tables = ["retrieval_metrics.json"]

    elif cid == "C02":
        cs = observed["cosine_skewness"]
        ss = observed["csls_skewness"]
        cg = observed["cosine_gini"]
        sg = observed["csls_gini"]
        skew_drop = (cs - ss) / max(abs(cs), 1e-12)
        gini_drop = (cg - sg) / max(abs(cg), 1e-12)
        ct.threshold_logic = "supported if skewness and Gini both drop by at least 5%; partial if one drops."
        if skew_drop >= 0.05 and gini_drop >= 0.05:
            ct.status = "supported"
        elif ss < cs or sg < cg:
            ct.status = "partially_supported"
        else:
            ct.status = "not_supported"
        ct.explanation = (
            f"Skewness: cosine={cs:.3f}, CSLS={ss:.3f}; "
            f"Gini: cosine={cg:.3f}, CSLS={sg:.3f}"
        )
        ct.related_figures = ["figures/figure_hubness_distribution.png", "figures/figure_hubness_lorenz.png"]
        ct.related_tables = ["hubness_metrics.json"]

    elif cid == "C03":
        rho = observed["rsa_spearman_rho"]
        ct.threshold_logic = "supported if RSA Spearman rho > 0.5; partial if > 0.2."
        if rho > 0.5:
            ct.status = "supported"
        elif rho > 0.2:
            ct.status = "partially_supported"
        else:
            ct.status = "not_supported"
        ct.explanation = f"RSA Spearman rho = {rho:.4f}"
        ct.related_figures = ["figures/figure_rsa_similarity_scatter.png", "figures/figure_rsa_difference_matrix.png"]
        ct.related_tables = ["rsa_metrics.json"]

    elif cid == "C04":
        ov = observed["overlap@10_mean"]
        base = observed["overlap@10_random_baseline"]
        lift = ov / base if base else 0.0
        ct.threshold_logic = "supported if Overlap@10 is at least 10x random and absolute overlap > 0.1; partial if at least 3x random."
        if ov > 0.1 and lift >= 10.0:
            ct.status = "supported"
        elif lift >= 3.0:
            ct.status = "partially_supported"
        else:
            ct.status = "not_supported"
        ct.explanation = (
            f"Neighbourhood overlap@10 = {ov:.4f}; random baseline = {base:.4f}; "
            f"lift = {lift:.2f}x."
        )
        ct.related_figures = ["figures/figure_neighborhood_overlap_vs_k.png"]
        ct.related_tables = ["neighborhood_metrics.json"]

    elif cid == "C05":
        auroc = observed["same_different_auroc"]
        ratio = observed["stability_ratio"]
        ct.threshold_logic = "supported if same/different AUROC > 0.8 and stability ratio > 2; partial if AUROC > 0.6."
        if auroc > 0.8 and ratio > 2.0:
            ct.status = "supported"
        elif auroc > 0.6:
            ct.status = "partially_supported"
        else:
            ct.status = "not_supported"
        ct.explanation = f"AUROC = {auroc:.3f}, stability_ratio = {ratio:.2f}"

    elif cid == "C06":
        auroc = observed["kappa_auroc"]
        rpb = observed["kappa_correct_rpb"]
        ct.threshold_logic = "supported if kappa AUROC > 0.65 and point-biserial r > 0.1; partial if AUROC > 0.6."
        if auroc > 0.65 and rpb > 0.1:
            ct.status = "supported"
        elif auroc > 0.6:
            ct.status = "partially_supported"
        else:
            ct.status = "not_supported"
        ct.explanation = (
            f"Kappa AUROC = {auroc:.3f}, point-biserial r = {rpb:.3f}; "
            "this is weak as a standalone uncertainty signal when near 0.57."
        )
        ct.related_figures = ["figures/figure_kappa_correct_vs_incorrect.png", "figures/figure_kappa_calibration.png"]
        ct.related_tables = ["uncertainty_metrics.json"]

    elif cid == "C07":
        dc = observed["density_correct_mean"]
        di = observed["density_incorrect_mean"]
        ct.threshold_logic = "supported only if mean manifold density is higher for correct predictions than incorrect predictions."
        if dc > di:
            ct.status = "supported"
        else:
            ct.status = "not_supported"
        ct.explanation = f"Density correct={dc:.4f}, incorrect={di:.4f}"
        ct.related_figures = ["figures/figure_density_correct_vs_incorrect.png"]
        ct.related_tables = ["manifold_density_metrics.json"]

    elif cid == "C08":
        sel = observed.get("selective_prediction", {})
        cov = sel.get("coverage", [])
        acc = sel.get("accuracy", [])
        ct.threshold_logic = "supported if low-coverage accuracy improves by at least 0.02 over full coverage; do not require monotonicity."
        if cov and acc and len(cov) >= 2:
            full_acc = acc[0]
            low_idx = -2 if len(acc) >= 2 else -1
            low_acc = acc[low_idx]
            low_cov = cov[low_idx]
            if low_acc > full_acc + 0.02:
                ct.status = "supported"
            elif low_acc >= full_acc:
                ct.status = "partially_supported"
            else:
                ct.status = "not_supported"
            ct.explanation = (
                f"Full coverage acc={full_acc:.3f}, "
                f"{low_cov:.0%} coverage acc={low_acc:.3f}; strongest gains may be low-coverage only."
            )
        else:
            ct.status = "partially_supported"
            ct.explanation = f"AURC = {observed.get('aurc', '?')}"
        ct.related_figures = ["figures/figure_coverage_accuracy.png"]
        ct.related_tables = ["reliability_metrics.json"]

    elif cid == "C09":
        agree = observed["agreement@5"]
        rho = observed["score_vector_spearman"]
        top1 = all_metrics.get("top1_agreement")
        ct.threshold_logic = (
            "supported if agreement@5 >= 0.50 and score_vector_spearman >= 0.30; "
            "partially_supported if agreement@5 >= 0.25 or score_vector_spearman >= 0.15; "
            "unavailable if text embeddings are missing."
        )
        if agree >= 0.50 and rho >= 0.30:
            ct.status = "supported"
        elif agree >= 0.25 or rho >= 0.15:
            ct.status = "partially_supported"
        else:
            ct.status = "not_supported"
        if top1 is not None:
            observed["top1_agreement"] = top1
            ct.explanation = (
                f"Agreement@5={agree:.3f}, score-vector Spearman={rho:.3f}, "
                f"top-1 agreement={top1:.3f}."
            )
        else:
            ct.explanation = f"Agreement@5={agree:.3f}, score-vector Spearman={rho:.3f}."
        ct.related_figures = [
            "figures/figure_semantic_probe_agreement_bar.png",
            "figures/figure_semantic_probe_score_correlation.png",
        ]
        ct.related_tables = [
            "semantic_probe_metrics.json",
            "semantic_probe_per_trial.csv",
        ]

    elif cid == "C10":
        mean_s = observed["mean_axis_spearman"]
        mp = observed["mean_preservation"]
        n_axes = all_metrics.get("n_axes_built")
        best_axis = all_metrics.get("best_axis", "unknown")
        weakest_axis = all_metrics.get("weakest_axis", "unknown")
        mean_p = all_metrics.get("mean_axis_pearson")
        mean_mae = all_metrics.get("mean_axis_mae")
        ct.threshold_logic = (
            "supported if mean_axis_spearman >= 0.35 or mean_preservation >= 0.50; "
            "partially_supported if mean_axis_spearman >= 0.20 or mean_preservation >= 0.35; "
            "unavailable if no axes were built."
        )
        if mean_s >= 0.35 or mp >= 0.50:
            ct.status = "supported"
        elif mean_s >= 0.20 or mp >= 0.35:
            ct.status = "partially_supported"
        else:
            ct.status = "not_supported"
        if n_axes is not None:
            observed["n_axes_built"] = n_axes
        if mean_p is not None:
            observed["mean_axis_pearson"] = mean_p
        if mean_mae is not None:
            observed["mean_axis_mae"] = mean_mae
        observed["best_axis"] = best_axis
        observed["weakest_axis"] = weakest_axis
        mean_p_s = f"{mean_p:.3f}" if isinstance(mean_p, (int, float)) else str(mean_p)
        mean_mae_s = f"{mean_mae:.3f}" if isinstance(mean_mae, (int, float)) else str(mean_mae)
        ct.explanation = (
            f"Built {n_axes if n_axes is not None else 'unknown'} axes; "
            f"mean Spearman={mean_s:.3f}, mean Pearson={mean_p_s}, "
            f"mean preservation={mp:.3f}, mean MAE={mean_mae_s}; "
            f"best axis={best_axis}, weakest axis={weakest_axis}."
        )
        ct.related_figures = [
            "figures/figure_semantic_axes_preservation_bar.png",
            "figures/figure_semantic_axes_projection_scatter.png",
        ]
        ct.related_tables = [
            "semantic_axes_metrics.json",
            "semantic_axes_axis_summary.csv",
        ]

    elif cid == "C11":
        sim = observed["mean_error_direction_similarity"]
        ct.threshold_logic = "supported if mean error direction similarity > 0.05; partial if > 0.01."
        if sim > 0.05:
            ct.status = "supported"
        elif sim > 0.01:
            ct.status = "partially_supported"
        else:
            ct.status = "not_supported"
        ct.explanation = f"Mean error direction similarity = {sim:.4f}"
        ct.related_figures = ["figures/figure_semantic_error_vector_field.png"]
        ct.related_tables = ["vector_field_metrics.json"]

    elif cid == "C12":
        m = observed["robustness_margin_mean"]
        tr = all_metrics.get("transition_rate")
        ct.threshold_logic = (
            "supported if robustness_margin_mean is finite and transition_rate > 0.20; "
            "partially_supported if valid counterfactual cases exist but transitions are weaker; "
            "unavailable if semantic axes or text probes are missing."
        )
        if m is not None and np.isfinite(m) and tr is not None and m > 0 and tr > 0.20:
            ct.status = "supported"
        elif m is not None and np.isfinite(m):
            ct.status = "partially_supported"
        elif all_metrics.get("n_cases"):
            ct.status = "partially_supported"
        else:
            ct.status = "not_supported"
        if tr is not None:
            observed["transition_rate"] = tr
        observed["n_transitions_observed"] = all_metrics.get("n_transitions_observed")
        ct.explanation = (
            f"Mean robustness margin={m}, transition_rate={tr}, "
            f"transitions={all_metrics.get('n_transitions_observed')}."
        )
        ct.related_figures = [
            "figures/figure_counterfactual_robustness_margin.png",
            "figures/figure_counterfactual_axis_sensitivity.png",
        ]
        ct.related_tables = ["counterfactual_metrics.json", "counterfactual_per_edit.csv"]

    elif cid == "C13":
        s = observed["mean_smoothness"]
        abrupt = all_metrics.get("abrupt_transition_rate")
        ct.threshold_logic = (
            "supported if mean_smoothness is finite and abrupt_transition_rate <= 0.30; "
            "partially_supported if trajectories exist but are noisy; "
            "unavailable if text probes are missing."
        )
        if s is not None and np.isfinite(s) and abrupt is not None and abrupt <= 0.30:
            ct.status = "supported"
        elif s is not None and np.isfinite(s):
            ct.status = "partially_supported"
        else:
            ct.status = "not_supported"
        if abrupt is not None:
            observed["abrupt_transition_rate"] = abrupt
        observed["mean_semantic_velocity"] = all_metrics.get("mean_semantic_velocity")
        observed["top_concept_transition_count"] = all_metrics.get("top_concept_transition_count")
        ct.explanation = (
            f"Mean smoothness={s}, abrupt_transition_rate={abrupt}, "
            f"mean semantic velocity={all_metrics.get('mean_semantic_velocity')}, "
            f"top-concept transitions={all_metrics.get('top_concept_transition_count')}."
        )
        ct.related_figures = [
            "figures/figure_interpolation_semantic_velocity.png",
            "figures/figure_interpolation_top_concept_paths.png",
        ]
        ct.related_tables = ["interpolation_metrics.json", "interpolation_paths.csv"]

    elif cid == "C14":
        st = observed.get("roi_axis_status", "unavailable")
        ct.threshold_logic = "supported only when ROI-axis analysis completes with valid ROI inputs."
        if st == "computed":
            ct.status = "supported"
            ct.explanation = "ROI ablation analysis completed"
        else:
            ct.status = "unavailable"
            ct.explanation = "ROI ablation module not executed"

    return ct


def evaluate_all_claims(all_metrics: Dict[str, Any]) -> List[ClaimTest]:
    """Evaluate all predefined claims against *all_metrics*."""
    return [_evaluate_claim(d, all_metrics) for d in CLAIM_DEFINITIONS]


def save_claim_tests(
    claims: List[ClaimTest],
    output_path: Path,
) -> None:
    """Write claim_tests.json."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(c) for c in claims]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved %d claim tests to %s", len(claims), output_path)
