"""
vMF-Compatible Risk-Coverage Evaluation for Selective Brain Decoding
====================================================================

Extends the standard risk-coverage framework to work with vMF uncertainty
(kappa, delta) instead of Gaussian logvar.  This is the first application
of selective prediction with risk-coverage curves to neural image decoding.

Key functions:
    1. compute_vmf_risk_coverage: Risk-coverage curve using kappa as
       the confidence ordering (higher kappa = lower risk).
    2. compute_dual_risk_coverage: Joint kappa + delta risk ordering.
    3. compute_hierarchical_selective: Fall back from instance-level to
       category-level prediction when uncertain.
    4. compute_vmf_aurc: Area Under Risk-Coverage curve.

The risk-coverage curve answers: "If we abstain on the X% least certain
predictions, how much does average error decrease?"

Clinical relevance: For BCI deployment, selective prediction with a
known error guarantee is essential for safety.

References:
    - El-Yaniv & Wiener (2010) On the foundations of noise-free selective classification
    - Geifman & El-Yaniv (2017) Selective classification for deep neural networks
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskCoverageResult:
    """Container for risk-coverage analysis."""
    coverages: np.ndarray        # (n_points,) coverage levels
    risks: np.ndarray            # (n_points,) risk at each coverage
    aurc: float                  # Area Under Risk-Coverage curve
    e_aurc: float                # Excess AURC (vs oracle ordering)
    oracle_aurc: float           # AURC with perfect ordering
    r_at_80: float               # Risk at 80% coverage
    r_at_90: float               # Risk at 90% coverage
    r_at_95: float               # Risk at 95% coverage
    optimal_coverage: float      # Coverage at which risk = target


def compute_vmf_risk_coverage(
    errors: np.ndarray,
    kappas: np.ndarray,
    n_points: int = 200,
    risk_metric: str = "cosine_error",
) -> RiskCoverageResult:
    """
    Compute risk-coverage curve using vMF kappa as confidence ordering.

    Higher kappa = more confident = should have lower error.
    Ordering by descending kappa and computing cumulative risk gives
    the selective prediction risk-coverage curve.

    Args:
        errors: (N,) per-sample error values (higher = worse).
        kappas: (N,) per-sample kappa concentrations.
        n_points: Number of points on the curve.
        risk_metric: Label for the risk metric type.

    Returns:
        RiskCoverageResult with curves and summary statistics.
    """
    N = len(errors)

    # Sort by kappa descending (most confident first)
    sorted_idx = np.argsort(-kappas)
    sorted_errors = errors[sorted_idx]

    coverages, risks = _compute_risk_curve(sorted_errors, N, n_points)
    aurc = float(np.trapezoid(risks, coverages))

    # Oracle: sort by error ascending (best predictions first)
    oracle_idx = np.argsort(errors)
    oracle_errors = errors[oracle_idx]
    _, oracle_risks = _compute_risk_curve(oracle_errors, N, n_points)
    oracle_aurc = float(np.trapezoid(oracle_risks, coverages))

    return RiskCoverageResult(
        coverages=coverages,
        risks=risks,
        aurc=aurc,
        e_aurc=aurc - oracle_aurc,
        oracle_aurc=oracle_aurc,
        r_at_80=_risk_at_coverage(sorted_errors, N, 0.80),
        r_at_90=_risk_at_coverage(sorted_errors, N, 0.90),
        r_at_95=_risk_at_coverage(sorted_errors, N, 0.95),
        optimal_coverage=_optimal_coverage(sorted_errors, N, target_risk=0.3),
    )


def compute_dual_risk_coverage(
    errors: np.ndarray,
    kappas: np.ndarray,
    deltas: np.ndarray,
    alpha_delta: float = 0.5,
    n_points: int = 200,
) -> RiskCoverageResult:
    """
    Risk-coverage using combined kappa + delta confidence.

    Confidence = kappa_norm * (1 - alpha * delta)

    This jointly considers within-region certainty (kappa) and
    between-region agreement (delta).

    Args:
        errors: (N,) per-sample errors.
        kappas: (N,) per-sample kappa values.
        deltas: (N,) per-sample delta disagreement scores.
        alpha_delta: Weight for delta in combined confidence.
        n_points: Number of curve points.

    Returns:
        RiskCoverageResult.
    """
    # Normalize kappa to [0, 1]
    kn = (kappas - kappas.min()) / (kappas.max() - kappas.min() + 1e-8)

    # Combined confidence
    confidence = kn * (1.0 - alpha_delta * np.clip(deltas, 0.0, 1.0))

    N = len(errors)
    sorted_idx = np.argsort(-confidence)
    sorted_errors = errors[sorted_idx]

    coverages, risks = _compute_risk_curve(sorted_errors, N, n_points)
    aurc = float(np.trapezoid(risks, coverages))

    oracle_idx = np.argsort(errors)
    oracle_errors = errors[oracle_idx]
    _, oracle_risks = _compute_risk_curve(oracle_errors, N, n_points)
    oracle_aurc = float(np.trapezoid(oracle_risks, coverages))

    return RiskCoverageResult(
        coverages=coverages,
        risks=risks,
        aurc=aurc,
        e_aurc=aurc - oracle_aurc,
        oracle_aurc=oracle_aurc,
        r_at_80=_risk_at_coverage(sorted_errors, N, 0.80),
        r_at_90=_risk_at_coverage(sorted_errors, N, 0.90),
        r_at_95=_risk_at_coverage(sorted_errors, N, 0.95),
        optimal_coverage=_optimal_coverage(sorted_errors, N, target_risk=0.3),
    )


def compute_hierarchical_selective(
    errors_instance: np.ndarray,
    errors_category: np.ndarray,
    kappas: np.ndarray,
    kappa_threshold: float = 0.3,
    n_points: int = 200,
) -> Dict[str, np.ndarray]:
    """
    Hierarchical selective prediction: fall back to category level.

    When instance-level identification is uncertain (low kappa), fall
    back to category-level prediction (faces, places, objects) instead
    of fully abstaining.

    Args:
        errors_instance: (N,) instance-level errors (e.g., 1 - R@1).
        errors_category: (N,) category-level errors (e.g., 1 - category_acc).
        kappas: (N,) per-sample kappa concentrations.
        kappa_threshold: Normalized kappa below which to use category-level.
        n_points: Number of curve points.

    Returns:
        Dict with:
            coverage_instance: Coverage using instance-level only
            coverage_hierarchical: Coverage using instance + fallback
            risk_instance: Risk at each coverage (instance only)
            risk_hierarchical: Risk at each coverage (hierarchical)
            aurc_instance: AURC for instance-only
            aurc_hierarchical: AURC for hierarchical
    """
    N = len(errors_instance)
    kn = (kappas - kappas.min()) / (kappas.max() - kappas.min() + 1e-8)

    # Hierarchical: use instance when confident, category when not
    use_instance = kn >= kappa_threshold
    hierarchical_errors = np.where(use_instance, errors_instance, errors_category)

    sorted_idx_inst = np.argsort(-kappas)
    sorted_idx_hier = np.argsort(-kappas)

    coverages, risks_inst = _compute_risk_curve(
        errors_instance[sorted_idx_inst], N, n_points
    )
    _, risks_hier = _compute_risk_curve(
        hierarchical_errors[sorted_idx_hier], N, n_points
    )

    return {
        "coverages": coverages,
        "risk_instance": risks_inst,
        "risk_hierarchical": risks_hier,
        "aurc_instance": float(np.trapezoid(risks_inst, coverages)),
        "aurc_hierarchical": float(np.trapezoid(risks_hier, coverages)),
        "instance_fraction": float(use_instance.mean()),
    }


def compare_uncertainty_orderings(
    errors: np.ndarray,
    orderings: Dict[str, np.ndarray],
    n_points: int = 200,
) -> Dict[str, RiskCoverageResult]:
    """
    Compare multiple uncertainty orderings on the same errors.

    Useful for ablation: compare kappa-only, delta-only, combined, random.

    Args:
        errors: (N,) per-sample errors.
        orderings: Dict mapping name -> (N,) confidence scores
                   (higher = more confident).

    Returns:
        Dict mapping name -> RiskCoverageResult.
    """
    results = {}
    N = len(errors)

    for name, confidence in orderings.items():
        sorted_idx = np.argsort(-confidence)
        sorted_errors = errors[sorted_idx]

        coverages, risks = _compute_risk_curve(sorted_errors, N, n_points)
        aurc = float(np.trapezoid(risks, coverages))

        oracle_idx = np.argsort(errors)
        oracle_errors = errors[oracle_idx]
        _, oracle_risks = _compute_risk_curve(oracle_errors, N, n_points)
        oracle_aurc = float(np.trapezoid(oracle_risks, coverages))

        results[name] = RiskCoverageResult(
            coverages=coverages,
            risks=risks,
            aurc=aurc,
            e_aurc=aurc - oracle_aurc,
            oracle_aurc=oracle_aurc,
            r_at_80=_risk_at_coverage(sorted_errors, N, 0.80),
            r_at_90=_risk_at_coverage(sorted_errors, N, 0.90),
            r_at_95=_risk_at_coverage(sorted_errors, N, 0.95),
            optimal_coverage=_optimal_coverage(sorted_errors, N, target_risk=0.3),
        )

    return results


# ---------- Internal helpers ----------


def _compute_risk_curve(
    sorted_errors: np.ndarray, N: int, n_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (coverage, risk) from pre-sorted errors."""
    coverages = np.linspace(0, 1, n_points)
    risks = np.zeros(n_points)
    for i, cov in enumerate(coverages):
        if cov <= 0:
            risks[i] = 0.0
        else:
            n_covered = max(1, int(cov * N))
            risks[i] = sorted_errors[:n_covered].mean()
    return coverages, risks


def _risk_at_coverage(
    sorted_errors: np.ndarray, N: int, coverage: float,
) -> float:
    n = max(1, int(coverage * N))
    return float(sorted_errors[:n].mean())


def _optimal_coverage(
    sorted_errors: np.ndarray, N: int, target_risk: float,
) -> float:
    """Find maximum coverage achieving target risk."""
    for n in range(N, 0, -1):
        if sorted_errors[:n].mean() <= target_risk:
            return n / N
    return 0.0
