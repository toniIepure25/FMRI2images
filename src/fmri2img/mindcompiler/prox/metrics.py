"""MINDIR-PROX canonical metric registry. Every metric carries metadata (question, definition, range, direction,
null interpretation, failure modes, invariances, minimum sample, participant aggregation, allowed / forbidden
use) and a versioned implementation. Geometry metrics are basis-invariant by construction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

METRIC_VERSION = "prox-metric/1.0.0"


def _orthob(rows):
    return np.linalg.qr(np.asarray(rows, np.float64).T)[0]      # dim x k orthonormal columns


# ---------- geometry ----------
def principal_angles(A, B):
    Qa, Qb = _orthob(A), _orthob(B)
    s = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), -1, 1)
    return np.arccos(s)


def subspace_overlap(A, B):
    """mean squared principal cosine in [0,1]."""
    Qa, Qb = _orthob(A), _orthob(B)
    s = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)
    return float(np.mean(s ** 2))


def grassmann_distance(A, B):
    th = principal_angles(A, B)
    return float(np.sqrt(np.sum(th ** 2)))


def chordal_distance(A, B):
    th = principal_angles(A, B)
    return float(np.sqrt(np.sum(np.sin(th) ** 2)))


def projector_frobenius(A, B):
    Qa, Qb = _orthob(A), _orthob(B)
    Pa, Pb = Qa @ Qa.T, Qb @ Qb.T
    return float(np.linalg.norm(Pa - Pb, "fro"))


def canonical_correlations(A, B):
    Qa, Qb = _orthob(A), _orthob(B)
    return np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)


# ---------- prediction ----------
def pattern_correlation(pred, actual):
    p = np.asarray(pred, np.float64).ravel(); a = np.asarray(actual, np.float64).ravel()
    p = p - p.mean(); a = a - a.mean()
    d = np.linalg.norm(p) * np.linalg.norm(a)
    return float(p @ a / d) if d > 0 else 0.0


def cosine_similarity(pred, actual):
    p = np.asarray(pred, np.float64).ravel(); a = np.asarray(actual, np.float64).ravel()
    d = np.linalg.norm(p) * np.linalg.norm(a)
    return float(abs(p @ a) / d) if d > 0 else 0.0


# ---------- support / calibration ----------
def support_fraction(within_energy, total_energy):
    return float(within_energy / total_energy) if total_energy > 0 else 0.0


def outside_support_residual(within_energy, total_energy):
    return float(1.0 - support_fraction(within_energy, total_energy))


def total_recovery(R, R0, R_native):
    return float((R - R0) / max(R_native - R0, 1e-12))


def full_capacity_fraction(R, R0, R111):
    return float((R - R0) / max(R111 - R0, 1e-12))


def sample_efficiency(recovery, n_obs):
    return float(recovery / n_obs) if n_obs > 0 else 0.0


def calibration_regret(recovery_achieved, recovery_oracle):
    return float(max(0.0, recovery_oracle - recovery_achieved))


# ---------- stability ----------
def repeat_agreement(subspaces):
    """mean pairwise subspace overlap across repeat-estimated subspaces."""
    ov = [subspace_overlap(subspaces[i], subspaces[j]) for i in range(len(subspaces)) for j in range(i + 1, len(subspaces))]
    return float(np.mean(ov)) if ov else 0.0


def split_half_reliability(a_half, b_half):
    return subspace_overlap(a_half, b_half)


# ---------- registry ----------
@dataclass
class Metric:
    name: str
    family: str
    question: str
    definition: str
    fn: Callable
    range: str
    direction: str                # higher_better / lower_better / two_sided
    null_interpretation: str
    failure_modes: str
    invariances: str
    min_sample: str
    participant_aggregation: str
    allowed_use: str
    forbidden_interpretation: str
    version: str = METRIC_VERSION

    def meta(self):
        return {k: v for k, v in self.__dict__.items() if k != "fn"}


REGISTRY: Dict[str, Metric] = {}


def _reg(m: Metric):
    REGISTRY[m.name] = m


_reg(Metric("subspace_overlap", "GEOMETRY", "How aligned are two subspaces?",
            "mean squared principal cosine", subspace_overlap, "[0,1]", "higher_better",
            "0 under random rank-matched subspaces; 1 identical", "collapses to trivial if either subspace is full-rank",
            "orthogonal-basis invariant; symmetric", "k+1 samples per subspace", "median over participants",
            "compare two representational subspaces", "do NOT read as biological similarity"))
_reg(Metric("grassmann_distance", "GEOMETRY", "How far apart are two subspaces?",
            "sqrt sum of squared principal angles", grassmann_distance, "[0, inf)", "lower_better",
            "large under random subspaces", "unstable if ranks differ", "orthogonal-basis invariant",
            "k+1 samples", "median", "subspace distance", "not a distance in neural space"))
_reg(Metric("chordal_distance", "GEOMETRY", "Projector chordal distance",
            "sqrt sum sin^2 principal angles", chordal_distance, "[0, sqrt k]", "lower_better",
            "large under random", "rank-sensitive", "orthogonal-basis invariant", "k+1", "median",
            "subspace distance", "not neural distance"))
_reg(Metric("projector_frobenius", "GEOMETRY", "Projector Frobenius distance",
            "||P_A - P_B||_F", projector_frobenius, "[0, sqrt(2k)]", "lower_better", "large under random",
            "rank-sensitive", "orthogonal-basis invariant", "k+1", "median", "projector distance", "not neural distance"))
_reg(Metric("pattern_correlation", "PREDICTION", "Held-out pattern correlation",
            "Pearson r of vectorized patterns", pattern_correlation, "[-1,1]", "higher_better",
            "0 under permuted labels", "inflated by low-dim structure; sensitive to outliers",
            "invariant to global scale/shift", ">=3 participants for inference", "median + sign-flip",
            "predicted vs actual pattern", "not R^2; not causal"))
_reg(Metric("cosine_similarity", "PREDICTION", "Directional alignment",
            "|cos| of vectors", cosine_similarity, "[0,1]", "higher_better", "~0 under random", "magnitude-insensitive",
            "scale invariant", ">=3", "median + sign-flip", "direction alignment", "not magnitude match"))
_reg(Metric("support_fraction", "SUPPORT", "Fraction of target energy within perception support",
            "within_energy / total_energy", support_fraction, "[0,1]", "context", "n/a (descriptive)",
            "depends on frozen support basis", "invariant to within-support rotation", "1 target", "median",
            "quantify perception-explained energy", "not proof perception determines imagery"))
_reg(Metric("outside_support_residual", "SUPPORT", "Fraction outside perception support",
            "1 - support_fraction", outside_support_residual, "[0,1]", "context", "n/a", "representation-dependent",
            "invariant to within-support rotation", "1 target", "median", "quantify residual", "not a fixed biological quantity"))
_reg(Metric("total_recovery", "CALIBRATION", "Recovery toward native oracle",
            "(R-R0)/(R_native-R0)", total_recovery, "(-inf,1]", "higher_better", "0 = baseline; 1 = oracle",
            "unstable if R_native~R0", "ratio invariant to affine R shift", "1 fold", "clip[0,1] then median",
            "calibration recovery", "not a universal minimum"))
_reg(Metric("full_capacity_fraction", "CALIBRATION", "Recovery toward full-resource ceiling",
            "(R-R0)/(R111-R0)", full_capacity_fraction, "(-inf,1]", "higher_better", "0 baseline; 1 ceiling",
            "unstable if R111~R0", "ratio invariant", "1 fold", "clip[0,1] then median", "capacity fraction", "estimator-specific"))
_reg(Metric("calibration_regret", "ADAPTIVE", "Recovery lost vs oracle stop",
            "max(0, oracle - achieved)", calibration_regret, "[0,1]", "lower_better", "0 = matched oracle",
            "needs a defined oracle", "n/a", "1 participant", "median", "adaptive-policy cost", "not a p-value"))
_reg(Metric("repeat_agreement", "STABILITY", "Cross-repeat subspace agreement",
            "mean pairwise overlap", repeat_agreement, "[0,1]", "higher_better", "low under unreliable repeats",
            "needs >=2 repeats", "orthogonal-basis invariant", ">=2 repeats", "median", "reliability of repeat geometry",
            "not scalar SNR"))
_reg(Metric("split_half_reliability", "STABILITY", "Split-half subspace reliability",
            "overlap of two halves", split_half_reliability, "[0,1]", "higher_better", "low under noise",
            "sensitive to split", "orthogonal-basis invariant", "4 samples", "median", "reliability", "not test-retest"))


def list_registry():
    return {name: m.meta() for name, m in REGISTRY.items()}
