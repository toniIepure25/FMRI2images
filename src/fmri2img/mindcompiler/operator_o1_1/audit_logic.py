"""O1.1 frozen pure decision logic (audit only; no fitting, no data, no new models)."""
from __future__ import annotations

EPS = 0.05


def operator_gain(o4: float, o0: float) -> float:
    """G = OPERATOR_INCREMENTAL_GAIN = O4 - O0 (raw O4 is NOT operator evidence)."""
    return o4 - o0


def headroom(o0: float) -> float:
    return 1.0 - o0


def normalized_gain(o4: float, o0: float, eps: float = EPS) -> float:
    return (o4 - o0) / max(1.0 - o0, eps)


def std_median_diff(median_full: float, median_no: float, pooled_mad: float) -> float:
    return (median_full - median_no) / (pooled_mad if pooled_mad > 1e-12 else 1e-12)


def fm4_beta_status(rho_img: float, rho_vis: float, n: int = 8) -> str:
    if n < 6 or (rho_img != rho_img and rho_vis != rho_vis):
        return "BETA_OPERATOR_DEPENDENCE_INCONCLUSIVE"
    best = max([x for x in (rho_img, rho_vis) if x == x], default=float("nan"))
    if best >= 0.5:
        return "BETA_OPERATOR_DEPENDENCE_TRACKS_RELIABILITY"
    if best >= 0.2:
        return "BETA_OPERATOR_DEPENDENCE_PARTLY_TRACKS_RELIABILITY"
    return "BETA_OPERATOR_DEPENDENCE_NOT_EXPLAINED_BY_RELIABILITY"


def fm5_stability_status(std_diff: float, median_rho: float) -> str:
    if std_diff >= 1.0 and median_rho >= 0.5:
        return "INSTABILITY_DOMINANT"
    if (0.4 <= std_diff < 1.0) or (0.2 <= median_rho < 0.5):
        return "INSTABILITY_CONTRIBUTES"
    if std_diff < 0.4 and abs(median_rho) < 0.2:
        return "STABILITY_NOT_EXPLANATORY"
    return "MIXED"


def fm_measurement_status(median_rho: float) -> str:
    if median_rho >= 0.4:
        return "SUPPORTED_PATTERN"
    if median_rho >= 0.2:
        return "PARTIAL_PATTERN"
    return "NOT_SUPPORTED"


def participant_regime(n_positive_g_rois: int) -> str:
    if n_positive_g_rois >= 4:
        return "BROAD_OPERATOR_EVIDENCE"
    if n_positive_g_rois >= 1:
        return "REGION_SPECIFIC_OPERATOR_EVIDENCE"
    return "WEAK_OPERATOR_EVIDENCE"


def o2_ready(a: bool, b: bool, c: bool, d: bool, e: bool) -> str:
    return "O2_READY_FOR_SHARED_OPERATOR_TEST" if (a and b and c and d and e) else "O2_NOT_READY"
