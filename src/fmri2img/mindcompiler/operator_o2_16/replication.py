"""O2.16 Independent Prospective Replication with Calibration-Quality Monitoring (frozen config 2da2cc79).
FROZEN protocol + implementation, ready for an independent N>=8 cohort. Two axes: (A) exact O2.9 M4T2 direct
target-state calibration operational replication; (B) a training-only quality monitor computable from the 8
M4T2 observations THEMSELVES -- REPEAT-A vs REPEAT-B geometry agreement (unlike O2.15's 2-vs-6, which needs
extra repeats). No new estimator / no perception prior / no covariate correction / no threshold search.

This module is the frozen formula (Q_MON_IN, Q_MON_OUT, M4T2 replication criterion, 4-test classification) and
its technical certification. Execution requires an eligible independent cohort (see Stage-0 data audit); until
one is available the gate seals O2_16_INDEPENDENT_REPLICATION_DATA_UNAVAILABLE without running inference."""
from __future__ import annotations

import numpy as np

sys_ok = True  # placeholder for import symmetry; real driver wires O2.9 SVD helpers when data exist


def _within_basis(z, q):
    """Top-q left singular vectors of the K x 4 coord matrix (columns = identity coords). z is (4 x K)."""
    U, s, _ = np.linalg.svd(np.asarray(z, np.float64).T, full_matrices=False)
    eps = np.finfo(np.float64).eps * max(z.shape) * (s[0] if s.size else 0.0)
    if int(np.sum(s > eps)) < q:
        return None
    return U[:, :q]


def _outside_basis(dout, D):
    """Top-D left singular vectors of the V x 4 outside-residual matrix. dout is (V x 4)."""
    U, s, _ = np.linalg.svd(np.asarray(dout, np.float64), full_matrices=False)
    eps = np.finfo(np.float64).eps * max(dout.shape) * (s[0] if s.size else 0.0)
    if int(np.sum(s > eps)) < D:
        return None
    return U[:, :D]


def _proj_sim(U, V, k):
    return float(np.sum((U.T @ V) ** 2) / k)


def q_mon(W, dA, dB, r_best):
    """1-vs-1 training-only quality monitor from a single M4T2 acquisition (4 identities, repeats A and B).
    dA, dB: (4 x V) repeat-A / repeat-B target residuals for the 4 calibration identities. Returns (Q_MON_IN,
    Q_MON_OUT), each trace(P_A P_B)/rank in [0,1]. Symmetric in A<->B. Uses NO held-out imagery."""
    W = np.asarray(W, np.float64); dA = np.asarray(dA, np.float64); dB = np.asarray(dB, np.float64)
    q = min(int(r_best), 4)
    zA = dA @ W; zB = dB @ W                                                    # (4 x K)
    UA = _within_basis(zA, q); UB = _within_basis(zB, q)
    oA = np.stack([dA[i] - W @ (W.T @ dA[i]) for i in range(dA.shape[0])])      # (4 x V)
    oB = np.stack([dB[i] - W @ (W.T @ dB[i]) for i in range(dB.shape[0])])
    BA = _outside_basis(oA.T, 2); BB = _outside_basis(oB.T, 2)
    q_in = _proj_sim(UA, UB, q) if (UA is not None and UB is not None) else float("nan")
    q_out = _proj_sim(BA, BB, 2) if (BA is not None and BB is not None) else float("nan")
    return q_in, q_out


def q_mon_comp(q_in, q_out, q):
    """Descriptive composite (NOT used in primary inference)."""
    return (q * q_in + 2 * q_out) / (q + 2)


# ---------------- M4T2 operational replication criterion ----------------
def _ceil75(n):
    return int(np.ceil(0.75 * n))


def m4t2_operational_pass(tots, fcfs):
    """tots/fcfs: clipped [0,1] per-participant. Historical 6/8 generalized to ceil(0.75*N)."""
    n = len(tots)
    if n == 0:
        return False
    thr = _ceil75(n)
    return bool(np.median(tots) >= 0.50 and sum(t >= 0.5 for t in tots) >= thr
                and np.median(fcfs) >= 0.50 and sum(f >= 0.5 for f in fcfs) >= thr)


def quality_monitor_pass(rhos, holm_reject):
    """rhos: per-participant Spearman; pass iff median>0 AND >=75% >0 AND Holm reject."""
    n = len([r for r in rhos if np.isfinite(r)])
    if n == 0:
        return False
    pos = sum(r > 0 for r in rhos if np.isfinite(r))
    return bool(np.median([r for r in rhos if np.isfinite(r)]) > 0 and pos >= _ceil75(n) and holm_reject)


def classify_roi(m4t2_pass, qin_rep, qout_rep):
    if not m4t2_pass and (qin_rep or qout_rep):
        return "QUALITY_MONITOR_REPLICATED_M4T2_NOT_OPERATIONALLY_SUFFICIENT"
    if not m4t2_pass:
        return "M4T2_AND_MONITOR_NOT_REPLICATED"
    if qin_rep and qout_rep:
        return "M4T2_AND_QUALITY_MONITOR_REPLICATED"
    if qin_rep or qout_rep:
        return "M4T2_REPLICATED_QUALITY_MONITOR_PARTIAL"
    return "M4T2_REPLICATED_QUALITY_MONITOR_NOT_REPLICATED"


def classify_program(roi_status, rois=("ventral", "lateral")):
    sv, sl = roi_status[rois[0]], roi_status[rois[1]]
    full = "M4T2_AND_QUALITY_MONITOR_REPLICATED"
    if sv == sl == full:
        return "INDEPENDENT_M4T2_AND_CALIBRATION_QUALITY_REPLICATION_PASS"
    m4_pass = {"M4T2_AND_QUALITY_MONITOR_REPLICATED", "M4T2_REPLICATED_QUALITY_MONITOR_PARTIAL", "M4T2_REPLICATED_QUALITY_MONITOR_NOT_REPLICATED"}
    if sv in m4_pass and sl in m4_pass and (sv == "M4T2_REPLICATED_QUALITY_MONITOR_NOT_REPLICATED" and sl == "M4T2_REPLICATED_QUALITY_MONITOR_NOT_REPLICATED"):
        return "INDEPENDENT_M4T2_REPLICATION_PASS_MONITOR_FAIL"
    if sv == sl == "QUALITY_MONITOR_REPLICATED_M4T2_NOT_OPERATIONALLY_SUFFICIENT":
        return "INDEPENDENT_FAILURE_MODE_REPLICATION_PASS_M4T2_FAIL"
    if sv == sl == "M4T2_AND_MONITOR_NOT_REPLICATED":
        return "INDEPENDENT_REPLICATION_NOT_CONFIRMED"
    return "INDEPENDENT_REPLICATION_PARTIAL"
