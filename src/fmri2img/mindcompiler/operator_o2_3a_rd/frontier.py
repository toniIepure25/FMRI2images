"""O2.4R calibration-frontier orchestration (frozen config ad959446), built on the certified geometry.

Operates on abstract per-cell state (shared-space coords X, target coords Z, W_target, U_res, held-out
target-native deltas), so it is fully deterministic and data-free-testable. Consumes the O2.3A-RD-produced
per subject x ROI x fold state; produces R_CAL(M), the identity-correspondence null, participant effects
E_M, exact sign-flip + Holm inference, and M_STAR. No RNG except the frozen null permutation seed.
"""
from __future__ import annotations

import hashlib
import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import geometry as G

BUDGETS = [0, 2, 4, 6, 8, 10]


# ----------------------------------------------------------------- balanced subset enumeration
def balanced_subsets(simple_ids: Sequence[int], nat_ids: Sequence[int], M: int) -> List[Tuple[int, ...]]:
    """All balanced calibration subsets of size M (M/2 simple + M/2 naturalistic). Deterministic order."""
    if M == 0:
        return [tuple()]
    h = M // 2
    out = []
    for cs in itertools.combinations(sorted(simple_ids), h):
        for cn in itertools.combinations(sorted(nat_ids), h):
            out.append(tuple(cs) + tuple(cn))
    return out


def _null_perm(seed_text: str, m: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(G.seed_uint64(seed_text)))
    return rng.permutation(m)


def r_cal_for_Q(Q: np.ndarray, U_res: np.ndarray, W_target: np.ndarray, deltas: Sequence[np.ndarray]) -> float:
    P = G.native_projector(Q, U_res, W_target)
    return float(np.mean([G.retention(P, d) for d in deltas]))


def cell_frontier(cell: Dict, subject: str, roi: str, fold: int, n_null: int = 100,
                  budgets: Sequence[int] = BUDGETS) -> Dict[int, Dict[str, float]]:
    """One subject x ROI x fold. `cell` keys: X (n_ids x K shared coords indexed by identity id),
    Z (n_ids x K target coords), W_target (p x K), U_res (K x r), deltas_test (list held-out native deltas),
    simple_ids, nat_ids (outer-training identity ids), R_CAL0 (float, = retention of P_ZERO_RD)."""
    X, Z, U_res, W = cell["X"], cell["Z"], cell["U_res"], cell["W_target"]
    deltas = cell["deltas_test"]
    res = {0: {"r_cal_true": float(cell["R_CAL0"]), "r_cal_null_mean": float(cell["R_CAL0"]), "n_subsets": 1}}
    for M in budgets:
        if M == 0:
            continue
        subs = balanced_subsets(cell["simple_ids"], cell["nat_ids"], M)
        true_vals, null_vals = [], []
        for si, C in enumerate(subs):
            Xc, Zc = X[list(C)], Z[list(C)]
            Q = G.orthogonal_procrustes(Xc, Zc)
            true_vals.append(r_cal_for_Q(Q, U_res, W, deltas))
            nv = []
            for it in range(n_null):
                seed = "O2.4R|%s|%s|%d|%d|%d|%d" % (subject, roi, fold, M, si, it)
                Zc_perm = Zc[_null_perm(seed, len(C))]
                Qn = G.orthogonal_procrustes(Xc, Zc_perm)
                nv.append(r_cal_for_Q(Qn, U_res, W, deltas))
            null_vals.append(float(np.mean(nv)))
        res[M] = {"r_cal_true": float(np.mean(true_vals)), "r_cal_null_mean": float(np.mean(null_vals)), "n_subsets": len(subs)}
    return res


# ----------------------------------------------------------------- participant aggregation + inference
def participant_E(cell_frontiers: Sequence[Dict[int, Dict[str, float]]], M: int) -> Tuple[float, float]:
    """Average over the participant's folds (each already averaged over held-out identities + balanced
    subsets). Returns (E_M, DELTA_ZERO_M) for one participant."""
    tr = np.mean([cf[M]["r_cal_true"] for cf in cell_frontiers])
    nu = np.mean([cf[M]["r_cal_null_mean"] for cf in cell_frontiers])
    z0 = np.mean([cf[0]["r_cal_true"] for cf in cell_frontiers])
    return float(tr - nu), float(tr - z0)


def signflip_p_onesided(effects: Sequence[float]) -> float:
    """Exact one-sided sign-flip p over 2^N sign configurations (H1: mean effect > 0)."""
    e = np.asarray(effects, dtype=np.float64)
    N = len(e)
    obs = e.sum()
    ge = 0
    for signs in itertools.product([1.0, -1.0], repeat=N):
        if float(np.dot(signs, e)) >= obs - 1e-12:
            ge += 1
    return ge / (2 ** N)


def holm(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, bool]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    reject = {}
    still = True
    for i, (k, p) in enumerate(items):
        thr = alpha / (m - i)
        if still and p <= thr:
            reject[k] = True
        else:
            still = False
            reject[k] = False
    return reject


def m_star(per_M: Dict[int, Dict[str, float]]) -> object:
    """per_M[M] = {median_E, frac_pos, holm_p, median_delta_zero, median_oracle_recovery}. Smallest M with all."""
    for M in [2, 4, 6, 8, 10]:
        d = per_M.get(M)
        if d and d["median_E"] > 0 and d["frac_pos"] >= 6 / 8 and d["holm_p"] < 0.05 \
           and d["median_delta_zero"] > 0 and d["median_oracle_recovery"] >= 0.50:
            return M
    return "NOT_REACHED"
