"""M2 -- Zero / Near-Zero Shot Subject-Specific Geometry Prediction.

Question: can we predict the subject-specific MISSING imagery geometry BEFORE direct imagery calibration?

Strict, participant-disjoint (leave-one-subject-out) prediction of a low-dimensional summary of the target
imagery geometry from richer NON-imagery descriptors (perception phenotype, anatomy, behaviour). Two protocols:
  ZERO_TARGET       -- no imagery of the target participant is used at all.
  NEAR_ZERO_TARGET  -- at most 1-2 imagery observations of the target may anchor sign/scale (declared, bounded).
Bounded, low-capacity model only (ridge / low-rank linear) with a tiny parameter budget appropriate for N~12;
an identifiability audit enforces the budget. Prospective metric: alignment between predicted and actual target
geometry summary; prospective null = permuted participant labels. On real Cohort B this runs only AFTER the
O2.16 unlock chain; here it is SYNTHETIC engineering only and any pass is not scientific."""
from __future__ import annotations

import numpy as np

from . import common as CM

# frozen model budget (declared before data)
RIDGE_LAMBDA = 10.0                 # strong regularization for N~12
MAX_DESCRIPTOR_RANK = 3             # low-rank cap on the descriptor->geometry map
GEOMETRY_SUMMARY_DIM = 2           # predict a 2-d summary of the outside-support orientation
N_PERM_NULL = 2000
PROTOCOLS = ("ZERO_TARGET", "NEAR_ZERO_TARGET")


def identifiability_audit(n_participants, descriptor_dim):
    """Guard against over-parameterization at N~12. Effective params must be << N * summary_dim observations."""
    eff_params = min(descriptor_dim, MAX_DESCRIPTOR_RANK) * GEOMETRY_SUMMARY_DIM
    n_obs = n_participants * GEOMETRY_SUMMARY_DIM
    ok = eff_params <= max(1, n_obs // 3)   # keep params <= ~1/3 of observations
    return {"n_participants": n_participants, "descriptor_dim": descriptor_dim,
            "effective_params": eff_params, "pseudo_observations": n_obs, "ridge_lambda": RIDGE_LAMBDA,
            "max_rank": MAX_DESCRIPTOR_RANK, "identifiable_budget_ok": bool(ok),
            "note": "low-capacity bounded model required at N~12; audit fails closed if over-parameterized"}


def _ridge_lowrank_fit(Xtr, Ytr):
    """Ridge then rank-truncate the map to MAX_DESCRIPTOR_RANK (bounded capacity)."""
    d = Xtr.shape[1]
    W = np.linalg.solve(Xtr.T @ Xtr + RIDGE_LAMBDA * np.eye(d), Xtr.T @ Ytr)   # (d x summary)
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    k = min(MAX_DESCRIPTOR_RANK, len(s))
    return (U[:, :k] * s[:k]) @ Vt[:k]


def _align(pred, actual):
    """Alignment = cosine between predicted and actual summary vectors (sign-invariant magnitude)."""
    a = pred / (np.linalg.norm(pred) + 1e-12); b = actual / (np.linalg.norm(actual) + 1e-12)
    return abs(float(a @ b))


def loso_predict(descriptors, geometry, protocol="ZERO_TARGET"):
    """descriptors: (N x d); geometry: (N x summary). Returns per-participant alignment under LOSO."""
    N = descriptors.shape[0]; aligns = []
    for i in range(N):
        tr = [j for j in range(N) if j != i]
        W = _ridge_lowrank_fit(descriptors[tr], geometry[tr])
        pred = descriptors[i] @ W
        actual = geometry[i].copy()
        if protocol == "NEAR_ZERO_TARGET":
            actual = actual  # 1-2 imagery obs would only fix sign/scale; alignment is already sign-invariant
        aligns.append(_align(pred, actual))
    return np.array(aligns)


def permutation_null(descriptors, geometry, protocol, n_perm=N_PERM_NULL, name="m2"):
    r = CM.rng("M2|perm|%s|%s" % (protocol, name))
    med_real = float(np.median(loso_predict(descriptors, geometry, protocol)))
    null = []
    N = descriptors.shape[0]
    for it in range(n_perm):
        perm = r.permutation(N)
        null.append(float(np.median(loso_predict(descriptors, geometry[perm], protocol))))
    null = np.array(null)
    p = float((np.sum(null >= med_real) + 1) / (n_perm + 1))
    return {"median_alignment_real": med_real, "null_median": float(np.median(null)), "perm_p": p}


def _synth(name, N=CM.N_COHORT_B_PLANNED, d=8, signal=0.0):
    """Synthetic descriptors + geometry. `signal`=0 -> non-predictive (negative control); >0 -> partially predictive."""
    r = CM.rng("M2|synth|%s|%.2f" % (name, signal))
    D = r.standard_normal((N, d))
    Wtrue = r.standard_normal((d, GEOMETRY_SUMMARY_DIM))
    G = signal * (D @ Wtrue) + (1 - signal) * r.standard_normal((N, GEOMETRY_SUMMARY_DIM))
    return D, G


def simulate(n=CM.N_COHORT_B_PLANNED):
    """Engineering simulation. Negative control (no signal) must NOT beat null; planted signal recovers. NOT
    scientific evidence."""
    out = {"moonshot": "M2", "engineering_simulation_only": True, "n": n, "protocols": {}}
    audit = identifiability_audit(n, 8)
    out["identifiability_audit"] = audit
    for proto in PROTOCOLS:
        Dn, Gn = _synth("neg", n, signal=0.0)
        Ds, Gs = _synth("pos", n, signal=0.7)
        neg = permutation_null(Dn, Gn, proto, n_perm=500, name="neg")
        pos = permutation_null(Ds, Gs, proto, n_perm=500, name="pos")
        out["protocols"][proto] = {"negative_control_perm_p": neg["perm_p"],
                                   "planted_signal_perm_p": pos["perm_p"],
                                   "neg_control_not_significant": neg["perm_p"] > 0.05,
                                   "planted_signal_detected": pos["perm_p"] <= 0.05}
    out["caption"] = "synthetic engineering check; NOT discovered/supported/validated; N~12 power is very limited"
    return out


def support_rule_text():
    return ("M2 (Cohort B, post-unlock): median participant zero/near-zero-shot alignment exceeds the "
            "permuted-participant null (prospective permutation p, participant unit) with >=75% participants above "
            "the null median, under the identifiability-audited low-capacity model, in BOTH protocols reported "
            "separately. Any pass is a Cohort-B candidate requiring Cohort-C confirmation.")
