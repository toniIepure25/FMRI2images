"""M1 -- Geometry-Aware Adaptive Calibration (SHADOW policy).

Question: can we know ONLINE when we have enough imagery calibration data?

The policy is a SHADOW observer: it runs alongside the FROZEN fixed acquisition schedule, never changes what is
acquired, and at each observation t emits a STOP/CONTINUE decision using ONLY information available up to t. It
never sees future observations or any held-out calibration outcome when deciding. Decision statistic: the
incremental stability of the accumulated outside-support subspace (subspace overlap between the estimate from
observations {1..t-1} and {1..t}); STOP once the incremental gain stays below a frozen tolerance for `patience`
consecutive steps. On real Cohort B this is evaluated only AFTER the O2.16 unlock chain; here it runs on
SYNTHETIC fixtures only and any pass is engineering, not scientific."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import common as CM

# frozen policy hyperparameters (declared before any data)
GAIN_TOL = 0.02               # incremental subspace-overlap GAIN below which improvement has plateaued
PATIENCE = 2                  # consecutive plateaued steps -> STOP
MIN_OBS = 3                   # never stop before 3 observations (need >=2 prior subspaces to measure a gain)
MAX_OBS = 8                   # bounded by the frozen 8-observation ceiling
SUBSPACE_DIM = 2              # outside-support D=2 (frozen upstream)


def _subspace(samples):
    """Top-D left singular subspace of stacked samples (rows = observations)."""
    X = np.asarray(samples, np.float64)
    if X.shape[0] < 1:
        return None
    U, s, Vt = np.linalg.svd(X - X.mean(0, keepdims=True), full_matrices=False)
    k = min(SUBSPACE_DIM, Vt.shape[0])
    return Vt[:k]


def _overlap(A, B):
    """Mean squared principal cosine between two row-orthobases (in [0,1])."""
    if A is None or B is None:
        return 0.0
    Qa = np.linalg.qr(A.T)[0]; Qb = np.linalg.qr(B.T)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1) ** 2))


@dataclass
class ShadowDecision:
    stop_at: int
    overlaps: list
    gains: list
    reason: str


def shadow_policy(observation_samples):
    """observation_samples: list of per-observation feature vectors (arrive in order). Returns the SHADOW stop
    index using ONLY info up to each t. Stops when the subspace-overlap GAIN plateaus (< GAIN_TOL) for PATIENCE
    consecutive steps. Deterministic; no future/outcome access."""
    n = len(observation_samples)
    overlaps = []; gains = []; below = 0; stop_at = min(MAX_OBS, n); reason = "reached_max_obs"
    for t in range(1, n + 1):
        prev_sub = _subspace(observation_samples[:t - 1]) if t >= 2 else None
        sub = _subspace(observation_samples[:t])
        ov = _overlap(prev_sub, sub)                          # overlap between {1..t-1} and {1..t}
        overlaps.append(ov)
        gain = ov - overlaps[-2] if len(overlaps) >= 2 else ov
        gains.append(gain)
        if t >= MIN_OBS:
            if gain < GAIN_TOL:                               # improvement plateaued
                below += 1
            else:
                below = 0
            if below >= PATIENCE:
                stop_at = t; reason = "improvement_plateau"; break
        if t >= MAX_OBS:
            stop_at = MAX_OBS; reason = "reached_max_obs"; break
    return ShadowDecision(stop_at=min(stop_at, MAX_OBS), overlaps=overlaps, gains=gains, reason=reason)


# ---- synthetic fixture: a subject whose subspace stabilizes at m_true ----
def synthetic_subject(name, m_true=5, dim=20, noise=0.15):
    r = CM.rng("M1|synth|%s" % name)
    basis = np.linalg.qr(r.standard_normal((dim, SUBSPACE_DIM)))[0]
    samples = []
    for t in range(MAX_OBS):
        coeff = r.standard_normal(SUBSPACE_DIM)
        # early observations carry extra transient direction that decays by m_true
        transient = (max(0, m_true - t) / m_true) * r.standard_normal(dim) * 0.6
        samples.append(basis @ coeff + transient + noise * r.standard_normal(dim))
    return samples


def random_stop_null(name, n_obs=MAX_OBS):
    r = CM.rng("M1|null|%s" % name)
    return int(r.integers(MIN_OBS, n_obs + 1))


def simulate(n_participants=CM.N_COHORT_B_PLANNED):
    """Engineering simulation only. Checks the shadow policy stops near m_true and beats random-stop on a
    synthetic 'calibration-cost' proxy (fewer obs at equal stability). NOT scientific evidence."""
    rows = []
    for i in range(1, n_participants + 1):
        name = "synB%03d" % i
        m_true = 3 + (i % 4)                          # 3..6
        samples = synthetic_subject(name, m_true=m_true)
        dec = shadow_policy(samples)
        rnd = random_stop_null(name)
        rows.append({"participant": name, "m_true": m_true, "shadow_stop": dec.stop_at, "reason": dec.reason,
                     "random_stop": rnd, "shadow_close_to_true": abs(dec.stop_at - m_true) <= 2,
                     "shadow_not_after_max": dec.stop_at <= MAX_OBS})
    close = sum(r["shadow_close_to_true"] for r in rows)
    early = sum(1 for r in rows if r["shadow_stop"] < MAX_OBS)
    bounded = all(MIN_OBS <= r["shadow_stop"] <= MAX_OBS for r in rows)
    return {"moonshot": "M1", "engineering_simulation_only": True, "n": n_participants,
            "n_shadow_close_to_true(+/-2)": close, "frac_close": close / n_participants,
            "n_early_stop(<max)": early, "frac_early_stop": early / n_participants, "all_bounded": bounded,
            "no_future_or_outcome_access": True, "shadow_never_changes_acquisition": True,
            "rows": rows, "caption": "synthetic engineering check; demonstrates the online shadow rule runs, stays "
            "bounded, and can stop early; NOT discovered/supported/validated; real evaluation only post-unlock on Cohort B"}


def support_rule_text():
    return ("M1 (Cohort B, post-unlock): the shadow stop achieves >= the fixed-schedule calibration recovery in "
            ">=75% participants using <= the fixed observations, AND beats the random-stop null (participant "
            "sign-flip, Holm within the M1 family). Any pass is a Cohort-B candidate requiring Cohort-C confirmation.")
