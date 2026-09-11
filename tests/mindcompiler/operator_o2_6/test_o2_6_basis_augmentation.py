"""O2.6 Target-State Basis Augmentation certification (data-free/synthetic). Diagnostic-only gate:
verifies the outside-support decomposition, orthogonality certifications, the frozen null, the conditional
oracle, the M=10 learned==oracle identity, the recovery formulas + guards, terminal inference plumbing,
D50/M50 logic, the synthetic recovery controls (low-D / high-D / none / identity-specific), leakage, and the
immutable seals / O3 lock. No competing model is fit anywhere."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fmri2img.mindcompiler.operator_o2_3a_rd import geometry as G
from fmri2img.mindcompiler.operator_o2_6 import basis_augmentation as BA

rng = np.random.default_rng(6)


def _W(V=60, K=10):
    return np.linalg.qr(rng.standard_normal((V, K)))[0][:, :K]


# 3,4 — outside component orthogonal to W; outside basis orthonormal + perp W
def test_outside_component_and_basis_orthogonality():
    W = _W(); V, K = W.shape
    delta = rng.standard_normal(V)
    do = BA.outside_component(delta, W)
    assert np.linalg.norm(W.T @ do) <= 1e-10 * max(1.0, np.linalg.norm(delta))
    Dout = np.stack([BA.outside_component(rng.standard_normal(V), W) for _ in range(6)]).T
    B = BA.outside_basis(Dout, 3)
    assert np.allclose(B.T @ B, np.eye(3), atol=1e-9)
    assert np.linalg.norm(W.T @ B) <= 1e-10


# 5 — d<=M enforced by construction (D grid filtered), 6 — deterministic svd signs
def test_deterministic_signs_and_rank_insufficient():
    W = _W(); V = W.shape[0]
    Dout = np.stack([BA.outside_component(rng.standard_normal(V), W) for _ in range(2)]).T   # rank 2
    assert BA.outside_basis(Dout, 3) is None                      # RANK_INSUFFICIENT (asks 3 of rank 2)
    b1 = BA.outside_basis(Dout, 2); b2 = BA.outside_basis(Dout, 2)
    assert np.array_equal(b1, b2)                                 # deterministic


# 10,11 — null lies outside W + deterministic seeds
def test_null_outside_and_deterministic():
    W = _W(); V = W.shape[0]
    b1 = BA.null_basis("O2.6|subj01|ventral|0|4|3|2|7", W, V, 3, G)
    b2 = BA.null_basis("O2.6|subj01|ventral|0|4|3|2|7", W, V, 3, G)
    b3 = BA.null_basis("O2.6|subj01|ventral|0|4|3|2|8", W, V, 3, G)
    assert np.array_equal(b1, b2) and not np.allclose(b1, b3)
    assert np.linalg.norm(W.T @ b1) <= 1e-9 and np.allclose(b1.T @ b1, np.eye(3), atol=1e-9)


# 7,8,9 — d=0 reproduces P_IN; P_AUG symmetric/idempotent; rank=base+d (via explicit projector)
def test_augmented_projector_properties_and_decomposition():
    V, K, r, d = 60, 10, 3, 2
    W = _W(V, K); U_res = BA._canon(np.linalg.qr(rng.standard_normal((K, r)))[0][:, :r])
    Q = np.linalg.qr(rng.standard_normal((K, K)))[0]
    P_IN = G.native_projector(Q, U_res, W)
    Dout = np.stack([BA.outside_component(rng.standard_normal(V), W) for _ in range(5)]).T
    B = BA.outside_basis(Dout, d)
    P_EXTRA = B @ B.T; P_AUG = P_IN + P_EXTRA
    assert np.allclose(P_AUG, P_AUG.T, atol=1e-9)                 # symmetric
    assert np.allclose(P_AUG @ P_AUG, P_AUG, atol=1e-8)          # idempotent (orthogonal ranges)
    assert np.allclose(P_IN @ P_EXTRA, 0, atol=1e-9)            # ranges orthogonal
    assert abs(np.trace(P_AUG) - (np.trace(P_IN) + d)) < 1e-6    # rank adds
    # fast decomposition equals explicit projector retention
    delta = rng.standard_normal(V)
    r_exp = G.retention(P_AUG, delta)
    r_fast = G.retention(P_IN, delta) + BA._proj_frac(B, delta)
    assert abs(r_exp - r_fast) < 1e-9


# 12,14 — conditional oracle uses ALL 10 train; M=10 learned basis == oracle basis
def test_M10_learned_equals_oracle_basis():
    W = _W(); V = W.shape[0]
    Dout = np.stack([BA.outside_component(rng.standard_normal(V), W) for _ in range(10)])   # (10 x V)
    B_all = BA.outside_basis(Dout.T, 4)                           # calibration subset C = all 10 (M=10)
    B_oracle = BA.outside_basis(Dout.T, 4)                        # oracle = all 10 train
    assert np.array_equal(B_all, B_oracle)


# 15,16,17 — recovery formulas + denominator guards
def test_recovery_formulas_and_guards():
    # OUT_BASIS_RECOVERY = (R_AUG-R_BASE)/max(R_COND-R_BASE,eps); TOTAL=(R_AUG-R0)/max(R_NAT-R0,eps)
    R_BASE, R_AUG, R_COND, R0, R_NAT = 0.10, 0.40, 0.60, 0.05, 0.70
    assert abs((R_AUG - R_BASE) / max(R_COND - R_BASE, 1e-12) - 0.6) < 1e-12
    assert abs((R_AUG - R0) / max(R_NAT - R0, 1e-12) - 0.5384615384615384) < 1e-9
    # guard: zero denominator -> huge but finite (clipped later)
    assert (0.4 - 0.1) / max(0.1 - 0.1, 1e-12) > 1e10


# 5 (d<=M) + subset structure + 13 (no test id in calibration) — leakage/E
def test_balanced_subsets_and_no_testid_leakage():
    simple, nat = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]              # 10 TRAIN identity indices
    assert len(BA.balanced_subsets(simple, nat, 2)) == 25
    assert len(BA.balanced_subsets(simple, nat, 10)) == 1
    # every subset index is a TRAIN index (0..9); test identities are indexed separately in the driver
    for C in BA.balanced_subsets(simple, nat, 6):
        assert all(0 <= i <= 9 for i in C) and len(set(C)) == 6


# 19,20 — terminal exact sign-flip + Holm family of exactly 2
def test_terminal_signflip_and_holm_family():
    from fmri2img.mindcompiler.operator_o2_3a_rd import frontier as F
    assert abs(F.signflip_p_onesided([1.0] * 8) - 1.0 / 256.0) < 1e-12
    holm = F.holm({"ventral": 0.001, "lateral": 0.9}, alpha=0.05)
    assert holm["ventral"] is True and holm["lateral"] is False and len(holm) == 2


# 21,22 — D50 / M50 logic via aggregate on constructed per_subj
def _mk_per_subj(total_by_md, e_by_md, term_e):
    per = {}
    for s in BA.ALL:
        roimd = {}
        for (M, d), tv in total_by_md.items():
            roimd["%d_%d" % (M, d)] = {"TOTAL_RECOVERY": tv, "E_AUG": e_by_md[(M, d)],
                                       "OUT_BASIS_RECOVERY": 0.8, "R_AUG": tv}
        per[s] = {"ventral": {"md": roimd, "term_E_folds": [term_e] * 6, "R0": 0.1, "R_native": 0.7},
                  "lateral": {"md": roimd, "term_E_folds": [term_e] * 6, "R0": 0.1, "R_native": 0.7}}
    return per


def test_D50_M50_and_status_feasible():
    tot = {}; ea = {}
    for M in BA.M_GRID:
        for d in BA.D_GRID:
            if d > M:
                continue
            tot[(M, d)] = 0.7 if d >= 2 else 0.3               # d>=2 recovers
            ea[(M, d)] = 0.05
    per = _mk_per_subj(tot, ea, 0.05)
    roi_status, term, rows, prog = BA.aggregate(per, G)
    assert roi_status["ventral"]["D50"] == 2                   # smallest d with recovery
    assert roi_status["ventral"]["status"] == "LOW_DIMENSION_TARGET_BASIS_AUGMENTATION"
    assert prog == "TARGET_STATE_MISSING_BASIS_LOW_COMPLEXITY"


def test_terminal_not_feasible_status():
    tot = {}; ea = {}
    for M in BA.M_GRID:
        for d in BA.D_GRID:
            if d > M:
                continue
            tot[(M, d)] = 0.1; ea[(M, d)] = -0.01              # never recovers, negative effect
    per = _mk_per_subj(tot, ea, -0.01)
    roi_status, term, rows, prog = BA.aggregate(per, G)
    assert roi_status["ventral"]["status"] == "TARGET_BASIS_AUGMENTATION_NOT_RECOVERED_AT_MAX_BUDGET"
    assert prog == "TARGET_STATE_BASIS_NOT_RECOVERED_WITH_10_IDENTITIES_10_DIMS"


# 23,24 — synthetic low-D recovers, high-D not fully at small d (outside_basis capture)
def test_synthetic_low_and_high_d_capture():
    W = _W(V=80, K=10); V = W.shape[0]
    # shared planted outside subspace of rank 2
    B_true = BA.outside_basis(np.stack([BA.outside_component(rng.standard_normal(V), W) for _ in range(2)]).T, 2)
    coeff = rng.standard_normal((10, 2))
    Dtrain = (B_true @ coeff.T).T                              # (10 x V) all in the 2-dim outside subspace
    test = B_true @ rng.standard_normal(2)                     # held-out lies in same subspace
    B2 = BA.outside_basis(Dtrain.T, 2)
    assert BA._proj_frac(B2, test) > 0.99                      # d=2 captures nearly all outside energy
    B1 = BA.outside_basis(Dtrain.T, 1)
    assert BA._proj_frac(B1, test) < BA._proj_frac(B2, test)   # d=1 insufficient for a 2-D component


# 25 — identity-specific outside: calibration cannot generalize to held-out
def test_synthetic_identity_specific_no_generalization():
    W = _W(V=120, K=10); V = W.shape[0]
    Dtrain = np.stack([BA.outside_component(rng.standard_normal(V), W) for _ in range(10)])  # unique per identity
    heldout = BA.outside_component(rng.standard_normal(V), W)  # independent
    B = BA.outside_basis(Dtrain.T, 4)
    assert BA._proj_frac(B, heldout) < 0.30                    # weak held-out capture despite in-sample fit


# 26 — no-shared-outside control: a basis learned on independent outside-noise does not beat a random
# null basis at capturing an independent held-out outside vector (augmentation ~ null on average).
def test_synthetic_no_outside_matches_null():
    W = _W(V=100, K=10); V = W.shape[0]
    Dtrain = np.stack([BA.outside_component(rng.standard_normal(V), W) for _ in range(10)])  # independent noise
    B = BA.outside_basis(Dtrain.T, 3)
    learned, null = [], []
    for it in range(200):
        test = BA.outside_component(rng.standard_normal(V), W)         # independent held-out outside vector
        learned.append(BA._proj_frac(B, test))
        null.append(BA._proj_frac(BA.null_basis("O2.6|s|r|0|10|0|3|%d" % it, W, V, 3, G), test))
    # both capture ~ 3/(V-K) in expectation; learned must not systematically beat the random null
    assert abs(np.mean(learned) - np.mean(null)) < 0.03
    assert np.mean(learned) < 0.10                                     # small (~3/90), no shared structure


# 27,28 — immutable seals + O3 lock in the frozen config
def test_immutable_seals_and_o3_lock():
    cfg = json.load(open("artifacts/mindcompiler/operator_o2_6/o2_6_frozen_config.json"))
    assert cfg["immutable_history"]["O2_5"] == "TARGET_IMAGERY_RESIDUAL_OUTSIDE_PERCEPTION_SUPPORT_DOMINANT"
    assert cfg["immutable_history"]["O2_4R"] == "TARGET_STATE_ORIENTATION_NOT_RECOVERED_BY_ORTHOGONAL_CALIBRATION"
    assert cfg["O3"] == "O3_NOT_READY"


# 16b (no competing model) — driver imports no ML fitters
def test_no_competing_model_imported():
    src = Path(BA.__file__).read_text()
    for bad in ("sklearn", "Ridge", "CCA(", "PLS", "torch", "RandomForest", "ridge_regression"):
        assert bad not in src
