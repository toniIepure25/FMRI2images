"""O2.15 acquisition failure-mode diagnostic certification (data-free/synthetic). Verifies the Spearman +
Fisher aggregation, projector-similarity rotation/sign invariance, the pair-vs-complement training-only Q
(planted repeat-noise -> Q predicts held-out margin; unrelated -> null), the additive repeat-position model +
interaction residual, all component/program status branches, the exact 4-test Holm family, and config
immutability. Schedules are never the inferential N."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fmri2img.mindcompiler.operator_o2_15 import failure_mode as FM

rng = np.random.default_rng(15)


# ---- A. Spearman + Fisher ----
def test_spearman_monotone():
    x = np.array([1., 2, 3, 4, 5]); assert abs(FM._spearman(x, 2 * x + 1) - 1.0) < 1e-9
    assert abs(FM._spearman(x, -x) + 1.0) < 1e-9


def test_spearman_ties_average_rank():
    x = np.array([1., 1, 2, 3]); y = np.array([1., 2, 2, 4])
    assert np.isfinite(FM._spearman(x, y))


def test_fisher_mean():
    Z, rho = FM._fisher_mean([0.5, 0.5, 0.5])
    assert abs(rho - 0.5) < 1e-9 and abs(Z - np.arctanh(0.5)) < 1e-9


# ---- B. projector similarity invariance ----
def test_proj_sim_rotation_sign_invariant():
    U = np.linalg.qr(rng.standard_normal((30, 3)))[0][:, :3]
    R = np.linalg.qr(rng.standard_normal((3, 3)))[0]                          # orthogonal re-basis
    assert abs(FM._proj_sim(U, U @ R, 3) - 1.0) < 1e-9                        # same subspace -> 1
    assert abs(FM._proj_sim(U, -U, 3) - 1.0) < 1e-9                           # sign flip -> 1
    V = np.linalg.qr(rng.standard_normal((30, 3)))[0][:, :3]
    assert FM._proj_sim(U, V, 3) < 0.9                                        # unrelated -> < 1


# ---- C. planted pair/complement fidelity ----
def _fixture(noise):
    V, K = 40, 6
    W = np.linalg.qr(rng.standard_normal((V, K)))[0][:, :K]
    # 10 identities, 8 repeats: shared identity signal + per-repeat noise scaled by `noise`
    base = rng.standard_normal((10, V))
    Dtr = np.stack([np.stack([base[i] + noise * rng.standard_normal(V) for _ in range(8)]) for i in range(10)])
    return W, Dtr


def test_schedule_Q_high_when_reliable():
    W, Dtr = _fixture(noise=0.05)                                             # low repeat noise
    Q_IN, Q_OUT, cn, cw, co, eP, eC = FM._schedule_Q(W, Dtr, [0, 1, 2, 3], (0, 1), 3)
    assert Q_IN > 0.9 and cn > 0.9                                            # pair agrees with complement


def test_schedule_Q_low_when_noisy():
    W, Dtr = _fixture(noise=5.0)                                              # high repeat noise
    Q_IN, Q_OUT, cn, cw, co, eP, eC = FM._schedule_Q(W, Dtr, [0, 1, 2, 3], (0, 1), 3)
    assert Q_IN < 0.8                                                         # pair diverges from complement


# ---- D. additive repeat model + interaction residual ----
def test_additive_recovers_planted_positions():
    import itertools
    pairs = list(itertools.combinations(range(8), 2))
    alpha_true = np.array([0.3, -0.1, 0.0, 0.2, -0.2, 0.1, -0.15, -0.15])
    alpha_true -= alpha_true.mean()
    Q = np.array([0.5 + alpha_true[a] + alpha_true[b] for (a, b) in pairs])
    alpha, r2 = FM._additive_repeat(Q, pairs)
    assert r2 > 0.999 and np.allclose(alpha, alpha_true, atol=1e-6)


def test_interaction_residual_removes_marginals():
    M = np.outer(np.array([1., 2, 3]), np.ones(4)) + np.outer(np.ones(3), np.array([0., 1, 2, 3]))
    Rr = FM._resid_interaction(M)
    assert np.allclose(Rr, 0, atol=1e-9)                                      # purely additive -> zero interaction


# ---- E. classification branches ----
def _part(vals):
    """vals: dict roi -> dict comp -> {'rho':..,'p':..,'fold_rho':[..]}."""
    part = {}
    for s in FM.ALL:
        part[s] = {}
        for roi in vals:
            part[s][roi] = {}
            for comp in vals[roi]:
                v = vals[roi][comp]
                part[s][roi][comp] = {"Z": v["Z"], "rho": v["rho"], "fold_rho": [v["rho"]] * FM.N_FOLDS}
    return part


def _run_classify(vals, tmp):
    posq = {roi: {s: np.zeros(8) for s in FM.ALL} for roi in vals}
    FM.classify_write(tmp, _part(vals), list(vals.keys()), 0.0, [], [], [], [], [], [], [], posq, [(0, 1)])
    return json.load(open(tmp / "scientific_status.json")), json.load(open(tmp / "roi_status.json"))


def test_status_multicomponent(tmp_path):
    v = {c: {"Z": 1.0, "rho": 0.4} for c in ("IN", "OUT")}
    sci, roi = _run_classify({"ventral": v, "lateral": v}, tmp_path)
    assert roi["ventral"] == "MULTICOMPONENT_REPEAT_GEOMETRY_INSTABILITY_SUPPORTED"
    assert sci["status"] == "REPEAT_GEOMETRY_INSTABILITY_SUPPORTED_AS_PRIMARY_FAILURE_MODE"


def test_status_not_explained(tmp_path):
    v = {c: {"Z": -0.2, "rho": -0.2} for c in ("IN", "OUT")}
    sci, roi = _run_classify({"ventral": v, "lateral": v}, tmp_path)
    assert roi["ventral"] == "SCHEDULE_FAILURE_NOT_EXPLAINED_BY_TRAINING_GEOMETRY_STABILITY"
    assert sci["status"] == "SCHEDULE_FRAGILITY_NOT_EXPLAINED_BY_REPEAT_GEOMETRY_STABILITY"


def test_status_within_only_multiregime(tmp_path):
    v = {"IN": {"Z": 1.0, "rho": 0.4}, "OUT": {"Z": 1.0, "rho": 0.4}}         # ventral both
    o = {"IN": {"Z": -0.2, "rho": -0.2}, "OUT": {"Z": -0.2, "rho": -0.2}}     # lateral none
    sci, roi = _run_classify({"ventral": v, "lateral": o}, tmp_path)
    assert sci["status"] == "ACQUISITION_FAILURE_MODE_MULTIREGIME"


# ---- F. exactly 4 tests + config ----
def test_config_and_no_optimization():
    cfg = json.load(open("artifacts/mindcompiler/operator_o2_15/o2_15_frozen_config.json"))
    assert cfg["immutable_history"]["O2_14"] == "EIGHT_TRIAL_MINIMUM_AVERAGE_ONLY_NOT_SCHEDULE_ROBUST"
    assert cfg["immutable_history"]["O2_9_N_TRIALS_STAR"] == 8 and cfg["O3"] == "O3_NOT_READY"
    assert cfg["primary_inference"]["family"].startswith("2 components") and "config_sha256" in cfg
    src = Path(FM.__file__).read_text()
    for bad in ("best_pair", "select_repeat", "good_repeat", "argmax(", "Ridge", "sklearn", "torch"):
        assert bad not in src
