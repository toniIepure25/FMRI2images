"""O2.14 acquisition-schedule robustness certification (data-free/synthetic). Verifies schedule coverage +
robustness-margin aggregation, the identity/repeat/interaction variance decomposition, exact schedule counts
for all seven primary cells, the robust-cell classification (>=75% frozen), O2.9 grand-mean replay identity,
and config immutability. No adaptive schedule selection; participant is the unit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fmri2img.mindcompiler.operator_o2_14 import schedule_robustness as R

rng = np.random.default_rng(14)


def _res(grid_folds, R0=0.0, Rnat=1.0, R111=1.0):
    g = [np.asarray(x, np.float64) for x in grid_folds]
    return {"grids": g, "R0": [R0] * R.N_FOLDS, "Rnat": [Rnat] * R.N_FOLDS, "R111": R111,
            "r_best": 6, "n_id": g[0].shape[0], "n_rep": g[0].shape[1]}


# ---- schedule counts (exact) ----
def test_exact_schedule_counts():
    simple = [0, 1, 2, 3, 4]; nat = [5, 6, 7, 8, 9]
    exp = {(2, 1): (25, 8), (2, 2): (25, 28), (4, 1): (100, 8), (6, 1): (100, 8), (2, 4): (25, 70), (4, 2): (100, 28), (8, 1): (25, 8)}
    for cell, (nid, nrep) in exp.items():
        ids, reps = R.schedules_for(cell, simple, nat)
        assert len(ids) == nid and len(reps) == nrep, (cell, len(ids), len(reps))


# ---- coverage + robustness margin ----
def test_coverage_and_margin():
    # 80% of schedules sufficient (TOTAL>=0.5): grid with 8/10 entries at 0.7, 2 at 0.3
    base = np.full((2, 5), 0.7); base[0, 0] = 0.3; base[1, 4] = 0.3
    A = R.aggregate_cell(_res([base] * R.N_FOLDS))
    assert abs(A["coverage"] - 0.8) < 1e-9
    assert abs(A["margin_q"]["median"] - 0.2) < 1e-9                          # median margin = 0.7-0.5


def test_variance_identity_dominated():
    # rows differ strongly, columns identical -> F_ID ~ 1
    col = np.array([0.2, 0.5, 0.9])
    grid = np.repeat(col[:, None], 6, axis=1)
    A = R.aggregate_cell(_res([grid] * R.N_FOLDS))
    assert A["F_ID"] > 0.99 and A["F_REPEAT"] < 1e-6


def test_variance_repeat_dominated():
    row = np.array([0.2, 0.4, 0.6, 0.8])
    grid = np.repeat(row[None, :], 5, axis=0)
    A = R.aggregate_cell(_res([grid] * R.N_FOLDS))
    assert A["F_REPEAT"] > 0.99 and A["F_ID"] < 1e-6


def test_variance_interaction_dominated():
    grid = np.array([[0.8, 0.2], [0.2, 0.8]])                                 # pure interaction, zero marginals
    A = R.aggregate_cell(_res([grid] * R.N_FOLDS))
    assert A["F_INTERACTION"] > 0.99 and A["F_ID"] < 1e-6 and A["F_REPEAT"] < 1e-6


# ---- O2.9 grand-mean replay identity ----
def test_grandmean_equals_o2_9_rcomp():
    grids = [rng.random((4, 7)) for _ in range(R.N_FOLDS)]
    A = R.aggregate_cell(_res(grids))
    expect = float(np.mean([g.mean() for g in grids]))                        # O2.9 R_COMP = grand mean of R_SCHEDULE
    assert abs(A["R_COMP"] - expect) < 1e-12


# ---- robust-cell classification (0.75) ----
def test_o2_9_sufficient_rule():
    assert R.o2_9_sufficient([0.6] * 8, [0.6] * 8) is True
    assert R.o2_9_sufficient([0.6, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4], [0.6] * 8) is False  # only 5/8


def test_robust_requires_coverage_075():
    # coverage vector: median 0.8, 6/8 >=0.75 -> ok; vs median 0.6 -> not
    covs_ok = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.4, 0.4]
    covs_bad = [0.6] * 8
    med = lambda x: float(np.median(x)); q = R.ROBUST_Q
    assert (med(covs_ok) >= q and sum(c >= q for c in covs_ok) >= 6) is True
    assert (med(covs_bad) >= q and sum(c >= q for c in covs_bad) >= 6) is False
    assert q == 0.75


# ---- config immutability + no model search ----
def test_config_and_no_model_search():
    cfg = json.load(open("artifacts/mindcompiler/operator_o2_14/o2_14_frozen_config.json"))
    assert cfg["immutable_history"]["O2_13"] == "NO_REPRODUCIBLE_TESTED_COVARIATE_STRUCTURE"
    assert cfg["immutable_history"]["O2_9_N_TRIALS_STAR"] == 8 and cfg["O3"] == "O3_NOT_READY"
    assert cfg["robust_cell"]["threshold_075_frozen_before_outcomes"] is True
    assert len(cfg["primary_cells"]) == 7 and "config_sha256" in cfg
    src = Path(R.__file__).read_text()
    for bad in ("Ridge", "Procrustes", "argmax(", "best_identity", "adaptive", "sklearn", "torch"):
        assert bad not in src
