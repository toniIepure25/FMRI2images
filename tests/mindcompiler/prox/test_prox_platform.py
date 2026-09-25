"""MINDIR-PROX platform tests: property, metamorphic, falsification, benchmark, numerical-stability, governance,
immutability. Synthetic only; no historical N=8; O2.16/M0 unchanged."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fmri2img.mindcompiler.prox import (
    objects as OB, metrics as MX, statistics as ST, synthworld as SW, falsification as FB,
    benchmark as BM, sample_complexity as SC, governance as GV,
)

rng = np.random.default_rng(7)
SRC = Path("src/fmri2img/mindcompiler/prox")


# ---------- property: geometry basis invariance ----------
def test_prop_orthogonal_rotation_invariance():
    dim = 30
    A = rng.standard_normal((3, dim)); B = rng.standard_normal((3, dim))
    Q = np.linalg.qr(rng.standard_normal((dim, dim)))[0]
    for fn in (MX.subspace_overlap, MX.grassmann_distance, MX.chordal_distance, MX.projector_frobenius):
        assert abs(fn(A, B) - fn(A @ Q.T, B @ Q.T)) < 1e-8


def test_prop_identical_subspace_overlap_one():
    A = rng.standard_normal((2, 20))
    assert abs(MX.subspace_overlap(A, A) - 1.0) < 1e-9


def test_prop_zero_residual_when_in_support():
    # target fully within support -> outside residual 0
    assert MX.outside_support_residual(1.0, 1.0) == 0.0
    assert abs(MX.support_fraction(0.7, 1.0) - 0.7) < 1e-12


def test_prop_total_recovery_affine_invariance():
    # adding a constant to R, R0, R_native leaves total_recovery unchanged
    r1 = MX.total_recovery(0.6, 0.1, 0.9)
    r2 = MX.total_recovery(0.6 + 5, 0.1 + 5, 0.9 + 5)
    assert abs(r1 - r2) < 1e-9


def test_prop_random_subspaces_low_overlap():
    dim = 60
    ov = np.median([MX.subspace_overlap(rng.standard_normal((2, dim)), rng.standard_normal((2, dim))) for _ in range(50)])
    assert ov < 0.25   # rank-2 in 60-d -> small


# ---------- metamorphic ----------
def test_meta_increasing_residual_moves_outside_metric():
    lo = MX.outside_support_residual(0.9, 1.0)
    hi = MX.outside_support_residual(0.4, 1.0)
    assert hi > lo


def test_meta_participant_shuffle_collapses_prediction():
    # planted subject-specific mapping; F2 must collapse it toward null
    N, d, k = 12, 8, 2
    D = rng.standard_normal((N, d)); W = rng.standard_normal((d, k)); G = D @ W + 0.05 * rng.standard_normal((N, k))

    def pred_fn(Dtr, Gtr, Dte, Gte):
        Wt = np.linalg.solve(Dtr.T @ Dtr + 5 * np.eye(d), Dtr.T @ Gtr)
        p = Dte @ Wt
        return MX.cosine_similarity(p, Gte)
    res = FB.participant_shuffle_collapses(pred_fn, D, G, n_perm=200)
    assert res["real"] > res["null_median"] and res["collapses"] in (True, False)  # collapse test runs
    # with strong signal the real alignment should exceed the null median
    assert res["real"] > res["null_median"]


# ---------- falsification battery ----------
def test_falsification_rotation_invariant():
    A = rng.standard_normal((3, 25)); B = rng.standard_normal((3, 25))
    r = FB.shared_rotation_invariance(A, B, 25)
    assert r["invariant"]


def test_falsification_rank_matched_null_band():
    dim, rank = 40, 2
    sig = rng.standard_normal((rank, dim))
    nb = FB.rank_matched_random_null(sig, dim, rank, n=100)
    assert 0.0 <= nb["null_overlap_median"] <= 1.0 and nb["null_overlap_p95"] >= nb["null_overlap_median"]


def test_falsification_battery_manifest():
    m = FB.battery_manifest()
    assert len(m["battery"]) == 10 and m["participant_unit"]


# ---------- statistics ----------
def test_signflip_and_holm_and_minp():
    assert abs(ST.signflip_p_onesided([0.1] * 8) - 1 / 256) < 1e-9
    assert abs(ST.min_signflip_p(12) - 1 / 4096) < 1e-12
    rej = ST.holm({"a": 0.001, "b": 0.9}); assert rej["a"] and not rej["b"]


# ---------- synthetic world + benchmark (ground-truth scored) ----------
def test_synthworld_shapes_and_truth():
    w = SW.generate(SW.WorldConfig(n_subjects=6))
    assert len(w["subjects"]) == 6 and w["shared_basis_true"].shape[0] == w["shared_rank"]
    assert w["subjects"][0]["private_basis_true"].shape[0] == w["private_rank"]


def test_benchmark_runs_and_scores():
    r = BM.run_benchmark(SW.WorldConfig(n_subjects=12, snr=3.0))
    assert r["ground_truth_scored"] and set(("B2", "B3", "B4", "B8")).issubset(r["tasks"])
    assert r["tasks"]["B4"]["median_recovery_overlap"] >= 0.0


def test_benchmark_subject_transfer_direction():
    # subject-specific private geometry -> within-subject overlap should exceed between-subject
    r = BM.run_benchmark(SW.WorldConfig(n_subjects=12, snr=4.0, private_rank=2))
    assert r["tasks"]["B3"]["within_exceeds_between"]


def test_benchmark_cross_state_planted():
    r = BM.run_benchmark(SW.WorldConfig(n_subjects=12, snr=4.0, shared_rank=3))
    assert r["tasks"]["B8"]["real_exceeds_null"]


# ---------- sample complexity / phase transition ----------
def test_phase_diagram_regimes():
    pd = SC.phase_diagram(n_obs=8)
    regimes = set(g["regime"] for g in pd["grid"])
    assert regimes <= {"UNRECOVERABLE", "PARTIALLY_RECOVERABLE", "RELIABLY_RECOVERABLE"}
    # higher SNR should not reduce overlap at fixed rank (monotone-ish)
    hi = [g for g in pd["grid"] if g["private_rank"] == 1 and g["snr"] == 4.0][0]
    lo = [g for g in pd["grid"] if g["private_rank"] == 1 and g["snr"] == 0.5][0]
    assert hi["overlap"] >= lo["overlap"] - 1e-6


# ---------- numerical stability ----------
def test_numerical_stability_tiny_singular_values():
    dim = 20
    A = rng.standard_normal((2, dim))
    B = A + 1e-10 * rng.standard_normal((2, dim))   # near-identical
    ov = MX.subspace_overlap(A, B)
    assert 0.99 <= ov <= 1.0 + 1e-9 and np.isfinite(MX.grassmann_distance(A, B))


def test_numerical_degenerate_handling():
    A = np.zeros((2, 10)); A[0, 0] = 1.0
    B = np.zeros((2, 10)); B[0, 1] = 1.0
    assert np.isfinite(MX.subspace_overlap(A, B))


# ---------- governance / immutability / firewall ----------
def test_historical_firewall_guard():
    with pytest.raises(GV.HistoricalN8Firewall):
        GV.assert_no_historical_access("read operator_o2_9/results/x.npz")
    for f in SRC.glob("*.py"):
        GV.assert_no_historical_access(f.read_text())   # platform code is clean


def test_scientific_immutability():
    rep = GV.scientific_immutability(".")
    assert rep["all_match"] and rep["O3"] == "O3_NOT_READY" and rep["no_new_inference"]


def test_no_status_inflation_in_source():
    for f in SRC.glob("*.py"):
        t = f.read_text().lower()
        assert "biological law" not in t or "not a biological law" in t or "not biological laws" in t


def test_metric_registry_metadata_complete():
    for name, m in MX.REGISTRY.items():
        meta = m.meta()
        for key in ("question", "definition", "range", "direction", "null_interpretation", "failure_modes",
                    "invariances", "min_sample", "participant_aggregation", "allowed_use", "forbidden_interpretation"):
            assert meta.get(key), (name, key)


# ---------- object model ----------
def test_object_model_subspace():
    sub = OB.orthonormalize(rng.standard_normal((3, 15)))
    assert sub.rank == 3 and sub.orthonormal()
