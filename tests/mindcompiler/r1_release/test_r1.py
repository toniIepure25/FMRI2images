"""MINDIR-R1 release + hardening tests. Synthetic only; no biological claims; O2.16/M0 immutable."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fmri2img.mindcompiler.prox import (
    release as RL, hardening as HD, errors as ER, governance as GV, cli as CLI,
)
from fmri2img.mindcompiler.prox import __version__


# ---- packaging / version ----
def test_version_and_schemas():
    assert __version__ == "0.1.0-rc1"
    from fmri2img.mindcompiler.prox import SCHEMA_VERSIONS
    assert set(SCHEMA_VERSIONS) >= {"object", "metric", "benchmark", "artifact", "protocol", "provenance"}


def test_cli_entrypoint_importable():
    # console_scripts target resolves
    assert callable(CLI.main)


# ---- one-command demo ----
def test_demo_runs_cpu():
    out = RL.demo()
    r = out["report"]
    assert r["synthetic_only"] and r["no_external_data"]
    assert 0.0 <= r["support_fraction_median"] <= 1.0
    assert 0.0 <= r["private_correction_overlap_median"] <= 1.0
    assert r["falsification_F5_rotation_invariant"] is True
    assert "provenance" in out and out["provenance"]["frozen"] is False


# ---- reproduce classes never conflated ----
def test_reproduce_classes():
    assert RL.reproduce({})["reproduction_class"] == "REPRODUCTION_UNAVAILABLE"
    method = RL.reproduce({"code_commit": "abc", "config_hash": "h", "data_hashes": {"a": "1"},
                           "metric_version": "prox-metric/1.0.0"})
    assert method["reproduction_class"] == "METHOD_REPRODUCTION"
    exact = RL.reproduce({"code_commit": "abc", "config_hash": "h", "data_hashes": {"a": "1"},
                          "metric_version": "prox-metric/1.0.0", "seed": 1, "bitwise_expected": True})
    assert exact["reproduction_class"] == "EXACT_REPRODUCTION"


# ---- provenance graph ----
def test_provenance_graph_complete():
    g = RL.provenance_graph()
    assert g["nodes"][0] == "DATA" and g["nodes"][-1] == "MANUSCRIPT_FIGURE_TABLE"
    assert len(g["edges"]) == len(g["nodes"]) - 1 and g["every_artifact_traceable_backwards"]


# ---- data contract errors are actionable ----
def test_data_contract_errors():
    with pytest.raises(ER.ShapeError):
        RL.validate_target_state(np.zeros(5), 2)
    with pytest.raises(ER.RankDeficiencyError):
        RL.validate_target_state(np.ones((6, 8)), 3)   # rank 1 < 3
    assert RL.validate_target_state(np.random.default_rng(0).standard_normal((8, 8)), 2)


# ---- numerical backend cross-check ----
def test_numerical_crosscheck_agrees():
    r = HD.numerical_crosscheck()
    assert r["agree"] and r["max_abs_dev_vs_reference"] < 1e-8


# ---- precision audit ----
def test_precision_audit_small_dev():
    r = HD.precision_audit()
    assert r["default_precision"] == "float64" and r["max_f32_f64_dev"] < 1e-2


# ---- determinism ----
def test_determinism():
    d = HD.determinism_check()
    assert d["same_seed_bitwise_equal"] and d["different_seed_differs"]


# ---- adversarial worlds reported honestly ----
def test_adversarial_report_reports_failures():
    r = HD.adversarial_report()
    assert any(row["degraded"] for row in r["rows"])   # some hard worlds must degrade (failures not hidden)


# ---- bug injection: all caught ----
def test_bug_injection_all_caught():
    r = HD.bug_injection_report()
    assert r["all_bugs_caught"], r["results"]


# ---- adaptive benchmark ----
def test_adaptive_benchmark():
    r = HD.adaptive_benchmark()
    assert r["median_savings_stability"] >= 0 and r["stability_beats_random_regret"]


# ---- zero-shot stress: safe under null, detects strong ----
def test_zero_shot_stress():
    r = HD.zero_shot_stress()
    assert r["safe_under_null"] and r["detects_strong"]


# ---- cross-state stress: discriminates full sharing ----
def test_cross_state_stress():
    r = HD.cross_state_stress()
    assert r["full_shared"]["discriminates"]


# ---- sensitivity: effect=0 ~ alpha false-positive ----
def test_sensitivity_false_positive_rate():
    s = HD.sensitivity_analysis(effects=(0.0,), ns=(12,), n_rep=300)
    fp = s["rows"][0]["detection_freq"]
    assert fp <= 0.15   # one-sided alpha=0.05 with MC noise; must not be wildly inflated


def test_sensitivity_monotone_in_effect():
    s = HD.sensitivity_analysis(effects=(0.0, 0.2), ns=(16,), n_rep=200)
    d0 = [r for r in s["rows"] if r["effect"] == 0.0][0]["detection_freq"]
    d2 = [r for r in s["rows"] if r["effect"] == 0.2][0]["detection_freq"]
    assert d2 >= d0


# ---- cohort-C planner ----
def test_cohort_c_plan():
    p = HD.cohort_c_plan(candidate_effect=0.3, metric="subspace_overlap", minimum_meaningful_effect=0.2)
    assert "recommended_min_n" in p and set(["candidate effect definition", "metric", "null"]).issubset(set(p["requires"]))


# ---- immutability / firewall ----
def test_scientific_immutability_and_firewall():
    rep = GV.scientific_immutability(".")
    assert rep["all_match"] and rep["O3"] == "O3_NOT_READY"
    with pytest.raises(GV.HistoricalN8Firewall):
        GV.assert_no_historical_access("open trial_" + "native")


def test_no_biological_claim_language():
    for f in Path("src/fmri2img/mindcompiler/prox").glob("*.py"):
        t = f.read_text().lower()
        assert "biological discovery" not in t
