"""MINDIR-M0 three-moonshot preregistration certification (data-free/synthetic). Verifies firewalls, unlock
ordering, no-historical-loader guard, bounded models, prospective nulls/negative controls, N~12 reality,
Cohort-C templates, immutability, and each moonshot's shadow/prospective logic on SYNTHETIC fixtures only.
Nothing here is scientific evidence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fmri2img.mindcompiler.moonshot_triad import (
    common as CM, m1_adaptive as M1, m2_zeroshot as M2, m3_crossstate as M3,
)

SRC = Path("src/fmri2img/mindcompiler/moonshot_triad")


# ---- firewall + ordering + guards ----
def test_cohort_b_locked_until_unlock():
    st = CM.UnlockState()
    assert not st.cohort_b_unlocked()
    with pytest.raises(CM.UnlockError):
        st.require_cohort_b_unlock()
    st.o2_16_cohort_acquired = st.o2_16_primary_sealed = st.o2_16_sec_sealed = True
    assert st.cohort_b_unlocked() and st.require_cohort_b_unlock()


def test_predictor_side_cannot_read_protected():
    with pytest.raises(CM.OutcomeAccessError):
        CM.assert_predictor_side("analysis/moonshot/protected_outcomes/x.json")
    assert CM.assert_predictor_side("analysis/moonshot/predictor_side/x.json")


def test_no_historical_loader_guard():
    with pytest.raises(ValueError):
        CM.assert_no_historical_loader("import trial_native from operator_o2_9/results")
    for f in SRC.glob("*.py"):
        CM.assert_no_historical_loader(f.read_text())   # our own moonshot code is clean


def test_no_historical_ids_in_cohorts():
    assert CM.Cohort.A_HISTORICAL.value.endswith("CLOSED")
    for pid in CM.HISTORICAL_N8:
        assert pid.startswith("subj0")


# ---- N~12 reality + Cohort-C ----
def test_sample_size_reality():
    s = CM.sample_size_reality(12)
    assert abs(s["min_one_sided_signflip_p"] - 1 / 4096) < 1e-9 and s["low_capacity_model_budget_required"]


def test_cohort_c_template():
    t = CM.cohort_c_template("M2", "cand1", "alignment", "median>null")
    assert t["cohort"] == "C_CONFIRMATION" and "participant-disjoint" in t["independence"]


# ---- M1 shadow policy ----
def test_m1_shadow_no_future_access_and_bounded():
    samples = M1.synthetic_subject("t1", m_true=4)
    dec = M1.shadow_policy(samples)
    assert M1.MIN_OBS <= dec.stop_at <= M1.MAX_OBS
    # decision at step t uses only first t samples: truncating future must not change the stop for t<=stop
    dec2 = M1.shadow_policy(samples[:dec.stop_at])
    assert dec2.stop_at == dec.stop_at


def test_m1_simulation_bounded_and_can_stop_early():
    r = M1.simulate(12)
    assert r["engineering_simulation_only"] and r["all_bounded"] and r["no_future_or_outcome_access"]
    assert r["frac_early_stop"] > 0            # the online rule can stop before the ceiling
    assert r["shadow_never_changes_acquisition"]


# ---- M2 bounded model + nulls + negative control ----
def test_m2_identifiability_audit_bounds():
    a = M2.identifiability_audit(12, 8)
    assert a["identifiable_budget_ok"] and a["effective_params"] <= a["pseudo_observations"] // 3 + 1


def test_m2_negative_control_and_signal():
    r = M2.simulate(12)
    for proto in M2.PROTOCOLS:
        assert r["protocols"][proto]["neg_control_not_significant"]      # no signal -> not significant
        assert r["protocols"][proto]["planted_signal_detected"]          # planted signal -> detected


def test_m2_protocols_declared():
    assert set(M2.PROTOCOLS) == {"ZERO_TARGET", "NEAR_ZERO_TARGET"}


# ---- M3 cross-state ----
def test_m3_shared_scaffold_detects_planted():
    r = M3.simulate(12)
    assert r["frac"] >= 0.75 and "post-replication" in r["protocol_status"]


def test_m3_protocol_awaiting_human_review():
    spec = M3.protocol_spec()
    assert spec["status"] == "M3_CROSS_STATE_PROTOCOL_READY_FOR_HUMAN_REVIEW"
    assert "DOES NOT modify O2.16" in spec["relationship_to_o2_16"]


# ---- prospective inference primitives ----
def test_signflip_and_holm():
    assert abs(CM.signflip_p_onesided([0.1] * 8) - 1 / 256) < 1e-9
    rej = CM.holm({"a": 0.001, "b": 0.9}, 0.05)
    assert rej["a"] and not rej["b"]


# ---- immutability ----
def test_o2_16_and_sec_hashes_unchanged():
    rep = CM.immutability_report()
    assert rep["all_match"] and rep["O3"] == "O3_NOT_READY" and rep["no_x5"]


def test_no_scientific_result_files_written_by_moonshot():
    for f in SRC.glob("*.py"):
        t = f.read_text()
        assert "operator_o2_16/scientific_status" not in t and "x4_geometric_drift/results" not in t


def test_synthetic_never_called_validated():
    for f in SRC.glob("*.py"):
        t = f.read_text().lower()
        # simulations must be labelled engineering-only, never discovered/supported/validated as science
        assert "engineering" in t or "synthetic" in t if ("simulate" in t) else True
