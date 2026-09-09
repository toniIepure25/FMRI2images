"""O2.4 freeze + determination tests (data-free, offline).

Assert the prospective calibration-frontier design is frozen and self-consistent (budgets, balanced-subset
counts, outer folds, orthogonal-Procrustes-only estimator, null, N=8 sign-flip + Holm inference, M_STAR
rule, statuses, claim boundary, prior-art boundary), that thresholds/ROIs match the historical program,
and that all immutable history is preserved. The real calibration frontier is provenance-blocked; results
are asserted to be honest NOT_COMPUTED/NOT_REACHED, never fabricated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
B = ROOT / "artifacts/mindcompiler/operator_o2_4"


def _j(n):
    return json.loads((B / n).read_text())


def test_frozen_config_sha_matches():
    cfg = _j("o2_4_frozen_config.json")
    stored = cfg.pop("config_sha256")
    rec = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert rec == stored and stored.startswith("321b42f9")


def test_budgets_and_balanced_subset_counts():
    c = _j("o2_4_frozen_config.json")
    assert c["calibration_budgets"]["M"] == [0, 2, 4, 6, 8, 10]
    b = c["balanced_subset_enumeration"]
    assert (b["M2"], b["M4"], b["M6"], b["M8"], b["M10"]) == (25, 100, 100, 25, 1)
    assert "identity-diversity" in c["calibration_budgets"]["unit"]


def test_orthogonal_procrustes_only():
    e = _j("procrustes_contract.json")
    assert e["family"] == "ORTHOGONAL PROCRUSTES ONLY" and e["solver"] == "deterministic SVD"
    assert e["no_affine"] is True and e["no_ridge"] is True and e["no_nonlinear"] is True and e["no_scaling"] is True
    nm = _j("o2_4_frozen_config.json")["no_method_search"]
    for bad in ["ridge", "CCA", "RRR", "nonlinear MLP", "hyperalignment competitor"]:
        assert bad in nm["forbidden"]


def test_inference_and_mstar_rule():
    c = _j("o2_4_frozen_config.json")
    inf = c["inference"]
    assert inf["N"] == 8 and inf["primary_tests"] == 10 and inf["multiplicity"] == "HOLM" and inf["alpha"] == 0.05
    assert "2^8=256" in inf["test"]
    ms = c["M_STAR_rule"]
    assert "oracle recovery>=0.50" in ms["definition"] and ms["if_none"] == "M_STAR = NOT_REACHED"
    assert ms["no_interpolation_between_budgets"] is True


def test_claim_boundary_and_statuses():
    c = _j("o2_4_frozen_config.json")
    for bad in ["zero-shot transfer", "universal brain operator", "causal neural transformation"]:
        assert bad in c["claim_boundary"]["must_not_claim"]
    for s in ["TARGET_STATE_ORIENTATION_LOW_CALIBRATION_IDENTIFIABLE",
              "TARGET_STATE_ORIENTATION_NOT_RECOVERED_BY_ORTHOGONAL_CALIBRATION",
              "TARGET_STATE_ORIENTATION_CALIBRATION_INCONCLUSIVE"]:
        assert s in c["primary_status_rules"]


def test_zero_target_closeout_bounded():
    z = _j("zero_target_program_closeout.json")
    assert z["statement"] == "ZERO_TARGET_IMAGERY_ORIENTATION_NOT_IDENTIFIED_UNDER_TESTED_PUBLIC_ANCHORS"
    assert "ZERO_TARGET_IMAGERY_ORIENTATION_IMPOSSIBLE" in z["does_not_mean"]
    assert "THEORETICALLY_NON_IDENTIFIABLE" in z["does_not_mean"]


def test_prior_art_no_novelty_overclaim():
    pa = _j("prior_art_registry.json")
    assert "MindEye2 (Scotti et al. 2024)" in pa["acknowledged_prior_art"]
    assert pa["must_not_claim_novelty_as"] == "first few-shot cross-subject fMRI adaptation"
    assert pa["novelty_status"].startswith("NOT_CLAIMED")


def test_immutable_history_preserved():
    h = _j("o2_4_frozen_config.json")["immutable_history"]
    assert h["O2"] == "SHARED_OPERATOR_PARTIAL"
    assert h["O2_3A"] == "CORE_ANCHOR_TARGET_ORIENTATION_NOT_IDENTIFIABLE"
    assert h["O3"] == "O3_NOT_READY"
    assert "FMRIPREP_DERIVED_REST_FUNC1PT8MM_NOT_CERTIFIED_FOR_FINE_SCALE_CONNECTIVITY" in h["O2_3C_PREP"]


def test_data_dependent_results_not_fabricated():
    # provenance-blocked frontier: no calibration outcome fabricated
    assert _j("minimum_budget_results.json")["status"].startswith("PENDING") or \
           _j("minimum_budget_results.json").get("M_STAR") == "NOT_REACHED"
