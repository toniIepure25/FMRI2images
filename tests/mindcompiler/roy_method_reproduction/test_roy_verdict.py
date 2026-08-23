"""S2.7R data-free audit tests: verdict logic + artifact schema/consistency guards.

No modelling; validates the frozen decision function and the committed S2.7R artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from fmri2img.mindcompiler.roy_method_reproduction import roy_verdict as rv

ART = Path(__file__).resolve().parents[3] / "artifacts/mindcompiler/roy_s2_7r"


def _j(name):
    return json.loads((ART / name).read_text())


# ------------------------------------------------------------------ verdict logic (Parts H-K)
def _central_ok():
    return {"P-C1": "DIRECTIONALLY_CONCORDANT", "A-C2": "DIRECTIONALLY_CONCORDANT"}


def test_supported_requires_all_conditions():
    v = rv.decide_verdict("METHOD_SUFFICIENT_FOR_INDEPENDENT_REPRODUCTION_ASSESSMENT",
                          "ROY_PREDICTION_DIRECTIONALLY_CONCORDANT", "DIRECTIONALLY_CONCORDANT",
                          "DIRECTIONALLY_CONCORDANT", "DIRECTIONALLY_CONCORDANT", _central_ok())
    assert v == rv.SUPPORTED


def test_central_directional_discordance_blocks_supported():
    # D-C1 directionally discordant -> cannot be SUPPORTED even if everything else agrees
    v = rv.decide_verdict("METHOD_PARTIALLY_IDENTIFIABLE_BUT_ASSESSMENT_POSSIBLE",
                          "ROY_PREDICTION_DIRECTIONALLY_CONCORDANT", "MIXED",
                          "DIRECTIONALLY_DISCORDANT", "DIRECTIONALLY_CONCORDANT",
                          {"D-C1": "DIRECTIONALLY_DISCORDANT"})
    assert v != rv.SUPPORTED
    assert v == rv.PARTIAL


def test_partial_when_one_domain_concordant_and_a_central_discordant():
    v = rv.decide_verdict("METHOD_PARTIALLY_IDENTIFIABLE_BUT_ASSESSMENT_POSSIBLE",
                          "ROY_PREDICTION_DIRECTIONALLY_CONCORDANT", "ROY_DIMENSIONALITY_MIXED",
                          "DIRECTIONALLY_DISCORDANT", "ROY_ALIGNMENT_DIRECTIONALLY_CONCORDANT",
                          {"D-C1": "DIRECTIONALLY_DISCORDANT", "P-C1": "DIRECTIONALLY_CONCORDANT"})
    assert v == rv.PARTIAL


def test_not_supported_on_broad_failure():
    # prediction fails materially -> NOT_SUPPORTED
    v = rv.decide_verdict("METHOD_PARTIALLY_IDENTIFIABLE_BUT_ASSESSMENT_POSSIBLE",
                          "ROY_PREDICTION_DISCORDANT", "DIRECTIONALLY_DISCORDANT",
                          "DIRECTIONALLY_DISCORDANT", "DIRECTIONALLY_DISCORDANT",
                          {"P-C1": "DISCORDANT"})
    assert v == rv.NOT_SUPPORTED


def test_not_supported_when_both_geometry_domains_discordant():
    v = rv.decide_verdict("METHOD_PARTIALLY_IDENTIFIABLE_BUT_ASSESSMENT_POSSIBLE",
                          "ROY_PREDICTION_DIRECTIONALLY_CONCORDANT", "DIRECTIONALLY_DISCORDANT",
                          "DIRECTIONALLY_DISCORDANT", "DIRECTIONALLY_DISCORDANT",
                          {"D-C1": "DIRECTIONALLY_DISCORDANT"})
    assert v == rv.NOT_SUPPORTED


def test_public_methods_insufficient_when_method_not_sufficient():
    v = rv.decide_verdict("METHOD_INSUFFICIENT_FOR_STRONG_ASSESSMENT",
                          "ROY_PREDICTION_DIRECTIONALLY_CONCORDANT", "MIXED",
                          "MIXED", "DIRECTIONALLY_CONCORDANT", {})
    assert v == rv.INSUFFICIENT


def test_committed_verdict_matches_logic_on_committed_statuses():
    v = _j("reproduction_verdict.json")
    recomputed = rv.decide_verdict(v["method_sufficiency"], v["prediction_status"],
                                   v["dimensionality_status"], v["central_claim_status"]["D-C1"],
                                   v["alignment_status"], v["central_claim_status"])
    assert recomputed == v["FINAL_VERDICT"] == rv.PARTIAL


# ------------------------------------------------------------------ artifact schema / guards
def test_three_headline_domains_exactly():
    rubric = _j("final_verdict_rubric.json")
    assert rubric["headline_domains"] == ["PREDICTION", "DIMENSIONALITY", "ALIGNMENT"]
    ds = _j("domain_summary.json")
    assert set(ds) == {"METHOD", "PREDICTION", "DIMENSIONALITY", "ALIGNMENT", "FEATURE_CORRESPONDENCE"}


def test_claim_matrix_has_required_rows():
    with open(ART / "final_claim_matrix.csv", newline="") as f:
        ids = {r["claim_id"] for r in csv.DictReader(f)}
    required = {"M-DATA", "M-BETA", "M-ROI", "M-VOXEL", "M-FOLDS", "M-RIDGE", "M-PAIR-V2V", "M-PAIR-V2I",
                "M-DENOISE", "M-RANK", "M-PREDNULL", "P-C1", "D-C1", "D-C2", "D-C3", "A-C1", "A-C2", "A-C3", "F-C1"}
    assert required <= ids, f"missing claim rows: {required - ids}"


def test_central_importance_frozen():
    imp = _j("final_verdict_rubric.json")["importance"]
    for c in ("P-C1", "D-C1", "A-C1", "A-C2"):
        assert imp[c] == "CENTRAL"


def test_feature_claim_not_evaluated_not_proxied():
    fa = _j("feature_correspondence_audit.json")
    assert fa["status"] == "NOT_EVALUATED" and fa["reconstructed_in_track_R"] is False


def test_dataset_scope_guard():
    ds = _j("dataset_scope.json")
    assert "DATASET_1" in ds["verdict_scope"]
    assert "dataset_2" in ds["NOT_reconstructed"]
    v = _j("reproduction_verdict.json")
    assert "Dataset 2" in v["dataset_2_limitation"] and "NOT" in v["dataset_2_limitation"].upper()


def test_no_numerical_weighted_score():
    blob = (ART / "reproduction_verdict.json").read_text().lower()
    for bad in ("%", "/10 reproduced", "70%", "out of 10", "weighted score"):
        assert bad not in blob, f"forbidden score token present: {bad}"
    assert _j("reproduction_verdict.json")["no_numerical_score"] is True


def test_no_significantly_aligned_without_inferential_support():
    for name in ("alignment_final_audit.json", "reproduction_verdict.json", "domain_summary.json"):
        assert "significantly aligned" not in (ART / name).read_text().lower()


def test_exact_replication_terminology_guard():
    v = _j("reproduction_verdict.json")
    assert "exact_replication_limitation" in v
    blob = (ART / "reproduction_verdict.json").read_text().lower()
    for bad in ("the paper was reproduced", "roy was wrong", "the paper is disproven", "reproduction failed"):
        assert bad not in blob


def test_immutable_prior_artifacts_present_and_flagged():
    v = _j("reproduction_verdict.json")
    assert v["immutable"]["S2_5R_VERDICT_WITHHELD"] is True
    assert v["immutable"]["S2_6R_RESULTS_IMMUTABLE"] is True
    assert v["immutable"]["H_A_CROSS_PARTICIPANT_PARTIAL"] == "unchanged"


def test_no_model_execution_entry_point():
    # the S2.7R artifact dir carries no compute driver; audit only.
    assert not (ART / "run_roy_s2_7r.py").exists()
    prov = _j("execution_provenance.json")
    assert prov["no_models_fit"] is True and prov["class"] == "AUDIT_VERDICT_ONLY_NO_MODELING"


def test_supplement_no_material_gap():
    assert _j("supplement_evidence.json")["material_method_gap_found"] is False
