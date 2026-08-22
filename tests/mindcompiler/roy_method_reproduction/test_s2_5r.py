"""S2.5R data-free tests over the committed audit artifacts (no rescue, no proxy)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
S = REPO / "artifacts/mindcompiler/roy_s2_5r"
present = (S / "s2_5r_gate_status.json").exists()
pytestmark = pytest.mark.skipif(not present, reason="S2.5R artifacts not present")


def _j(n):
    return json.loads((S / n).read_text())


def test_beta_b2_maps_to_fithrf_B0():
    b = _j("beta_version_evidence.json")
    assert b["nsd_manual_mapping"]["b2"] == "betas_fithrf"
    assert b["conclusion"] == "PUBLIC_EVIDENCE_STRONGLY_IDENTIFIES_B0_AS_B2_COMPATIBLE"
    assert "bitwise" in b["caveat"].lower()  # 'similar to b2' != bitwise-identical file


def test_dataset1_98th_distinct_from_dataset2_80th():
    m = _j("public_method_claims.json")
    assert "98th percentile" in m["M_voxel"]["quote"]
    assert "80th" in m["M_voxel"]["interpretation"]  # explicitly distinguished


def test_vis2img_pairing_is_material_high_mismatch():
    mm = pd.read_csv(S / "method_concordance_matrix.csv")
    row = mm[mm.method_item == "vis2img_pairing"].iloc[0]
    assert row["status"] == "MATERIAL_MISMATCH" and row["materiality"] == "HIGH"
    assert "random" in row["Roy_public_method"].lower() and "index-aligned" in row["our_method"].lower()


def test_no_b1_failure_label_because_b1_is_not_b2():
    b1 = _j("B1_branch_audit.json")
    assert b1["interpretation"] == "METHOD_SENSITIVITY_BRANCH_NOT_PRIMARY_PAPER_RECONSTRUCTION"
    assert "not a reproduction failure" in b1["prediction_concordance"].lower()


def test_prediction_null_not_substituted():
    cm = pd.read_csv(S / "claim_concordance_matrix.csv")
    p = cm[cm.claim_id == "P1_prediction"].iloc[0]
    assert p["B0_status"] == "DIRECTIONALLY_CONCORDANT"       # NOT "CONCORDANT"
    assert "NULL_NOT_DIRECTLY_TESTED" in p["notes"]


def test_no_alignment_proxy_substituted():
    a = _j("alignment_audit.json")
    assert a["our_status"] == "ROY_ALIGNMENT_RATIO_NOT_YET_RECONSTRUCTED"
    assert "not substituted" in a["no_proxy"].lower()
    cm = pd.read_csv(S / "claim_concordance_matrix.csv")
    assert cm[cm.claim_id == "G1_alignment"].iloc[0]["B0_status"] == "NOT_EVALUATED"


def test_exact_replication_terminology_preserved():
    v = _j("reproduction_verdict.json")
    assert "ORIGINAL_CODE_REPRODUCTION_UNAVAILABLE" in v["exact_replication"]
    assert "BITWISE_REPLICATION_UNAVAILABLE" in v["exact_replication"]
    # no "exact reproduction" claim anywhere in the verdict
    assert "exact roy reproduction" not in json.dumps(v).lower()


def test_material_gap_status_and_no_forced_verdict():
    g = _j("s2_5r_gate_status.json")
    assert g["status"] == "S2_5R_MATERIAL_METHOD_GAP_IDENTIFIED"
    assert "NOT FORCED" in g["reproduction_verdict"]
    assert g["frozen_H_A_unchanged"] == "H_A_CROSS_PARTICIPANT_PARTIAL"
    assert "S2.5M" in g["next_action"]["next"]


def test_claim_matrix_covers_all_four_classes():
    cm = pd.read_csv(S / "claim_concordance_matrix.csv")
    assert set(cm["claim_class"]) == {"M", "P", "D", "G"}   # not collapsed to one score
    assert len(cm) >= 8


def test_rubric_frozen_from_public_claims_only():
    r = _j("concordance_rubric_frozen.json")
    assert r["RUBRIC_DERIVED_FROM_PUBLIC_CLAIMS_ONLY"] is True
    assert "tuned to our numbers" in r["chronology_honesty"].lower()
