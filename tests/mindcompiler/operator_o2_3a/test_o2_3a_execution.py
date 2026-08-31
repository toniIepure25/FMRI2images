"""O2.3A execution tests: mapping cert, anchor leakage, status logic, immutability, no-B1."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "artifacts/mindcompiler/operator_o2_3a"
O2 = ROOT / "artifacts/mindcompiler/operator_o2"


def _j(n):
    return json.loads((A / n).read_text())


def test_anchor_manifest_512_no_imagery_overlap_included():
    with open(A / "anchor_manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    inc = [r for r in rows if r["included"] in ("True", "true")]
    assert len(inc) == 512
    assert not any(r["overlaps_nsd_imagery"] in ("True", "true") for r in inc)   # zero imagery overlap in anchor


def test_mapping_certified_all_subjects():
    c = _j("core_imagery_mapping_certification.json")
    assert c["mapping_certified_all"] is True and c["no_resampling"] is True
    assert len(c["per_subject"]) == 8 and all(v["affine_matches_ncsnr"] for v in c["per_subject"].values())


def test_leakage_zero_target_imagery():
    with open(A / "leakage_certification.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 40                                                       # 8 subj x 5 ROI
    for r in rows:
        assert int(r["target_imagery_used_for_core_anchor"]) == 0
        assert int(r["target_imagery_used_for_K"]) == 0
        assert int(r["target_imagery_used_for_residual_template"]) == 0
        assert int(r["NSD_core_anchor_overlap_with_NSDimagery_identity"]) == 0


def test_execution_and_orientation_status():
    s = _j("scientific_status.json")
    assert s["execution_status"] == "O2_3A_CORE_ANCHOR_FEASIBILITY_PASS"
    assert s["orientation_status"] in {"CORE_ANCHOR_TARGET_ORIENTATION_IDENTIFIABLE", "CORE_ANCHOR_TARGET_ORIENTATION_PARTIAL",
                                       "CORE_ANCHOR_TARGET_ORIENTATION_NOT_IDENTIFIABLE", "CORE_ANCHOR_TARGET_ORIENTATION_INCONCLUSIVE"}
    assert s["leakage_clean"] is True and s["n_anchor_images"] == 512


def test_dense_perception_captures_more_than_small_span():
    # positive finding: R_Y_CORE exceeds O2.2 small-visual-span R_Y_VISFULL in primary ROIs
    d = _j("core_imagery_retention_summary.json")["per_ROI"]
    for r in ("ventral", "lateral"):
        assert d[r]["R_Y_CORE"] > d[r]["R_Y_VISFULL_small"]


def test_common_space_validated_primary():
    st = _j("anchor_common_space_status.json")
    for r in ("ventral", "lateral"):
        assert st[r]["status"] == "CORE_ANCHOR_COMMON_SPACE_VALIDATED"


def test_o2_o3_immutable():
    s = _j("scientific_status.json")
    assert s["immutable"]["O2"] == "SHARED_OPERATOR_PARTIAL" and s["immutable"]["O3"] == "O3_NOT_READY"
    assert json.loads((O2 / "scientific_status.json").read_text())["scientific_status"] == "SHARED_OPERATOR_PARTIAL"


def test_no_b1_and_frozen_methodology():
    p = _j("execution_provenance.json")
    assert p["no_b1"] is True and p["no_target_imagery_in_fitting"] is True
    assert p["frozen_methodology_sha"] == "39d7bc97"


def test_historical_blocker_preserved():
    # the first-attempt blocker status must remain documented, not rewritten
    s = _j("scientific_status.json")
    assert "O2_3A_CORE_DATA_UNAVAILABLE" in s["supersedes"]
    assert _j("core_data_acquisition_inventory.json")["NSD_PUBLIC_UNSIGNED_ACCESS_CONFIRMED"] is True
