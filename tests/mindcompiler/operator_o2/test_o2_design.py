"""O2 design/leakage guards (data-free): LOSO x identity folds, global holdout, contracts."""
from __future__ import annotations

import json
from pathlib import Path

from fmri2img.mindcompiler.operator_o1 import folds as ofo

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "artifacts/mindcompiler/operator_o2"
IDS = [f"A:{c}" for c in "EHLPRV"] + [f"B:{c}" for c in "BCDKTW"]
FAM = {i: ("simple" if i.startswith("A:") else "naturalistic") for i in IDS}
SUBJ = [f"subj0{k}" for k in range(1, 9)]


def _j(n):
    return json.loads((A / n).read_text())


def test_eight_loso_folds_each_subject_target_once():
    targets = [s for s in SUBJ]                              # LOSO = each subject held out once
    assert len(targets) == 8 and len(set(targets)) == 8


def test_six_identity_folds_global_holdout():
    outer = ofo.outer_folds(IDS, FAM)
    assert len(outer) == 6
    tested = [i for f in outer for i in f.test]
    assert sorted(tested) == sorted(IDS)                    # every identity globally held out once
    for f in outer:
        assert {FAM[i] for i in f.test} == {"simple", "naturalistic"} and len(f.train) == 10


def test_frozen_config_scope_and_measurement():
    c = _j("o2_frozen_config.json")
    assert c["primary_beta"] == "B0 only" and c["primary_rois"] == ["ventral", "lateral", "parietal"]
    assert c["excluded_rois"] == ["V2", "hV4"] and c["secondary_rois"] == ["V1", "V3"]
    assert c["common_space_from"] == "VISION_ONLY" and c["same_W_applied_to_vision_and_imagery"] is True
    assert c["target_subject_vision_only_calibration"] is True and c["global_identity_holdout"] is True
    assert c["B0_measurement_dependence_is_a_limitation"] is True


def test_operator_contract_forbids_per_target_and_diagonal():
    oc = _j("operator_contract.json")
    assert oc["estimator"] == "FULL_SHARED_SCALAR_RIDGE"
    for bad in ("per-target lambda", "diagonal penalty", "coordinate-specific sparsity"):
        assert bad in oc["forbidden"]
    assert oc["no_diagonal_baseline"] is True


def test_no_b1_and_no_rawW_and_v2hv4_excluded():
    c = _j("o2_frozen_config.json")
    assert "B1 primary" in c["prohibited"] and "raw-W cross-subject comparison" in c["prohibited"]
    assert "V2/hV4 primary" in c["prohibited"]


def test_status_contract_supported_requires_all():
    sc = _j("scientific_status_contract.json")["SHARED_OPERATOR_SUPPORTED_requires_ALL"]
    assert set(sc.keys()) == {"A", "B", "C", "D", "E", "F"}


def test_manifest_covers_all_targets_and_identities():
    import csv
    with open(A / "outer_subject_identity_manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len({r["target_subject"] for r in rows}) == 8
    assert len({r["identity_fold"] for r in rows}) == 6
    # each (target, fold) has exactly 2 test identities
    from collections import Counter
    c = Counter((r["target_subject"], r["identity_fold"]) for r in rows if r["role_identity"] == "test")
    assert all(v == 2 for v in c.values())
