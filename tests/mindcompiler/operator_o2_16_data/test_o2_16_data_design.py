"""O2.16-DATA pre-acquisition design-validation suite (data-free). Certifies the frozen acquisition design is
internally consistent and O2.16-compatible BEFORE any human acquisition: 512 perception anchors x3; 12 imagery
identities (6+6) x8 repeats; 6 outer folds each holding 1 simple + 1 naturalistic; 100 balanced M4 subsets x 28
repeat pairs = 2800 M4T2 schedules/fold; 8 observations per schedule; Q_MON compatible; independence + no
held-out leakage in the design. No scanning is performed by these tests."""
from __future__ import annotations

import json
from pathlib import Path

from fmri2img.mindcompiler.operator_o2_16_data import acquisition_design as D


def test_design_valid():
    ok, checks = D.validate_design()
    assert ok, [k for k, v in checks.items() if not v]


def test_exact_counts():
    assert D.N_PERCEPTION_ANCHORS == 512 and D.PERCEPTION_REPEATS == 3
    assert len(D.IMAGERY_IDENTITIES) == 12 and len(D.SIMPLE) == 6 and len(D.NAT) == 6
    assert D.IMAGERY_REPEATS == 8 and len(D.IMAGERY_IDENTITIES) * D.IMAGERY_REPEATS == 96


def test_folds_and_schedules():
    folds = D.outer_folds()
    assert len(folds) == 6
    for fd in folds:
        assert len(fd["train_simple"]) == 5 and len(fd["train_nat"]) == 5
        assert len(D.m4_subsets(fd["train_simple"], fd["train_nat"])) == 100
        assert D.schedules_per_fold(fd) == 2800
    assert len(D.repeat_pairs()) == 28
    assert D.M * D.T == 8


def test_no_imagery_in_perception_and_loso():
    ok, c = D.validate_design()
    assert c["no_imagery_in_perception"] and c["heldout_disjoint_from_training"] and c["full_loso_coverage"]


def test_config_and_seals():
    cfg = json.load(open("artifacts/mindcompiler/operator_o2_16_data/o2_16_data_frozen_config.json"))
    assert cfg["immutable_history"]["O2_16"] == "O2_16_INDEPENDENT_REPLICATION_DATA_UNAVAILABLE"
    assert cfg["resume_o2_16_unchanged_after_data"] == "2da2cc79" and cfg["O3"] == "O3_NOT_READY"
    assert cfg["recruitment"]["N_PLANNED"] == 12 and cfg["recruitment"]["N_MIN_confirmatory"] == 8
    assert cfg["imagery"]["identities"] == 12 and cfg["imagery"]["repeats"] == 8
    assert "config_sha256" in cfg
    # no scientific-claim / no-threshold guards present
    assert "M4T2 replicates" in cfg["no_scientific_claim"]["may_not_claim"]
    assert cfg["ethics"]["if_absent_stop_at"] == "O2_16_DATA_AWAITING_HUMAN_ACQUISITION_AUTHORIZATION"
