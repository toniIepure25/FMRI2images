"""S2.0 frozen-config immutability and B0/B1 registry integrity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
S2 = REPO / "artifacts/mindcompiler/roy_s2"


@pytest.mark.skipif(not (S2 / "s2_frozen_config.json").exists(), reason="frozen config not present")
def test_frozen_config_hash_is_self_consistent():
    d = json.loads((S2 / "s2_frozen_config.json").read_text())
    stored = d.pop("config_sha256")
    body = json.dumps(d, indent=2, sort_keys=True)
    assert hashlib.sha256(body.encode()).hexdigest() == stored


@pytest.mark.skipif(not (S2 / "s2_frozen_config.json").exists(), reason="frozen config not present")
def test_frozen_config_has_required_keys():
    d = json.loads((S2 / "s2_frozen_config.json").read_text())
    for k in ("beta_versions", "folds", "preprocessing_variants", "vis2vis_pairing_policies",
              "denoising", "ridge_grid", "rank_rule", "metric_definitions", "finite_fraction_min",
              "fold_aggregation", "test_evaluation_policy", "primary_experiment_matrix", "seeds"):
        assert k in d, f"missing frozen-config key {k}"
    assert d["finite_fraction_min"] == 0.90
    assert d["ridge_grid"] == {"lo": 1e-3, "hi": 1e5, "n": 100, "spacing": "log10"}


@pytest.mark.skipif(not (S2 / "s2_frozen_config.json").exists()
                    or not (REPO / "artifacts/mindcompiler/roy_s1/subj01_B0_download.json").exists(),
                    reason="artifacts not present")
def test_beta_registry_shas_match_download_provenance():
    cfg = json.loads((S2 / "s2_frozen_config.json").read_text())
    b0 = json.loads((REPO / "artifacts/mindcompiler/roy_s1/subj01_B0_download.json").read_text())
    b1 = json.loads((REPO / "artifacts/mindcompiler/roy_s1/subj01_B1_download.json").read_text())
    assert cfg["beta_versions"]["B0"]["sha256"] == b0["sha256"]
    assert cfg["beta_versions"]["B1"]["sha256"] == b1["sha256"]
    assert cfg["beta_versions"]["B0"]["sha256"] != cfg["beta_versions"]["B1"]["sha256"]
