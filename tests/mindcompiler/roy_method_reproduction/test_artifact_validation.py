"""Regression fixtures for the cross-artifact validator.

A synthetic but internally-consistent artifact set must PASS; each injected
S1.8-class contradiction must make the SPECIFIC check fail (and the report fail
overall). Also validates a freshly generated real replay directory when B0 is
present (Lane E), proving the current runner emits self-consistent artifacts.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from fmri2img.mindcompiler.roy_method_reproduction import artifact_validation as av
from fmri2img.mindcompiler.roy_method_reproduction.metrics import FINITE_FRACTION_MIN

REPO = Path(__file__).resolve().parents[3]


def _good():
    tt = pd.DataFrame({"beta_index0": [10, 11, 12, 13]})
    csv_bytes = tt.to_csv(index=False).encode()
    sha = hashlib.sha256(csv_bytes).hexdigest()
    split_manifest = {"split_seed": 1234, "trials": [
        {"row": 0, "beta_index0": 10, "identity": "A", "state": "vision", "split": "train"},
        {"row": 1, "beta_index0": 11, "identity": "A", "state": "vision", "split": "train"},
        {"row": 2, "beta_index0": 12, "identity": "A", "state": "vision", "split": "val"},
        {"row": 3, "beta_index0": 13, "identity": "A", "state": "vision", "split": "test"},
    ]}
    pairing_manifest = {
        "vis2vis": {"policy": "all_ordered_distinct", "split_seed": 1234,
                    "pairing_seed": None, "randomness_source": "none_after_split",
                    "train_src": [0, 1], "train_tgt": [1, 0]},
        "vis2img": {"policy": "historical_index_aligned", "split_seed": 1234,
                    "pairing_seed": None, "randomness_source": "none_after_split",
                    "train_src": [0], "train_tgt": [1]},
    }
    result = {
        "b0_sha": av.B0_SHA, "trial_table_sha": sha, "split_seed": 1234,
        "pairing_seed": None, "pairing_randomness_source": "split_seed",
        "finite_fraction_min": FINITE_FRACTION_MIN,
        "vis2vis": {"rank": 2, "rank_hard_max": 4, "rank_max": None, "finite_frac": 1.0},
        "vis2img": {"rank": 1, "rank_hard_max": 4, "rank_max": None, "finite_frac": 1.0},
    }
    return result, split_manifest, pairing_manifest, tt, csv_bytes


def _run(fix):
    result, split_manifest, pairing_manifest, tt, csv_bytes = fix
    return av.validate_artifacts(result, split_manifest, pairing_manifest, tt, csv_bytes)


def test_consistent_artifacts_pass():
    rep = _run(_good())
    assert rep.ok, rep.failures()


def test_b1_beta_sha_certified_passes():
    fix = _good()
    fix[0].pop("b0_sha", None)
    fix[0]["beta_version"] = "fithrf_GLMdenoise_RR"
    fix[0]["beta_sha"] = av.BETA_PINS["fithrf_GLMdenoise_RR"]
    rep = _run(fix)
    assert rep.ok, rep.failures()
    assert any(c.name == "result.beta_sha_is_certified" and c.ok for c in rep.checks)


def test_b1_wrong_beta_sha_fails():
    fix = _good()
    fix[0].pop("b0_sha", None)
    fix[0]["beta_version"] = "fithrf_GLMdenoise_RR"
    fix[0]["beta_sha"] = "00" * 32
    rep = _run(fix)
    assert not rep.ok
    assert any(c.name == "result.beta_sha_is_certified" for c in rep.failures())


def test_b0_with_beta_sha_certified_passes():
    fix = _good()
    fix[0]["beta_version"] = "fithrf"
    fix[0]["beta_sha"] = av.BETA_PINS["fithrf"]
    rep = _run(fix)
    assert rep.ok, rep.failures()


def test_s18_seed_recorded_in_manifest_but_null_in_result_fails():
    fix = _good()
    # exact S1.8 defect: split-deterministic manifest carries a stray seed
    fix[2]["vis2vis"]["pairing_seed"] = 1234
    rep = _run(fix)
    assert not rep.ok
    assert any("seed_matches_policy" in c.name for c in rep.failures())


def test_wrong_b0_sha_fails():
    fix = _good(); fix[0]["b0_sha"] = "00" * 32
    rep = _run(fix)
    assert not rep.ok and any(c.name == "result.b0_sha_is_certified" for c in rep.failures())


def test_trial_table_sha_mismatch_fails():
    fix = _good(); fix[0]["trial_table_sha"] = "deadbeef"
    rep = _run(fix)
    assert any(c.name == "result.trial_table_sha_matches_file" for c in rep.failures())


def test_split_beta_index_mismatch_fails():
    fix = _good(); fix[1]["trials"][0]["beta_index0"] = 999
    rep = _run(fix)
    assert any(c.name == "split.beta_index0_matches_trial_table" for c in rep.failures())


def test_pairing_train_row_leak_fails():
    fix = _good(); fix[2]["vis2vis"]["train_src"] = [2, 3]  # val/test rows, not train
    rep = _run(fix)
    assert any("train_pairs_are_train_rows" in c.name for c in rep.failures())


def test_rank_exceeds_hard_max_fails():
    fix = _good(); fix[0]["vis2vis"]["rank"] = 99
    rep = _run(fix)
    assert any("rank_within_hard_max" in c.name for c in rep.failures())


def test_seeded_p1_i1_consistent_passes():
    fix = _good()
    for model, pol in (("vis2vis", "deterministic_derangement"),
                       ("vis2img", "independent_within_identity_permutation")):
        fix[2][model].update(policy=pol, pairing_seed=1234,
                             randomness_source="child_seed(pairing_seed)")
    fix[0]["pairing_seed"] = 1234
    fix[0]["pairing_randomness_source"] = "child_seed(pairing_seed)"
    rep = _run(fix)
    assert rep.ok, rep.failures()


def test_seeded_manifest_but_null_result_seed_fails():
    fix = _good()
    fix[2]["vis2vis"].update(policy="deterministic_derangement", pairing_seed=1234,
                             randomness_source="child_seed(pairing_seed)")
    # result still says null -> result/manifest disagree
    rep = _run(fix)
    assert not rep.ok
    assert any("seed_equals_result" in c.name or "matches_any_seeded" in c.name
               for c in rep.failures())


@pytest.mark.skipif(
    not (REPO / "data/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/"
         "nsdimagerybetas_fithrf/betas_nsdimagery.hdf5").exists(),
    reason="B0 not present in this environment")
def test_real_replay_dir_is_self_consistent(tmp_path):
    out = tmp_path / "replay"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts/analysis/mindcompiler/run_roy_s1_smoke.py"),
         "--split-seed", "1234", "--vis2vis-pairing", "all-ordered-distinct",
         "--vis2img-pairing", "historical-index-aligned", "--output-dir", str(out)],
        capture_output=True, text=True, cwd=str(REPO), env=env)
    assert r.returncode == 0, r.stderr
    rep = av.validate_smoke_dir(out)
    assert rep.ok, rep.failures()
