"""O2.16-ENG certification (data-free/synthetic). The 27 required unit tests + integration behaviours. No
scientific inference; no historical N=8 data; no scientific result files touched."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fmri2img.mindcompiler.o2_16_experiment import (
    config as C, randomization as R, protocol as P, stimuli as S, mode_guard as MG,
    participant_registry as PR, scanner_sync as SS, response_device as RD, presentation as PRES,
    bids_logger as BL, run_controller as RC, session_controller as SC, replay as RP,
    stage_a_export as SA, simulation as SIM, sites as SITES, event_schema as ES,
)


def _cfg(mode=C.Mode.SIMULATION):
    return C.ExperimentConfig(mode=mode)


# 1. 512x3 perception count
def test_01_perception_count():
    seq, _ = R.perception_sequence("sub-SYN001", "01")
    ok, det = P.prove_perception(seq)
    assert ok and len(seq) == 1536 and det["each_anchor_exactly_3"]


# 2. 12x8 imagery count
def test_02_imagery_count():
    seq, _ = R.imagery_sequence("sub-SYN001", "01")
    ok, det = P.prove_imagery(seq)
    assert ok and len(seq) == 96 and det["each_identity_exactly_8"]


# 3. 6+6 family balance
def test_03_family_balance():
    seq, _ = R.imagery_sequence("sub-SYN002", "01")
    _, det = P.prove_imagery(seq)
    assert det["simple_count_6"] and det["nat_count_6"]


# 4. deterministic randomization
def test_04_deterministic():
    a, _ = R.perception_sequence("sub-SYN003", "01")
    b, _ = R.perception_sequence("sub-SYN003", "01")
    assert a == b


# 5. seed changes participant-to-participant
def test_05_seed_participant_specific():
    _, r1 = R.perception_sequence("sub-SYN003", "01")
    _, r2 = R.perception_sequence("sub-SYN004", "01")
    assert r1.seed != r2.seed
    a, _ = R.imagery_sequence("sub-SYN003", "01")
    b, _ = R.imagery_sequence("sub-SYN004", "01")
    assert a != b


# 6. exact replay
def test_06_exact_replay():
    seq, _ = R.imagery_sequence("sub-SYN005", "01")
    rec = RP.reconstruct("sub-SYN005", "01", "imagery")
    ok, why = RP.sequence_matches(rec, seq)
    assert ok, why


# 7. duplicate-stimulus guard (corrupted hash -> QC FAIL)
def test_07_duplicate_stimulus_hash_guard():
    out = SIM.simulate_participant("sub-SYN006", "01", n_runs=6, corrupt_hash=True)
    assert any(v == "FAIL" for v in out["qc"].values())


# 8. placeholder guard (confirmatory refuses placeholders)
def test_08_placeholder_guard():
    man = S.synthetic_imagery_manifest()
    ok, rep = S.verify_manifest(man, C.Mode.CONFIRMATORY)
    assert not ok and rep["n_placeholder"] == 12
    ok2, _ = S.verify_manifest(man, C.Mode.SIMULATION)
    assert ok2  # placeholders allowed in simulation


# 9. historical-ID independence guard
def test_09_historical_independence():
    assert not MG.participant_independence_ok("subj01")
    assert MG.participant_independence_ok("sub-IND001")


# 10. ethics guard
def test_10_ethics_guard():
    auth = MG.AuthorizationManifest()
    ok, missing = MG.ethics_guard(auth)
    assert not ok and "ethics_protocol_id" in missing


# 11. confirmatory-mode fail closed
def test_11_confirmatory_fail_closed():
    auth = MG.AuthorizationManifest()  # all false/None
    _, rep = S.verify_manifest(S.synthetic_imagery_manifest(), C.Mode.SIMULATION)
    with pytest.raises(MG.ConfirmatoryAbort):
        MG.confirmatory_precheck(C.Mode.CONFIRMATORY, auth, _cfg(C.Mode.CONFIRMATORY), rep, "sub-SITE001")


def test_11b_no_force_confirmatory_flag():
    from fmri2img.mindcompiler.o2_16_experiment import cli
    src = Path(cli.__file__).read_text()
    # no override flag is registered on the parser
    assert 'add_argument("--force' not in src and "force_confirmatory" not in src and "'--force" not in src


# 12. simulation mode works without scanner
def test_12_simulation_no_scanner():
    out = SIM.simulate_participant("sub-SYN007", "01")
    assert out["perception_trials"] == 1536 and out["imagery_trials"] == 96


# 13. pilot watermark
def test_13_pilot_watermark():
    assert MG.watermark(C.Mode.ENGINEERING_PILOT) == "NON_CONFIRMATORY_ENGINEERING_PILOT"


# 14. trigger counter
def test_14_trigger_counter():
    trg = SS.SimulatedTrigger(1.6, n_volumes=10); trg.emit_all()
    assert trg.count_volume() == 10


# 15. missing-trigger detection
def test_15_missing_trigger():
    trg = SS.SimulatedTrigger(1.6, n_volumes=10, drop_indices=[3, 7]); trg.emit_all()
    assert trg.detect_missing_trigger(10) == 2


# 16. duplicate-trigger detection
def test_16_duplicate_trigger():
    trg = SS.SimulatedTrigger(1.6, n_volumes=10, duplicate_indices=[4]); trg.emit_all()
    assert len(trg.detect_duplicate_trigger()) >= 1


# 17. BIDS events schema
def test_17_bids_schema(tmp_path):
    seq, _ = R.imagery_sequence("sub-SYN008", "01")
    run1 = [t for t in seq if t["run"] == 1]
    presenter = PRES.SimulationPresenter(); resp = RD.SimulatedResponseDevice("sub-SYN008", "01")
    recs = __import__("fmri2img.mindcompiler.o2_16_experiment.imagery_task", fromlist=["run_imagery_run"]).run_imagery_run(
        "sub-SYN008", "01", 1, run1, presenter, resp, None, _cfg())
    tsv = BL.write_events(recs, tmp_path, "SYN008", "01", "imagery", 1)
    header = Path(tsv).read_text().splitlines()[0].split("\t")
    for c in ES.EVENT_COLUMNS:
        assert c in header


# 18. BIDS round-trip
def test_18_bids_roundtrip(tmp_path):
    pid = "sub-SYN009"
    seq, _ = R.imagery_sequence(pid, "01")
    presenter = PRES.SimulationPresenter(); resp = RD.SimulatedResponseDevice(pid, "01")
    IT = __import__("fmri2img.mindcompiler.o2_16_experiment.imagery_task", fromlist=["run_imagery_run"])
    for run in sorted(set(t["run"] for t in seq)):
        recs = IT.run_imagery_run(pid, "01", run, [t for t in seq if t["run"] == run], presenter, resp, None, _cfg())
        BL.write_events(recs, tmp_path, "SYN009", "01", "imagery", run)
    ok, why = RP.roundtrip_from_bids(pid, "01", "imagery", tmp_path)
    assert ok, why


# 19. response logging
def test_19_response_logging():
    dev = RD.SimulatedResponseDevice("sub-SYN010", "01", miss_rate=1.0)
    r = dev.poll("t1", None, 1.3)
    assert r.missing is True and r.key is None


# 20. stimulus SHA validation
def test_20_sha_validation(tmp_path):
    f = tmp_path / "x.bin"; f.write_bytes(b"hello")
    h = S.sha256_file(f)
    e = S.StimulusEntry("id", "fam", "src", str(f), h, 1, 1, "image/png", "AUTHORIZED", True, False)
    ok, rep = S.verify_manifest([e], C.Mode.CONFIRMATORY)
    assert ok
    e2 = S.StimulusEntry("id", "fam", "src", str(f), "deadbeef", 1, 1, "image/png", "AUTHORIZED", True, False)
    ok2, rep2 = S.verify_manifest([e2], C.Mode.CONFIRMATORY)
    assert not ok2 and any("sha_mismatch" in i for i in rep2["issues"])


# 21. crash/recovery logging
def test_21_crash_recovery():
    pid = "sub-SYN011"
    seq, _ = R.imagery_sequence(pid, "01"); run1 = [t for t in seq if t["run"] == 1]
    man = RC.freeze_run_manifest(pid, "01", "imagery", 1, run1, _cfg(), "ENG", "synthetic", expected_triggers=len(run1))
    presenter = PRES.SimulationPresenter(); resp = RD.SimulatedResponseDevice(pid, "01")
    _, _, comp = RC.execute_run(man, presenter, resp, None, _cfg(), crash_after_trial=3)
    assert comp["status"] == "RUN_CRASHED" and comp["frozen_trial_order_preserved"] is True


# 22. run-manifest hash
def test_22_run_manifest_hash():
    seq, _ = R.perception_sequence("sub-SYN012", "01"); run1 = [t for t in seq if t["run"] == 1]
    m1 = RC.freeze_run_manifest("sub-SYN012", "01", "perception", 1, run1, _cfg(), "ENG", "s", len(run1))
    m2 = RC.freeze_run_manifest("sub-SYN012", "01", "perception", 1, run1, _cfg(), "ENG", "s", len(run1))
    assert m1["run_manifest_sha"] == m2["run_manifest_sha"] and len(m1["run_manifest_sha"]) == 64


# 23. Stage-A protected-path firewall
def test_23_stage_a_firewall():
    with pytest.raises(SA.StageBAccessError):
        SA.assert_not_stage_b("analysis/stage_b_protected/outcomes.json")
    assert SA.assert_not_stage_b("analysis/stage_a/predictors.json")


# 24. Stage-B release-token requirement
def test_24_stage_b_release_required(tmp_path):
    with pytest.raises(SA.StageBAccessError):
        SA.require_stage_b_release(tmp_path)
    (tmp_path / SA.RELEASE_FILE).write_text(json.dumps(SA.release_token_template()))
    with pytest.raises(SA.StageBAccessError):  # template still has TBDs -> incomplete
        SA.require_stage_b_release(tmp_path)


# 25. O2.16 config hash unchanged
def test_25_o2_16_hash_unchanged():
    import hashlib
    d = json.load(open("artifacts/mindcompiler/operator_o2_16/o2_16_frozen_config.json"))
    c2 = {k: v for k, v in d.items() if k != "config_sha256"}
    sha = hashlib.sha256(json.dumps(c2, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert sha == d["config_sha256"] == C.IMMUTABLE_HASHES["o2_16_primary_config_sha256"]


# 26. O2.16-SEC config hash unchanged
def test_26_o2_16_sec_hash_unchanged():
    import hashlib
    d = json.load(open("artifacts/mindcompiler/o2_16_secondary_prereg/secondary_prereg_frozen_config.json"))
    c2 = {k: v for k, v in d.items() if k != "config_sha256"}
    sha = hashlib.sha256(json.dumps(c2, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert sha == d["config_sha256"] == C.IMMUTABLE_HASHES["o2_16_sec_config_sha256"]


# 27. no scientific result files modified (engineering touches only its own trees)
def test_27_no_scientific_result_files():
    src = Path("src/fmri2img/mindcompiler/o2_16_experiment")
    for pyf in src.glob("*.py"):
        t = pyf.read_text()
        # engineering code must not write into sealed gate result dirs
        assert "operator_o2_9/results" not in t and "x4_geometric_drift/results" not in t


# ---- integration behaviours ----
def test_int_full_participant():
    out = SIM.simulate_participant("sub-SYN001", "01")
    assert out["replay_perception_exact"] and out["replay_imagery_exact"] and out["balance_all_pass"]


def test_int_12_participant_cohort():
    res = SIM.simulate_cohort(12)
    assert res["all_pass"] and res["no_historical_ids"] and res["no_stage_b_leakage"]


def test_int_trigger_failure_detected():
    trg = SS.SimulatedTrigger(1.6, n_volumes=20, drop_indices=[5, 6], duplicate_indices=[10]); trg.emit_all()
    assert trg.detect_missing_trigger(20) >= 1 and len(trg.detect_duplicate_trigger()) >= 1


def test_int_session_aborts_on_historical_confirmatory():
    auth = MG.AuthorizationManifest()
    _, rep = S.verify_manifest(S.synthetic_imagery_manifest(), C.Mode.SIMULATION)
    res = SC.precheck_session(C.Mode.CONFIRMATORY, "subj01", auth, _cfg(C.Mode.CONFIRMATORY), rep)
    assert res.aborted and "invalid_for_mode" in res.abort_reason or res.aborted


def test_int_site_validation_reviews():
    v = SITES.validate_site({"field_strength": "3T", "voxel_size": "2.0", "TR": "2.0"})
    assert v["status"] in ("SITE_REVIEW_REQUIRED", "SITE_TBD_INCOMPLETE")
