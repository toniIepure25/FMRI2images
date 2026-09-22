"""O2.16 Vienna site-prep certification (data-free/synthetic; official-source facts only). The 20 required
tests. No new science; candidate sites never validated; no access/validation claimed."""
from __future__ import annotations

import json
from pathlib import Path

from fmri2img.mindcompiler.o2_16_experiment import (
    config as C, protocol as P, randomization as R, sites as SITES, mode_guard as MG, stimuli as S,
    presentation as PRES, presentation_psychopy as PP, adapters as AD, acquisition_plan as AP,
    hardware_validation_harness as HV,
)

MEDUNI = "configs/mindcompiler/o2_16/sites/MEDUNI_HFMRC_CANDIDATE.yaml"
UNIVIE = "configs/mindcompiler/o2_16/sites/UNIVIE_MR_CENTER_CANDIDATE.yaml"
QUESTIONNAIRE = "docs/research/mindcompiler/site_integration/PI_MRI_SITE_QUESTIONNAIRE.md"


# 1. PsychoPy presenter interface
def test_01_psychopy_interface():
    for m in ("initialize_display", "load_stimulus", "draw_fixation", "draw_visual_stimulus", "draw_text_cue",
              "flip", "wait", "capture_response", "shutdown", "show"):
        assert hasattr(PP.PsychoPyPresenter, m)
    assert PP.PINNED_PSYCHOPY_VERSION


# 2. PsychoPy absent -> simulation still works
def test_02_psychopy_absent_simulation_ok():
    # importing the psychopy presenter module never requires psychopy
    p = PRES.SimulationPresenter()
    flip, off, dropped = p.show("x", 0.0, 0.1)
    assert off > flip and dropped == 0
    if not PP.psychopy_available():
        pr = PP.PsychoPyPresenter()
        try:
            pr.initialize_display(); assert False
        except PP.PsychoPyUnavailable:
            pass


# 3. candidate configs fail confirmatory (validation not MATCHES; status CANDIDATE_NOT_VALIDATED)
def test_03_candidates_fail_confirmatory():
    for f in (MEDUNI, UNIVIE):
        site = SITES.load_site_yaml(f)
        assert str(site.get("status")) == "CANDIDATE_NOT_VALIDATED"
        v = SITES.validate_site(site)
        assert v["status"] != "SITE_MATCHES_TARGETS"


# 4. verified public values parse correctly
def test_04_verified_values_parse():
    u = SITES.load_site_yaml(UNIVIE)
    assert "Skyra" in u["model"] and "3T" in str(u["field_strength"]) and "BOLDscreen" in u["display"]
    m = SITES.load_site_yaml(MEDUNI)
    assert "7T" in str(m["field_strength"]) or "7T" in m["model"]


# 5. TBD fields remain blockers
def test_05_tbd_blockers():
    for f in (MEDUNI, UNIVIE):
        v = SITES.validate_site(SITES.load_site_yaml(f))
        assert v["status"] == "SITE_TBD_INCOMPLETE" and len(v["tbd_fields"]) > 0


# 6. no public-source unknown silently populated (trigger_event TBD in both)
def test_06_no_silent_unknowns():
    for f in (MEDUNI, UNIVIE):
        site = SITES.load_site_yaml(f)
        assert C.is_tbd(site.get("trigger_event"))
        assert C.is_tbd(site.get("TR"))


# 7. trigger mapping configurable
def test_07_trigger_configurable():
    cfg = AD.TriggerAdapterConfig()
    ok, missing = cfg.confirmatory_ready()
    assert not ok and "trigger_value" in missing
    cfg2 = AD.TriggerAdapterConfig(trigger_type="KEYBOARD_KEY", trigger_value="t", debounce_ms=5,
                                   expected_TR_s=1.6, initial_dummy_count=4)
    ok2, _ = cfg2.confirmatory_ready(); assert ok2


# 8. response mapping configurable
def test_08_response_configurable():
    cfg = AD.ResponseAdapterConfig(device="Current Designs 4-Button", button_1="1", button_2="2", button_3="3", button_4="4")
    assert cfg.mapping() == {1: "1", 2: "2", 3: "3", 4: "4"}
    ok, _ = cfg.confirmatory_ready(); assert ok
    ok2, missing = AD.ResponseAdapterConfig().confirmatory_ready(); assert not ok2 and missing


# 9. run-duration calculator (TBD-aware)
def test_09_run_duration_calculator():
    est = AP.estimate_acquisition(C.ExperimentConfig())
    assert est["TR_s"] == "TBD_SITE_OPERATOR"
    assert "TBD" in str(est["total_scheduled_block"])
    assert est["perception"]["total_task_time_s"] > 0


# 10. run-partition invariants
def test_10_run_partition_invariants():
    for row in AP.run_partition_options():
        assert row["total_preserved"] and row["balanced_integer"]


# 11. all 512 anchors retained x3
def test_11_perception_counts():
    seq, _ = R.perception_sequence("sub-SYN001", "01")
    ok, det = P.prove_perception(seq)
    assert ok and det["each_anchor_exactly_3"]


# 12. all 12 imagery identities x8
def test_12_imagery_counts():
    seq, _ = R.imagery_sequence("sub-SYN001", "01")
    ok, det = P.prove_imagery(seq)
    assert ok and det["each_identity_exactly_8"]


# 13. session partitions preserve totals
def test_13_session_partitions():
    for o in AP.session_partition_options():
        assert o["perception_trials"] == 1536 and o["imagery_trials"] == 96 and o["preserves_counts"]


# 14. site configs do not mutate science
def test_14_site_configs_no_science_mutation():
    _ = SITES.load_site_yaml(MEDUNI); _ = SITES.load_site_yaml(UNIVIE)
    assert C.PERCEPTION_TRIALS == 1536 and C.IMAGERY_TRIALS == 96 and C.OUTER_FOLDS == 6
    assert C.IMMUTABLE_HASHES["o2_16_primary_config_sha256"].startswith("2da2cc79")


# 15. real-site validation cannot PASS without real certification artifact
def test_15_real_site_not_prevalidated():
    h = HV.define_harness()
    assert h["requires_real_hardware"] and all(t["status"] == "NOT_YET_RUN" for t in h["tests"])
    assert "SITE_DEFINED_ACCEPTANCE_THRESHOLD" in h["acceptance_threshold"]


# 16. stimulus permissions incomplete blocks confirmatory
def test_16_permissions_block_confirmatory():
    rows = [{"confirmatory_use_status": "AUTHORIZED"}, {"confirmatory_use_status": "LEGAL_OR_INSTITUTIONAL_REVIEW_REQUIRED"}]
    ok, blockers = MG.stimulus_permissions_ok(rows)
    assert not ok and len(blockers) == 1


# 17. imagery placeholder blocks confirmatory
def test_17_placeholder_blocks_confirmatory():
    ok, rep = S.verify_manifest(S.synthetic_imagery_manifest(), C.Mode.CONFIRMATORY)
    assert not ok and rep["n_placeholder"] == 12


# 18. site questionnaire maps every critical TBD
def test_18_questionnaire_maps_tbd():
    text = Path(QUESTIONNAIRE).read_text().lower()
    for term in ["tr", "trigger", "display", "button", "dummy", "bids", "voxel", "ethics", "field strength"]:
        assert term in text, term


# 19. O2.16 config hash unchanged
def test_19_o2_16_hash():
    import hashlib
    d = json.load(open("artifacts/mindcompiler/operator_o2_16/o2_16_frozen_config.json"))
    c2 = {k: v for k, v in d.items() if k != "config_sha256"}
    assert hashlib.sha256(json.dumps(c2, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == d["config_sha256"] == C.IMMUTABLE_HASHES["o2_16_primary_config_sha256"]


# 20. O2.16-SEC config hash unchanged
def test_20_o2_16_sec_hash():
    import hashlib
    d = json.load(open("artifacts/mindcompiler/o2_16_secondary_prereg/secondary_prereg_frozen_config.json"))
    c2 = {k: v for k, v in d.items() if k != "config_sha256"}
    assert hashlib.sha256(json.dumps(c2, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == d["config_sha256"] == C.IMMUTABLE_HASHES["o2_16_sec_config_sha256"]
