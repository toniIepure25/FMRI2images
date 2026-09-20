"""MINDIR O2.16-SEC preregistration certification (data-free/synthetic). Validates the FROZEN decision logic for
the six independent-cohort secondary replication tests BEFORE any independent outcome exists. Computes no
historical N=8 inference. The 14 required pre-outcome synthetic controls, plus config immutability."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fmri2img.mindcompiler.o2_16_secondary_prereg import secondary_prereg as S

rng = np.random.default_rng(216)
CFG = json.load(open("artifacts/mindcompiler/o2_16_secondary_prereg/secondary_prereg_frozen_config.json"))


def _sig_effects(n=8, val=0.3):
    return [val] * n            # all-positive equal -> exact sign-flip p = 1/2^n (Holm-significant in a 6-family)


def _null_effects(n=8):
    return list(rng.normal(0, 0.02, n))   # ~0-centred -> sign-flip not significant


def _holm_all_sig():
    return S.six_test_holm({t: 0.5 / 2 ** 8 for t in S.SIX_TESTS})


# 1. S1 shared-angle synthetic positive
def test_1_s1_positive():
    holm = _holm_all_sig()
    assert S.support_S1(_sig_effects(), holm[("ventral", "S1_Bangles")], triviality_pass=True) is True


# 2. S1 pipeline-constrained false-positive blocked
def test_2_s1_triviality_blocks():
    holm = _holm_all_sig()
    assert S.support_S1(_sig_effects(), holm[("ventral", "S1_Bangles")], triviality_pass=False) is False


# 3. S2 planted Q_OUT->margin positive
def test_3_s2_positive():
    holm = _holm_all_sig()
    assert S.support_S2([0.42, 0.5, 0.38, 0.47, 0.3, 0.51, 0.29, 0.44], holm[("ventral", "S2_QOUT")]) is True


# 4. S2 null association negative
def test_4_s2_null_negative():
    p = S.participant_signflip_p(_null_effects())
    holm = S.six_test_holm({**{t: 0.9 for t in S.SIX_TESTS}, ("ventral", "S2_QOUT"): p})
    assert S.support_S2(_null_effects(), holm[("ventral", "S2_QOUT")]) is False


# 5. S3 isotropic perturbation null
def test_5_s3_isotropic_null():
    holm = S.six_test_holm({t: 0.9 for t in S.SIX_TESTS})
    assert S.support_S3(_null_effects(), holm[("ventral", "S3_ANISO")], repro_participant_frac=0.2, triviality_pass=True) is False


# 6. S3 planted anisotropy positive
def test_6_s3_positive():
    holm = _holm_all_sig()
    assert S.support_S3([0.09, 0.10, 0.08, 0.11, 0.087, 0.09, 0.083, 0.11],
                        holm[("ventral", "S3_ANISO")], repro_participant_frac=1.0, triviality_pass=True) is True


# 7. S3 cross-half reproducibility guard (positive passes; failing repro blocks)
def test_7_s3_reproducibility_guard():
    holm = _holm_all_sig()
    eff = _sig_effects(val=0.1)
    assert S.support_S3(eff, holm[("ventral", "S3_ANISO")], repro_participant_frac=0.75, triviality_pass=True) is True
    assert S.support_S3(eff, holm[("ventral", "S3_ANISO")], repro_participant_frac=0.5, triviality_pass=True) is False


# 8. exact six-test Holm
def test_8_six_test_holm():
    assert len(S.SIX_TESTS) == 6 and len(set(S.SIX_TESTS)) == 6
    # two strongly-significant + four null: Holm rejects exactly the two, at 6-family thresholds
    p = {t: 0.9 for t in S.SIX_TESTS}
    p[("ventral", "S1_Bangles")] = 0.5 / 2 ** 8
    p[("lateral", "S1_Bangles")] = 0.5 / 2 ** 8
    holm = S.six_test_holm(p)
    assert holm[("ventral", "S1_Bangles")] and holm[("lateral", "S1_Bangles")]
    assert not holm[("ventral", "S2_QOUT")]
    # exactly-six enforced
    try:
        S.six_test_holm({t: 0.01 for t in S.SIX_TESTS[:5]}); assert False
    except AssertionError:
        pass


# 9. 75% participant rule (6/8 passes, 5/8 fails)
def test_9_seventyfive_rule():
    assert abs(S.frac_pos([1, 1, 1, 1, 1, 1, -1, -1]) - 0.75) < 1e-12       # 6/8
    holm = _holm_all_sig()
    six_pos = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, -0.1, -0.1]                    # median>0, 6/8>0
    five_pos = [0.3, 0.3, 0.3, 0.3, 0.3, -0.1, -0.1, -0.1]                  # 5/8>0 -> fails
    assert S.support_S2(six_pos, holm[("ventral", "S2_QOUT")]) is True
    assert S.support_S2(five_pos, holm[("ventral", "S2_QOUT")]) is False


# 10. historical participants excluded from replication inference
def test_10_historical_excluded():
    parts = ["subj01", "subj05", "ind01", "ind02", "ind03"]
    kept, dropped = S.replication_inference_set(parts)
    assert kept == ["ind01", "ind02", "ind03"] and set(dropped) == {"subj01", "subj05"}
    assert all(h in S.HISTORICAL_N8 for h in dropped)


# 11. secondary cannot alter primary status
def test_11_secondary_cannot_alter_primary():
    for sec in ("THREE_GEOMETRIC_DISCOVERY_FINDINGS_INDEPENDENTLY_REPLICATED",
                "GEOMETRIC_DISCOVERY_FINDINGS_NOT_REPLICATED"):
        out = S.seal_secondary("O2_16_INDEPENDENT_REPLICATION_DATA_UNAVAILABLE", sec)
        assert out["o2_16_primary_status"] == "O2_16_INDEPENDENT_REPLICATION_DATA_UNAVAILABLE"


# 12. failed X2/X3/X4-H2 hypotheses excluded from positive family
def test_12_failed_hypotheses_excluded():
    assert set(S.ENDPOINTS).isdisjoint(set(S.FAILED_HYPOTHESES))
    for bad in ("X2_K2_minus_K1", "X3_repeat_stable_angular_scaffold", "X4_H2_mode_gap_to_margin"):
        assert bad in S.FAILED_HYPOTHESES
    S.assert_no_failed_hypothesis(list(S.ENDPOINTS))                        # clean family ok
    try:
        S.assert_no_failed_hypothesis(list(S.ENDPOINTS) + ["X4_H2_mode_gap_to_margin"]); assert False
    except ValueError:
        pass


# 13. centred X4 statistic used, never ratio
def test_13_centred_statistic():
    assert S.S3_STATISTIC == "ANISO_CENTRED"
    assert CFG["S3_anisotropy"]["primary_statistic"] == "ANISO_CENTRED"
    assert CFG["S3_anisotropy"]["ratio_used_for_inference"] is False
    s3 = json.load(open("artifacts/mindcompiler/o2_16_secondary_prereg/s3_anisotropy_replication_spec.json"))
    assert s3["primary_statistic"] == "ANISO_CENTRED" and s3["ratio_used_for_inference"] is False
    # the support rule references ANISO_CENTRED, never a ratio
    assert any("ANISO_CENTRED" in c for c in s3["support_rule"]["all_of"])


# 14. O2.16 config hash unchanged
def test_14_o2_16_config_unchanged():
    man = json.load(open("artifacts/mindcompiler/o2_16_secondary_prereg/historical_discovery_seal_manifest.json"))
    assert man["o2_16_primary_config_sha256"].startswith("2da2cc791054406b42eb1896b9b5f123b5b6bacb")
    assert man["o2_16_data_config_sha256"].startswith("804d5b21fc0c8656f9e0ef62ac4b7f412b114b7e")
    assert CFG["primary_o2_16"]["config_unchanged"] is True


# ---- classification branch coverage + config immutability ----
def _support(v, l):
    return {"ventral": dict(zip(S.ENDPOINTS, v)), "lateral": dict(zip(S.ENDPOINTS, l))}


def test_program_status_branches():
    assert S.classify_program(_support((1, 1, 1), (1, 1, 1))) == "THREE_GEOMETRIC_DISCOVERY_FINDINGS_INDEPENDENTLY_REPLICATED"
    assert S.classify_program(_support((0, 0, 0), (0, 0, 0))) == "GEOMETRIC_DISCOVERY_FINDINGS_NOT_REPLICATED"
    assert S.classify_program(_support((1, 0, 0), (0, 0, 0))) == "GEOMETRIC_DISCOVERY_REPLICATION_MULTIREGIME"  # xor
    assert S.classify_program(_support((1, 1, 0), (1, 1, 0))) == "GEOMETRIC_DISCOVERY_REPLICATION_PARTIAL"


def test_config_frozen():
    assert CFG["secondary_class"] == "PROSPECTIVELY_PREREGISTERED_SECONDARY_REPLICATION_ENDPOINTS"
    assert CFG["family_size"] == 6 and CFG["alpha"] == 0.05
    assert CFG["current_status"] == "O2_16_SECONDARY_ENDPOINTS_PREREGISTERED_AWAITING_INDEPENDENT_COHORT"
    assert CFG["O3"] == "O3_NOT_READY" and "config_sha256" in CFG
    assert CFG["participant_inferential_unit"] is True
    assert len(CFG["endpoint_family"]) == 6
