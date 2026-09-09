"""O2.4-PROVREC forensic-replay determination tests (data-free, offline).

Assert the forensic inventory + frozen reconstruction contract are self-consistent, that the historical
references are the immutable ones, that the generator-code-absent finding + the genuinely-missing
implementation details are recorded, that no replay/M0/calibration was fabricated (tolerance NOT loosened,
closest scorer NOT picked, no target imagery opened), and that O2.4 stays INCONCLUSIVE with the stronger
reason. O2/O3 immutability preserved.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "artifacts/mindcompiler/operator_o2_4_provrec"
O24 = ROOT / "artifacts/mindcompiler/operator_o2_4"
O3A = ROOT / "artifacts/mindcompiler/operator_o2_3a"


def _j(base, n):
    return json.loads((base / n).read_text())


def test_provrec_config_sha_matches():
    cfg = _j(P, "provrec_frozen_config.json")
    stored = cfg.pop("config_sha256")
    rec = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert rec == stored and stored.startswith("9e3060af")


def test_historical_references_immutable():
    h = _j(P, "historical_reference_hashes.json")
    assert h["o2_3a_frozen_config_sha256"] == "39d7bc9725f71a711641d445d4aeeb3bda128b0d06c45566a1c229ca9a537050"
    assert h["o2_3a_execution"] == "5f13e4a33da4691cbc8ef9b26dcf5f9b227aec6e"
    assert h["o2_3a_final_status"] == "CORE_ANCHOR_TARGET_ORIENTATION_NOT_IDENTIFIABLE"
    assert h["o2_4_frozen_config_sha256"] == "321b42f91a910384aa33e8d9c37c4447940ebc83523e1096b4008c0290f970f8"
    # the historical O2.3A status file is untouched
    assert _j(O3A, "scientific_status.json")["orientation_status"] == "CORE_ANCHOR_TARGET_ORIENTATION_NOT_IDENTIFIABLE"


def test_generator_code_absent_finding():
    inv = _j(P, "historical_o2_3a_inventory.json")
    assert inv["generator_code_in_history"] is False
    assert inv["anchor_included_images"] == 512 and inv["anchor_data_present"] is False


def test_missing_details_named_and_512_kcandidates():
    cfg = _j(P, "provrec_frozen_config.json")
    md = cfg["genuinely_absent_historical_details_precluding_exact_replay"]
    joined = " ".join(md).lower()
    assert "generator source" in joined and "detsrm" in joined and "block partition" in joined and "random-subspace" in joined
    assert cfg["common_space"]["K_candidates"] == [2, 4, 8, 16, 32, 64]
    assert cfg["residual_template"]["rank_candidates"] == [1, 2, 3, 4, 5, 6]
    assert cfg["null"]["seed_rule"].startswith("SHA256('O2.3A'")


def test_replay_status_historical_detail_missing_no_shortcuts():
    s = _j(P, "replay_status.json")
    assert s["status"] == "O2_4_PROVREC_HISTORICAL_DETAIL_MISSING"
    assert s["tolerance_loosened"] is False and s["closest_scorer_selected"] is False
    assert _j(P, "replay_results_summary.json")["status"] == "NOT_RUN"


def test_no_m0_no_target_imagery_opened():
    m0 = _j(P, "m0_replay_certification.json")
    assert m0["status"] == "NOT_CERTIFIED"
    assert m0["M_gt0_calibration_opened"] is False
    assert m0["target_imagery_opened_for_calibration"] is False
    ep = _j(P, "execution_provenance.json")
    assert ep["replay_executed"] is False and ep["target_calibration_imagery_opened"] is False
    assert ep["nothing_fabricated"] is True


def test_o2_4_inconclusive_stronger_reason_o3_locked():
    ss = _j(O24, "scientific_status.json")
    assert ss["status"] == "TARGET_STATE_ORIENTATION_CALIBRATION_INCONCLUSIVE"
    assert ss["provrec"]["status"] == "O2_4_PROVREC_HISTORICAL_DETAIL_MISSING"
    assert ss["provrec"]["stronger_reason"] == "HISTORICAL_O2_3A_STATE_NOT_RECONSTRUCTABLY_REPRODUCIBLE"
    assert ss["immutable"]["O2"] == "SHARED_OPERATOR_PARTIAL"
    assert ss["immutable"]["O3"] == "O3_NOT_READY"
