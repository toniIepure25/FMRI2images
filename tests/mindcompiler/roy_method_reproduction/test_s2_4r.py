"""S2.4R data-free tests: frozen contract, formulas, category logic, no rescue."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DRIVER = REPO / "scripts/analysis/mindcompiler/run_roy_s2_4r.py"


def _drv():
    import sys
    sys.path.insert(0, str(REPO / "src"))
    spec = importlib.util.spec_from_file_location("s24r", DRIVER)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_exact_cohort_rois_betas():
    m = _drv()
    assert m.ALL == ["subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"]
    assert m.NEW == ["subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"]
    assert "subj01" not in m.NEW                      # subj01 = reference, not new-D1 cohort
    assert m.ROIS == ["V1", "V2", "V3", "hV4", "ventral", "lateral", "parietal"]
    assert set(m.BETA) == {"B0", "B1"}


def test_near_neutral_threshold_frozen_at_0_02_and_categories():
    m = _drv()
    assert m.NEAR_NEUTRAL == 0.02
    assert m.d1_category(0.05) == "D1_GAIN"
    assert m.d1_category(-0.05) == "D1_LOSS"
    assert m.d1_category(0.01) == "D1_NEAR_NEUTRAL"
    assert m.d1_category(-0.02) == "D1_NEAR_NEUTRAL"   # boundary -> neutral (strict <)


def test_cohort_beta_status_logic():
    m = _drv()
    assert m.cohort_beta_status([0.1, 0.1, 0.1, 0.1, 0.1, -0.1, -0.1]) == "D1_BETA_SENSITIVITY_PERSISTS"
    assert m.cohort_beta_status([-0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.1]) == "D1_BETA_SENSITIVITY_REVERSES"
    assert m.cohort_beta_status([0.1, 0.1, 0.1, -0.1, -0.1, -0.1, 0.0]) == "D1_BETA_SENSITIVITY_MIXED"


def test_no_beta_selection_or_verdict_helpers():
    m = _drv()
    bad = [n for n in dir(m) if any(k in n.lower() for k in
           ("select_beta", "choose_beta", "reproduc", "verdict", "exclude", "winner"))]
    assert bad == []


def test_frozen_config_prohibits_d1b_d2_and_pins_contract():
    cfg = json.loads((REPO / "artifacts/mindcompiler/roy_s2_4r/s2_4r_frozen_config.json").read_text())
    assert cfg["denoising"]["policy"] == "D1_STRICT_CROSSFIT"
    assert cfg["denoising"]["D1b"] == "PROHIBITED" and cfg["denoising"]["D2"] == "PROHIBITED"
    assert cfg["preprocessing"].startswith("PREPROC_ZSCORE_TRAIN_ONLY")
    assert cfg["vis2vis_pairing"].startswith("V2V-P0")
    assert cfg["expected_cells"] == 448
    assert cfg["D1_vs_RAW"]["NEAR_NEUTRAL_ABS_THRESHOLD"] == 0.02
    assert cfg["no_beta_selection"] and cfg["no_reproduction_verdict"]
    assert cfg["immutable_predecessors"]["H_A_CROSS_PARTICIPANT_PARTIAL"] == "frozen, not reopened"


def test_expected_cell_count_is_448():
    # 8 participants x 7 ROIs x 2 betas x 4 folds
    assert 8 * 7 * 2 * 4 == 448


def test_driver_uses_only_D1_denoising(monkeypatch):
    # the compute path must request denoising="D1" (never D1b/RAW) -- static check
    src = DRIVER.read_text()
    assert 'denoising="D1"' in src
    assert 'denoising="D1b"' not in src and "D1b" not in src
