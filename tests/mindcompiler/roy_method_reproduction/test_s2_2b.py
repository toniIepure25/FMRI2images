"""S2.2B data-free tests: ROI defs, selection policy, scorecard, sign conventions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fmri2img.mindcompiler.roy_method_reproduction import s2_2b_scorecard as sc
from fmri2img.mindcompiler.roy_method_reproduction import s2_roi as roi

REPO = Path(__file__).resolve().parents[3]


def test_exact_six_prospective_rois_and_v1_excluded():
    assert roi.PROSPECTIVE_ROIS == ("V2", "V3", "hV4", "ventral", "lateral", "parietal")
    assert roi.DISCOVERY_ROI == "V1"
    assert "V1" not in roi.PROSPECTIVE_ROIS


def test_roi_definitions_match_project_mapping():
    assert roi.ROI_DEFS["V2"] == ("prf-visualrois", (3, 4))
    assert roi.ROI_DEFS["V3"] == ("prf-visualrois", (5, 6))
    assert roi.ROI_DEFS["hV4"] == ("prf-visualrois", (7,))
    assert roi.ROI_DEFS["ventral"] == ("streams", (5,))
    assert roi.ROI_DEFS["lateral"] == ("streams", (6,))
    assert roi.ROI_DEFS["parietal"] == ("streams", (7,))


def test_unknown_roi_rejected():
    with pytest.raises(ValueError, match="unknown ROI"):
        roi.select_roi_voxels("V9", "a", "b", "c")


# --- scorecard: sign conventions & criteria ----------------------------------

def _perfect():
    # reliability loss and RAW loss both positive and monotonically related
    L_rel = {"V2": 0.05, "V3": 0.10, "hV4": 0.02, "ventral": 0.08, "lateral": 0.03, "parietal": 0.06}
    L_perf = {"V2": 0.05, "V3": 0.12, "hV4": 0.01, "ventral": 0.09, "lateral": 0.02, "parietal": 0.07}
    return L_rel, L_perf


def test_all_criteria_pass_supported():
    L_rel, L_perf = _perfect()
    s = sc.evaluate_criteria(L_rel, L_perf)
    assert s["criterion_A"] and s["criterion_B"] and s["criterion_C"] and s["criterion_D"]
    assert sc.classify(s) == "H_A_SPATIAL_PROFILE_SUPPORTED"
    assert s["rho_reliability_vs_RAW_loss"] > 0


def test_anti_correlated_not_supported():
    L_rel = {"V2": 0.05, "V3": 0.10, "hV4": 0.02, "ventral": 0.08, "lateral": 0.03, "parietal": 0.06}
    L_perf = {"V2": -0.05, "V3": -0.12, "hV4": -0.01, "ventral": -0.09, "lateral": -0.02, "parietal": -0.07}
    s = sc.evaluate_criteria(L_rel, L_perf)
    assert not s["criterion_B"] and not s["criterion_D"]
    assert sc.classify(s) in ("H_A_SPATIAL_PROFILE_NOT_SUPPORTED", "H_A_SPATIAL_PROFILE_PARTIAL")


def test_inconclusive_when_few_evaluable():
    L_rel = {"V2": 0.05, "V3": 0.10}
    L_perf = {"V2": 0.05, "V3": 0.12}
    s = sc.evaluate_criteria(L_rel, L_perf)
    assert s["n_evaluable"] == 2
    assert sc.classify(s) == "H_A_SPATIAL_PROFILE_INCONCLUSIVE"


def test_sign_convention_positive_means_b1_worse():
    # L = B0 - B1; positive iff B1 lower. Encode directly.
    B0_rel, B1_rel = 0.30, 0.14
    assert (B0_rel - B1_rel) > 0            # B1 lower reliability -> positive loss
    B0_perf, B1_perf = 0.14, -0.04
    assert (B0_perf - B1_perf) > 0          # B1 worse performance -> positive loss


def test_classification_partial_two_or_three():
    # A,B pass but C fails (flat/negative rho) and D fails -> 2 passed -> PARTIAL
    L_rel = {"V2": 0.05, "V3": 0.06, "hV4": 0.04, "ventral": 0.05, "lateral": 0.05, "parietal": 0.05}
    L_perf = {"V2": 0.05, "V3": -0.01, "hV4": -0.02, "ventral": -0.03, "lateral": 0.01, "parietal": -0.01}
    s = sc.evaluate_criteria(L_rel, L_perf)
    assert s["n_criteria_passed"] in (2, 3)
    assert sc.classify(s) == "H_A_SPATIAL_PROFILE_PARTIAL"


def test_leave_one_roi_out():
    L_rel, L_perf = _perfect()
    lo = sc.leave_one_roi_rhos(L_rel, L_perf)
    assert len(lo["rhos"]) == 6 and lo["n_positive"] >= 5


@pytest.mark.skipif(
    not (REPO / "data/nsd/nsddata/ppdata/subj01/func1pt8mm/roi/prf-visualrois.nii.gz").exists(),
    reason="ROI NIfTIs not present")
def test_v1_selection_regression_and_strict_gt():
    R = str(REPO / "data/nsd/nsddata/ppdata/subj01/func1pt8mm/roi")
    P = str(REPO / "data/nsd/nsddata/ppdata/subj01/func1pt8mm")
    N = str(REPO / "data/nsd/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf/ncsnr.nii.gz")
    r = roi.select_roi_voxels("V1", R, P, N)
    assert r["voxel_hash"] == "a1bc56fe7c55"          # discovery regression
    assert r["selected_voxels"] == 27
    assert r["ge_sensitivity_count"] == r["selected_voxels"] + r["threshold_ties"]  # strict > vs >=
