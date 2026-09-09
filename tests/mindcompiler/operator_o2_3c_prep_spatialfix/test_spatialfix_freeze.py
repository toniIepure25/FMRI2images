"""O2.3C-PREP-SPATIALFIX freeze tests (data-free, offline).

Assert the mean-EPI spatial-reference correction is frozen and self-consistent: meanFIRST5 fixed (not
outcome-selected), mean.nii not a competitor, R2 not used, moving = deterministic temporal mean,
rigid+affine only (no nonlinear), interpolation unchanged, temporal correction reused (no lag search, no
candidate resampling), thresholds unchanged, and all immutable history (historical R4 + BENCHFIX failures,
validated temporal correction, O2/O3) preserved. Result artifacts are honest PENDING placeholders here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
S = ROOT / "artifacts/mindcompiler/operator_o2_3c_prep_spatialfix"
BF = ROOT / "artifacts/mindcompiler/operator_o2_3c_prep_benchfix"
PREP = ROOT / "artifacts/mindcompiler/operator_o2_3c_prep"


def _j(base, n):
    return json.loads((base / n).read_text())


def test_frozen_config_sha_matches():
    cfg = _j(S, "spatialfix_frozen_config.json")
    stored = cfg.pop("frozen_config_sha")
    rec = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]
    assert rec == stored == "5e3954d9"


def test_prospective_not_optimization():
    cfg = _j(S, "spatialfix_frozen_config.json")
    assert cfg["analysis_class"] == "PROSPECTIVE_CORRECTION_OF_CONTRAST_MISMATCHED_SPATIAL_REFERENCE"
    assert cfg["corrected_target_fixed_before_metrics"] is True
    assert cfg["no_other_preprocessing_change"] is True
    r = cfg["no_optimization_rule"]
    assert r["only_evaluated_pipeline"] == "candidate temporal mean -> meanFIRST5 -> rigid+affine"
    assert r["if_it_fails_the_gate_fails"] is True


def test_meanfirst5_fixed_not_outcome_selected():
    ref = _j(S, "spatialfix_frozen_config.json")["primary_fixed_reference"]
    assert ref["name"] == "NSD_FUNC1PT8MM_MEANFIRST5"
    assert ref["s3_key"].endswith("func1pt8mm/meanFIRST5.nii.gz")
    assert ref["sha256"] == "d94d96db6ad79cddd3f539ad2df3b7ca8b2b0be2ce6a23a0cb154b47b0684755"
    assert ref["not_outcome_selected"] is True


def test_mean_nii_not_competitor_r2_not_used():
    cfg = _j(S, "spatialfix_frozen_config.json")
    assert "mean.nii.gz (competitor)" in cfg["forbidden_fixed_alternatives"]
    assert "R2.nii.gz" in cfg["forbidden_fixed_alternatives"]
    inv = _j(S, "reference_inventory.json")
    assert inv["mean_inventory_only"]["used_as_competitor"] is False
    assert inv["R2_used"] is False


def test_moving_is_deterministic_temporal_mean():
    m = _j(S, "spatialfix_frozen_config.json")["moving_reference"]
    assert m["definition"] == "TEMPORAL_MEAN_OF_EXISTING_FMRIPREP_DESC_PREPROC_BOLD"
    assert m["temporal_filtering"] is False and m["gt_involvement"] is False and m["outcome_adaptation"] is False


def test_rigid_affine_only_no_nonlinear():
    t = _j(S, "spatialfix_frozen_config.json")["transform"]
    assert t["transforms"] == ["rigid", "affine"] and t["metric"] == "Mattes MI"
    assert t["nonlinear"] is False and t["syn"] is False and t["bspline"] is False
    assert t["params"]["reused_from_BENCHFIX_byte_for_byte"] is True


def test_interpolation_unchanged():
    i = _j(S, "spatialfix_frozen_config.json")["interpolation"]
    assert i["bold_4d"] == "LanczosWindowedSinc" and i["binary_masks"] == "NearestNeighbor" and i["unchanged"] is True


def test_temporal_correction_reused_not_reopened():
    tc = _j(S, "spatialfix_frozen_config.json")["temporal_correction_reused_from_benchfix"]
    assert tc["closed_and_validated"] is True and tc["reopen"] is False
    assert tc["no_lag_search"] is True and tc["candidate_resampled"] is False


def test_candidate_byte_identical_expected():
    ci = _j(S, "spatialfix_frozen_config.json")["candidate_immutability"]
    assert ci["fmriprep_rerun"] is False
    assert ci["expected_desc_preproc_bold_sha256_16"] == "3838946fec582b54"
    # must match the BENCHFIX-recorded candidate hash exactly
    assert _j(BF, "candidate_immutability.json")["artifacts"]["desc-preproc_bold"]["sha256_16"] == "3838946fec582b54"


def test_thresholds_unchanged_vs_benchfix():
    q = _j(S, "spatialfix_frozen_config.json")["quality_metrics_frozen_UNCHANGED"]
    bf = _j(BF, "benchfix_frozen_config.json")["quality_metrics_frozen_UNCHANGED"]
    assert q["brainmask_dice_min"] == bf["brainmask_dice_min"] == 0.90
    assert q["mean_volume_spatial_r_min"] == bf["mean_volume_spatial_r_min"] == 0.95
    assert q["roi_mean_temporal_r_min"] == bf["roi_mean_temporal_r_min"] == 0.90
    assert q["voxelwise_temporal_r_median_min"] == bf["voxelwise_temporal_r_median_min"] == 0.70
    assert q["tsnr_ratio_range"] == bf["tsnr_ratio_range"] == [0.50, 2.00]
    assert q["roi_set"] == bf["roi_set"]


def test_immutable_history_preserved():
    h = _j(S, "spatialfix_frozen_config.json")["immutable_history"]
    assert h["historical_R4"] == "O2_3C_PREP_TASK_BENCHMARK_FAILURE"
    assert h["BENCHFIX"] == "O2_3C_PREP_CORRECTED_BENCHMARK_FAILURE"
    assert h["BENCHFIX_temporal_correction"] == "VALIDATED"
    assert h["O2"] == "SHARED_OPERATOR_PARTIAL"
    assert h["O2.3A"] == "CORE_ANCHOR_TARGET_ORIENTATION_NOT_IDENTIFIABLE"
    assert h["O2.3C"] == "O2_3C_REST_PRODUCT_INCOMPATIBLE"
    assert h["O3"] == "O3_NOT_READY"
    # historical status files themselves untouched
    assert _j(PREP, "r4_benchmark_status.json")["status"] == "O2_3C_PREP_TASK_BENCHMARK_FAILURE"
    assert _j(BF, "corrected_benchmark_status.json")["status"] == "O2_3C_PREP_CORRECTED_BENCHMARK_FAILURE"


def test_status_logic_present():
    r = _j(S, "spatialfix_frozen_config.json")["status_rules"]
    for k in ["O2_3C_PREP_SPATIALFIX_PASS", "O2_3C_PREP_SPATIALFIX_FAILURE",
              "O2_3C_PREP_SPATIALFIX_REFERENCE_UNAVAILABLE", "O2_3C_PREP_SPATIALFIX_TECHNICAL_FAILURE"]:
        assert k in r


def test_results_pending_not_fabricated():
    for n in ["meanFIRST5_certification.json", "candidate_immutability.json", "moving_reference_certification.json",
              "spatial_transform_certification.json", "matched_grid_certification.json", "spatialfix_metrics.json",
              "spatialfix_status.json", "author_transform_asset_inventory.json"]:
        assert _j(S, n)["status"] == "PENDING_EVALUATION"
