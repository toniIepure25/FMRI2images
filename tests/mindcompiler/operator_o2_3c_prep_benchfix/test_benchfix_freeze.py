"""O2.3C-PREP-BENCHFIX freeze tests (data-free, offline).

Assert the corrected-comparator config is frozen and self-consistent, the temporal axes/interpolation are
exactly the prospectively-derived ones (no lag search, candidate never resampled, GT-only resampling),
the spatial transform direction and target grid are frozen, quality thresholds are UNCHANGED from the
historical benchmark, and all immutable history (incl. the permanent historical R4 failure, O2/O2.3A/
O2.3C/O3) is preserved. Compute-derived results are asserted to be honest PENDING placeholders here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
B = ROOT / "artifacts/mindcompiler/operator_o2_3c_prep_benchfix"
PREP = ROOT / "artifacts/mindcompiler/operator_o2_3c_prep"


def _j(base, n):
    return json.loads((base / n).read_text())


def test_frozen_config_sha_matches():
    cfg = _j(B, "benchfix_frozen_config.json")
    stored = cfg.pop("frozen_config_sha")
    rec = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]
    assert rec == stored == "4b384f31"


def test_analysis_class_prospective_not_outcome_adaptive():
    cfg = _j(B, "benchfix_frozen_config.json")
    assert cfg["analysis_class"] == "PROSPECTIVE_CORRECTION_OF_INVALID_TECHNICAL_BENCHMARK_COMPARATOR"
    assert cfg["reason_known_before_any_metric"] is True
    assert cfg["outcome_adaptive_parameter_selection"] is False


def test_time_axes_exact():
    t = _j(B, "benchfix_frozen_config.json")["time_axes"]
    assert t["candidate"]["TR_RAW_s"] == 1.6 and t["candidate"]["N_CANDIDATE"] == 188
    assert t["candidate"]["offset_s"] == 0.8 and t["candidate"]["offset_prospectively_derived_not_estimated"] is True
    assert t["candidate"]["t_min_s"] == 0.8 and t["candidate"]["t_max_s"] == 300.0
    assert t["official_nsd"]["TR_NSD_s_exact"] == "4/3" and t["official_nsd"]["N_GT"] == 226
    assert abs(t["official_nsd"]["TR_NSD_s_float"] - 4.0 / 3.0) < 1e-12
    assert t["official_nsd"]["t_min_s"] == 0.0 and t["official_nsd"]["t_max_s"] == 300.0
    # bounds: candidate query inside NSD support
    assert t["candidate"]["t_min_s"] >= t["official_nsd"]["t_min_s"]
    assert t["candidate"]["t_max_s"] <= t["official_nsd"]["t_max_s"]


def test_harmonization_gt_only_no_lag():
    cfg = _j(B, "benchfix_frozen_config.json")
    h = cfg["harmonization"]
    assert h["direction"].startswith("resample OFFICIAL_NSD_REFERENCE -> CANDIDATE")
    assert h["candidate_temporally_resampled"] is False and h["result_nvols"] == 188
    itp = cfg["interpolation"]
    assert itp["implementation"] == "scipy.interpolate.CubicSpline"
    assert itp["extrapolation"].startswith("FORBIDDEN")
    assert itp["no_lag_search"] is True and itp["no_offset_search"] is True and itp["no_phase_search"] is True
    for bad in ["shift candidate +/-1 TR", "DTW", "temporal warping"]:
        assert bad in cfg["forbidden"]


def test_spatial_transform_frozen_direction():
    s = _j(B, "spatial_transform_contract.json")
    assert s["transforms"] == ["rigid", "affine"] and s["metric"] == "Mattes MI"
    assert s["bold_interpolation"] == "LanczosWindowedSinc" and s["atlas_interpolation"] == "NearestNeighbor"
    assert s["nonlinear"] is False and s["gt_timeseries_used_for_registration"] is False
    assert s["target_shape"] == [81, 104, 83]
    assert "coreg_boldref" in s["moving"] and "R2.nii.gz" in s["fixed"]


def test_quality_thresholds_unchanged_vs_historical():
    q = _j(B, "benchfix_frozen_config.json")["quality_metrics_frozen_UNCHANGED"]
    hist = _j(PREP, "task_benchmark_contract.json")["pass_thresholds"]
    assert q["brainmask_dice_min"] == hist["brainmask_dice_min"] == 0.90
    assert q["mean_volume_spatial_r_min"] == hist["mean_volume_spatial_r_min"] == 0.95
    assert q["roi_mean_temporal_r_min"] == hist["roi_mean_temporal_r_min"] == 0.90
    assert q["voxelwise_temporal_r_median_min"] == hist["voxelwise_temporal_r_min"] == 0.70
    assert q["tsnr_ratio_range"] == hist["group_tsnr_ratio_range"] == [0.50, 2.00]
    assert q["roi_set"] == {"ventral": 5, "lateral": 6, "parietal": 7, "atlas": "NSD streams.nii.gz (func1pt8mm)"}


def test_tsnr_uses_matched_gt():
    t = _j(B, "benchfix_frozen_config.json")["tsnr_rule"]
    assert t["gt_tSNR_on"].startswith("NSD_GT_MATCHED_1P6_EFFECTIVE")
    assert t["do_not_use_original_226_gt_tsnr"] is True


def test_status_rules_present():
    r = _j(B, "benchfix_frozen_config.json")["status_rules"]
    for k in ["O2_3C_PREP_CORRECTED_BENCHMARK_PASS", "O2_3C_PREP_CORRECTED_BENCHMARK_FAILURE",
              "O2_3C_PREP_CORRECTED_BENCHMARK_TECHNICAL_FAILURE"]:
        assert k in r


def test_historical_failure_and_immutables_preserved():
    cfg = _j(B, "benchfix_frozen_config.json")["immutable_history"]
    assert cfg["historical_R4"] == "O2_3C_PREP_TASK_BENCHMARK_FAILURE"
    assert cfg["R3"] == "PERSISTENT_NIPYPE_RESUME_CERTIFIED"
    assert cfg["O2"] == "SHARED_OPERATOR_PARTIAL"
    assert cfg["O2.3A"] == "CORE_ANCHOR_TARGET_ORIENTATION_NOT_IDENTIFIABLE"
    assert cfg["O2.3C"] == "O2_3C_REST_PRODUCT_INCOMPATIBLE"
    assert cfg["O3"] == "O3_NOT_READY"
    # the historical R4 status file itself is untouched
    assert _j(PREP, "r4_benchmark_status.json")["status"] == "O2_3C_PREP_TASK_BENCHMARK_FAILURE"
    rel = _j(B, "benchfix_frozen_config.json")["relation_to_historical"]
    assert rel["does_not_turn_old_into_pass"] is True


def test_interpretation_boundary_forbids_overclaim():
    ib = _j(B, "benchfix_frozen_config.json")["interpretation_boundary"]
    assert "fMRIPrep reproduces NSD preprocessing" in ib["forbidden_claims"]
    assert "bitwise equivalent" in ib["forbidden_claims"]


def test_results_pending_not_fabricated():
    for n in ["candidate_immutability.json", "gt_immutability.json", "synthetic_interpolation_test.json",
              "spatial_transform_certification.json", "matched_grid_certification.json",
              "corrected_benchmark_metrics.json", "corrected_benchmark_status.json"]:
        assert _j(B, n)["status"] == "PENDING_EVALUATION"
