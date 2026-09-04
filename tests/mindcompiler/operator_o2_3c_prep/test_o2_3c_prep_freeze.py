"""O2.3C-PREP Phase-0 freeze tests (data-free, offline).

Assert the FROZEN technical contracts + lane decision + deterministic selections only. No preprocessing
compute is exercised (Phases 1-4 are pod-blocked); result artifacts are asserted to be honest PENDING
placeholders, never fabricated. Also enforce prior-gate immutability incl. the historical
O2_3C_REST_PRODUCT_INCOMPATIBLE, and the no-imagery-access guard.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "artifacts/mindcompiler/operator_o2_3c_prep"
O23C = ROOT / "artifacts/mindcompiler/operator_o2_3c"
O2 = ROOT / "artifacts/mindcompiler/operator_o2"


def _j(base, n):
    return json.loads((base / n).read_text())


def test_frozen_config_sha_matches_content():
    cfg = _j(P, "o2_3c_prep_frozen_config.json")
    stored = cfg.pop("frozen_config_sha")
    rec = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]
    assert rec == stored == "2d1a26a5"


def test_prior_o2_3c_immutable_and_historical_blocker_preserved():
    s = _j(O23C, "scientific_status.json")
    assert s["execution_status"] == "O2_3C_REST_PRODUCT_INCOMPATIBLE"          # historical, unchanged
    assert s["orientation_status"] == "CONNECTIVITY_ANCHOR_TARGET_ORIENTATION_INCONCLUSIVE"
    assert _j(O2, "scientific_status.json")["scientific_status"] == "SHARED_OPERATOR_PARTIAL"
    imm = _j(P, "scientific_status.json")["immutable"]
    assert imm["O2.3C_first_attempt"] == "O2_3C_REST_PRODUCT_INCOMPATIBLE"
    assert imm["TRACK_R"] == "INDEPENDENT_METHOD_REPRODUCTION_PARTIAL"
    assert imm["O3"] == "O3_NOT_READY"


def test_no_imagery_access_guard():
    prov = _j(P, "execution_provenance.json")
    assert prov["target_imagery_files_opened_by_PREP"] == 0
    assert prov["o2_2_residual_modified"] is False
    g = _j(P, "o2_3c_prep_frozen_config.json")["guards"]
    assert g["target_imagery_files_opened_by_PREP"] == 0
    assert g["o2_2_residual_hash_checked_only"] is True


def test_lane_A_ineligible_lane_B_selected():
    r = _j(P, "lane_selection_result.json")
    assert r["lane_A_eligibility"]["eligible"] is False
    assert r["lane_A_eligibility"]["status"] == "NSD_AUTHOR_LINEAGE_NOT_EXECUTABLE_FROM_PUBLIC_RELEASE"
    assert r["lane_A_eligibility"]["decided_before_any_prepared_rest_QC_metric"] is True
    assert r["PREPROCESSING_LANE"] == "FMRIPREP_BIDS_RECONSTRUCTION"


def test_no_lane_switch_after_benchmark_contract():
    c = _j(P, "lane_selection_contract.json")
    assert c["no_lane_switch_after_benchmark"] is True
    assert c["no_third_lane"] is True and c["two_lanes_only"] is True


def test_fmriprep_pinned_no_latest():
    lb = _j(P, "o2_3c_prep_frozen_config.json")["lane_B"]
    assert lb["fmriprep_version"] == "25.2.5"
    assert lb["container_name"] == "nipreps/fmriprep"
    assert lb["no_latest_tag"] is True
    assert lb["no_gsr"] is True and lb["no_ica_aroma"] is True and lb["no_smoothing"] is True
    assert lb["temporal_cleaning_here"] is False


def test_exact_func1pt8mm_grid():
    ref = _j(P, "o2_3c_prep_frozen_config.json")["func1pt8mm_reference"]
    assert ref["grid"] == [81, 104, 83]


def test_registration_frozen_before_benchmark():
    reg = _j(P, "func1pt8mm_mapping_contract.json")
    assert reg["transforms"] == ["rigid", "affine"] and reg["no_nonlinear_SyN_rest_to_func"] is True
    assert reg["cost_metric"] == "Mattes mutual information"
    assert reg["bold_interpolation"] == "LanczosWindowedSinc"
    assert reg["atlas_interpolation"] == "NearestNeighbor"


def test_deterministic_benchmark_selection():
    with open(P / "task_benchmark_manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    for r in rows:
        assert r["session"] == "ses-nsd01"
        assert r["prepared_reference"] == "timeseries_session01_run01.nii.gz"
        assert "run-01" in r["raw_task_run"]


def test_benchmark_thresholds_frozen():
    t = _j(P, "task_benchmark_contract.json")["pass_thresholds"]
    assert t["brainmask_dice_min"] == 0.90
    assert t["mean_volume_spatial_r_min"] == 0.95
    assert t["roi_mean_temporal_r_min"] == 0.90
    assert t["voxelwise_temporal_r_min"] == 0.70
    assert t["no_subject_primary_roi_voxel_r_below"] == 0.50
    assert t["group_tsnr_ratio_range"] == [0.50, 2.00]


def test_motion_units_and_motion24():
    m = _j(P, "motion_contract.json")
    assert m["translation_units"] == "mm" and m["rotation_units"] == "rad" and m["params"] == 6
    assert "24" in m["motion24"]
    assert m["final_run_rejection_owned_by"] == "O2.3C (not PREP)"   # no scientific FD exclusion in PREP


def test_wm_csf_categorical_mapping():
    t = _j(P, "tissue_mask_contract.json")
    assert t["mapping_interpolation"] == "NearestNeighbor"
    assert "aparc+aseg" in t["wm_csf_source"]


def test_dk_exactly_68_and_nn_mapping():
    dk = _j(P, "DK_mapping_contract.json")
    assert dk["n_cortical_parcels"] == 68
    assert dk["interpolation"].startswith("NearestNeighbor")
    assert "nsd_mapdata" in dk["mapping"]
    assert dk["scientific_target_filtering_owned_by"].startswith("O2.3C")
    with open(P / "DK_parcel_manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 68
    assert len({r["fs_name"] for r in rows}) == 68
    assert sum(1 for r in rows if r["hemisphere"] == "lh") == 34
    assert not any(r["region"] == "corpuscallosum" for r in rows)   # DK-68 excludes corpuscallosum


def test_product_label_not_official_nsd():
    lbl = _j(P, "o2_3c_prep_frozen_config.json")["product_label"]
    assert lbl["name"] == "MINDIR_DERIVED_NSD_REST_FUNC1PT8MM"
    assert lbl["must_not_claim_official_nsd"] is True
    assert lbl["preprocessing_lane_field"] is True


def test_no_rest_timepoint_removal_or_denoise_in_prep():
    lb = _j(P, "o2_3c_prep_frozen_config.json")["lane_B"]
    assert lb["no_scientific_denoising"] is True
    assert lb["temporal_cleaning_here"] is False


def test_compute_results_are_honest_pending_not_fabricated():
    # every compute-derived artifact must be an explicit PENDING placeholder, never a value
    for n in ["task_benchmark_results.json", "func1pt8mm_mapping_certification.json",
              "tissue_mask_certification.json", "DK_mapping_certification.json",
              "cohort_product_certification.json", "task_benchmark_status.json"]:
        d = _j(P, n)
        assert d["status"] == "PENDING_POD_COMPUTE"
    st = _j(P, "scientific_status.json")
    assert st["terminal_execution_status"] is None            # no terminal verdict fabricated
    assert st["phase0_status"] == "PHASE0_FROZEN"


def test_o3_remains_locked():
    assert _j(P, "scientific_status.json")["immutable"]["O3"] == "O3_NOT_READY"
