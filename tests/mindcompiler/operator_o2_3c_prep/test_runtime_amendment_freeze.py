"""O2.3C-PREP-RUNTIME amendment freeze tests (data-free, offline).

Assert the R0 runtime-amendment contract is frozen and self-consistent, that it changes ONLY runtime
packaging (never scientific methodology / benchmark / lane algorithm / Track-O status), that the
historical blocker and Phase-0 freeze are preserved, and that all R1-R4 result artifacts are honest
PENDING placeholders -- never fabricated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "artifacts/mindcompiler/operator_o2_3c_prep"


def _j(n):
    return json.loads((P / n).read_text())


def test_amendment_contract_sha_matches_content():
    c = _j("runtime_amendment_contract.json")
    stored = c.pop("contract_sha")
    rec = hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]
    assert rec == stored == "24e6ca83"


def test_amendment_is_runtime_only_and_preserves_methodology():
    c = _j("runtime_amendment_contract.json")
    assert c["amendment_class"] == "TECHNICAL_RUNTIME_PACKAGING_ONLY"
    assert c["preserves"]["phase0_frozen_config_sha"] == "2d1a26a5"
    assert c["preserves"]["lane_B_algorithm"] == "fMRIPrep 25.2.5"
    assert c["preserves"]["historical_first_attempt"] == "O2_3C_REST_PRODUCT_INCOMPATIBLE"
    assert c["preserves"]["O3"] == "O3_NOT_READY"
    inv = c["invariants"]
    assert inv["no_benchmark_policy_change"] is True
    assert inv["no_registration_policy_change"] is True
    assert inv["no_scientific_methodology_change"] is True
    assert inv["target_imagery_files_opened"] == 0


def test_fmriprep_version_and_pins_unchanged():
    rt = _j("runtime_amendment_contract.json")["runtime_target"]
    assert rt["fmriprep"] == "25.2.5"
    ext = rt["external_dependencies_pinned"]
    assert ext["FSL"] == "6.0.7.7"
    assert ext["ANTs"] == "2.5.1"
    assert ext["AFNI"] == "24.0.05"
    assert ext["FreeSurfer"] == "7.3.2"
    assert ext["connectome-workbench"] == "1.5.0"


def test_persistent_paths_on_pvc():
    pp = _j("runtime_amendment_contract.json")["persistent_paths"]
    assert pp["all_on_persistent_PVC"] is True
    assert pp["no_ephemeral_for_post_restart_artifacts"] is True
    assert pp["env_prefix"].startswith("/home/jovyan/work")
    assert pp["work_dir"].startswith("/home/jovyan/work")
    assert pp["output_dir"].startswith("/home/jovyan/work")


def test_freesurfer_license_not_fabricated():
    fs = _j("runtime_amendment_contract.json")["freesurfer_license_policy"]
    assert fs["do_not_fabricate_or_download"] is True
    assert fs["if_required_stop_at"] == "O2_3C_PREP_FS_LICENSE_REQUIRED"
    env = _j("runtime_amendment_contract.json")["environment_variables_frozen"]
    assert "provided/fabricated" in env["note"] and "NOT" in env["note"]


def test_hard_stops_enumerated():
    cat = _j("runtime_amendment_contract.json")["terminal_status_catalog"]
    for s in ["O2_3C_PREP_BAREMETAL_RUNTIME_PASS", "O2_3C_PREP_BAREMETAL_DEPENDENCY_FAILURE",
              "O2_3C_PREP_BAREMETAL_PREEMPTION_BLOCKED", "O2_3C_PREP_FS_LICENSE_REQUIRED",
              "O2_3C_PREP_RUNTIME_HOST_REQUIRED", "O2_3C_PREP_RUNTIME_AMENDMENT_FAILURE"]:
        assert s in cat


def test_amendment_status_sealed_fs_license_required():
    s = _j("runtime_amendment_status.json")
    assert s["status"] == "RUNTIME_AMENDMENT_SEALED"
    assert s["terminal_status"] == "O2_3C_PREP_FS_LICENSE_REQUIRED"
    assert s["contract_sha"] == "24e6ca83"
    assert s["runtime_installable_user_space"] is True
    assert s["container_blocker_bypassed"] is True
    assert s["no_scientific_track_O_status_modified"] is True
    # exact licensed tool named (mri_coreg + unconditional gate)
    tool = s["exact_tool_requiring_license"]
    assert "mri_coreg" in tool["licensed_binary"]
    assert "check_valid_fs_license" in tool["unconditional_license_gate"]


def test_fs_license_determination_evidence_based_and_absent():
    inv = _j("baremetal_dependency_inventory.json")
    d = inv["freesurfer_license_determination"]
    assert d["required"] is True and d["present_on_pvc"] is False
    assert "installed fMRIPrep 25.2.5" in d["proven_from"]
    fs = next(x for x in inv["dependencies"] if x["name"] == "FreeSurfer")
    assert fs["invoked_by_frozen_path"] is True and fs["determination"] == "REQUIRED"
    # license neither fabricated nor downloaded
    ep = _j("execution_provenance.json")["runtime_amendment"]
    assert ep["fs_license_present_on_pvc"] is False
    assert ep["fs_license_fabricated_or_downloaded"] is False


def test_r3_r4_not_reached_and_r2_partial_honest():
    assert _j("baremetal_environment_manifest.json")["status"].startswith("PARTIAL_PYTHON_LAYER_INSTALLED")
    assert _j("baremetal_environment_manifest.json")["installed_versions"]["fmriprep"] == "25.2.5"
    for n in ["persistent_workdir_certification.json", "resume_certification.json", "single_run_runtime_result.json"]:
        assert _j(n)["status"] == "NOT_REACHED_FS_LICENSE_REQUIRED"
    # no benchmark metric fabricated
    assert _j("single_run_runtime_result.json")["benchmark"] is None


def test_prior_phase0_and_immutables_untouched():
    # Phase-0 frozen config SHA still recomputes; Track-O immutables preserved.
    cfg = _j("o2_3c_prep_frozen_config.json")
    stored = cfg.pop("frozen_config_sha")
    rec = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]
    assert rec == stored == "2d1a26a5"
    imm = _j("scientific_status.json")["immutable"]
    assert imm["O2.3C_first_attempt"] == "O2_3C_REST_PRODUCT_INCOMPATIBLE"
    assert imm["O3"] == "O3_NOT_READY"
    # terminal execution status still not fabricated
    assert _j("scientific_status.json")["terminal_execution_status"] is None
    assert _j("scientific_status.json")["phase0_status"] == "PHASE0_FROZEN"
