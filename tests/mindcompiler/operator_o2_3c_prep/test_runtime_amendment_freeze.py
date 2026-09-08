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


def test_amendment_status_sealed_dependency_failure():
    s = _j("runtime_amendment_status.json")
    assert s["status"] == "RUNTIME_AMENDMENT_SEALED"
    assert s["terminal_status"] == "O2_3C_PREP_BAREMETAL_DEPENDENCY_FAILURE"
    assert s["contract_sha"] == "24e6ca83"
    assert s["no_scientific_track_O_status_modified"] is True
    # recommended resolution points at the official container on a Docker host
    assert s["recommended_resolution"]["path"] == "O2_3C_PREP_RUNTIME_HOST_REQUIRED"
    assert s["recommended_resolution"]["external_host_spec"]["image"] == "nipreps/fmriprep:25.2.5"


def test_fs_license_resolved_hash_only_never_committed():
    # license blocker resolved by the user; recorded by hash only, never contents
    ls = _j("fs_license_status.json")
    assert ls["fs_license_present_on_pvc"] is True
    assert ls["fs_license_sha256"] == "6f7afab5b5201aa8b0aca10e29ff04802c6d279a0a1c7ddfeae9dd13ae52a152"
    assert ls["contents_committed_or_printed"] is False
    ep = _j("execution_provenance.json")["runtime_amendment"]
    assert ep["fs_license_present_on_pvc"] is True
    assert ep["fs_license_contents_committed_or_printed"] is False
    # the actual license file must never be committed to the repo
    P_root = ROOT
    assert not (P_root / "artifacts/mindcompiler/operator_o2_3c_prep/freesurfer_license.txt").exists()


def test_dependency_equivalence_failure_evidence_based():
    inv = _j("baremetal_dependency_inventory.json")
    auth = inv["authoritative_container_versions"]
    assert auth["ants"]["container"] == "2.6.2" and auth["ants"]["frozen_pin"] == "2.5.1"
    assert auth["connectome_workbench"]["container"] == "2.0.1" and auth["connectome_workbench"]["frozen_pin"] == "1.5.0"
    assert "componentized" in auth["fsl"]["container"]
    f = inv["container_equivalence_finding"]
    assert f["conclusion"] == "frozen bare-metal dependency set cannot be certified container-equivalent"
    assert len(f["definitive_mismatches"]) >= 3


def test_r3_r4_not_reached_and_r2_halted_honest():
    # bare-metal branch record (superseded by the orchestraiq direct-K8s path) stays honest:
    assert _j("baremetal_environment_manifest.json")["status"] == "HALTED_DEPENDENCY_EQUIVALENCE_FAILURE"
    assert _j("baremetal_environment_manifest.json")["installed_versions"]["fmriprep"] == "25.2.5"
    # the bare-metal placeholders remain NOT_REACHED (resume_certification.json was repurposed for the
    # orchestraiq R3 certification and is asserted separately in test_r3_resume_certified)
    for n in ["persistent_workdir_certification.json", "single_run_runtime_result.json"]:
        assert _j(n)["status"] == "NOT_REACHED_DEPENDENCY_FAILURE"
    # no benchmark metric fabricated
    assert _j("single_run_runtime_result.json")["benchmark"] is None


def test_host_migration_pending_and_honest():
    hm = _j("host_migration_provenance.json")
    assert hm["migration_class"] == "TECHNICAL_HOST_RUNTIME_MIGRATION_ONLY"
    assert hm["status"] == "ORCHESTRAIQ_DIRECT_K8S_FMRIPREP_FEASIBLE_PENDING_EXECUTION_APPROVAL"
    assert hm["orchestraiq_audit"]["verdict"] == "ORCHESTRAIQ_DIRECT_K8S_FMRIPREP_FEASIBLE"
    assert hm["orchestraiq_audit"]["cluster_mutation_performed"] is False
    assert hm["source_head"].startswith("4377100")
    assert hm["official_image"]["tag"] == "nipreps/fmriprep:25.2.5"
    # host-dependent fields must be explicit PENDING, never fabricated
    assert hm["official_image"]["full_repo_digest"] == "PENDING_HOST_PROVISIONING"
    assert hm["host_facts"]["os"] == "PENDING_HOST_PROVISIONING"
    assert hm["R4_benchmark"]["status"] == "PENDING_HOST_PROVISIONING"
    assert hm["no_scientific_methodology_change"] is True
    # license recorded hash-only, never committed
    assert hm["freesurfer_license"]["contents_committed_or_printed"] is False
    assert hm["freesurfer_license"]["sha256"] == "6f7afab5b5201aa8b0aca10e29ff04802c6d279a0a1c7ddfeae9dd13ae52a152"
    # bare-metal terminal preserved as history
    assert "O2_3C_PREP_BAREMETAL_DEPENDENCY_FAILURE" in hm["preserves"]["bare_metal_terminal"]


def test_benchmark_inputs_verified_on_s3():
    m = _j("benchmark_input_manifest.json")
    assert m["all_verified_present"] is True
    assert "run-01_bold.nii.gz" in m["inputs"]["raw_task_bold"]
    assert "phasediff" in m["inputs"]["fieldmap_phasediff"]
    assert "aparc+aseg.mgz" in m["inputs"]["freesurfer_aparc_aseg"]
    # no benchmark result fabricated anywhere
    assert _j("single_run_runtime_result.json")["benchmark"] is None


def test_orchestraiq_audit_verdict_feasible_and_readonly():
    a = _j("orchestraiq_infra_audit.json")
    assert a["verdict"] == "ORCHESTRAIQ_DIRECT_K8S_FMRIPREP_FEASIBLE"
    assert a["avoids_aws"] is True
    assert a["does_not_change_scientific_methodology"] is True
    # audit performed no cluster mutation
    assert a["persistence_test"]["executed"] is False
    assert "READ_ONLY" in a["audit_class"]
    # official image runs as native pod; egress + admission confirmed read-only
    assert "CONFIRMED" in a["container_runtime_capability"]["image_pull_egress"]
    assert "ACCEPTED" in a["container_runtime_capability"]["official_image_admission"]
    # the previously-unstable resource identified as the interactive workload, not batch jobs
    assert a["orchestraiq_workload_reconciliation"]["same_as_previously_tested_unstable_pod"] is True
    assert a["stability"]["batch_jobs_run_to_completion"] is True
    # preserves frozen states
    assert a["preserves"]["bare_metal_terminal"].startswith("O2_3C_PREP_BAREMETAL_DEPENDENCY_FAILURE")
    assert a["preserves"]["O3"] == "O3_NOT_READY"


def test_r3_resume_certified():
    r = _j("resume_certification.json")
    assert r["status"] == "PERSISTENT_NIPYPE_RESUME_CERTIFIED"
    assert r["new_pod_uid_confirmed"] is True
    assert r["completed_nodes_valid"] >= 3 and r["evidence_classes"] >= 2
    assert r["cache_corruption"] is False


def test_r4_benchmark_failure_temporal_incompatibility():
    s = _j("r4_benchmark_status.json")
    assert s["status"] == "O2_3C_PREP_TASK_BENCHMARK_FAILURE"
    assert s["stop"] is True
    assert s["cohort_phases_2_4"].startswith("NOT AUTHORIZED")
    m = _j("r4_benchmark_metrics.json")
    assert m["candidate"]["nvols"] == 188 and m["ground_truth"]["nvols"] == 226
    assert m["temporal_alignment_guard"]["verdict"] == "FAIL"
    assert m["thresholds_weakened"] is False
    assert m["pass_contract_satisfiable"] is False
    # transform not applied (no inverse-direction risk)
    assert _j("r4_transform_certification.json")["transform_applied"] is False


def test_o3_and_track_o_preserved_after_benchmark():
    imm = _j("scientific_status.json")["immutable"]
    assert imm["O2"] == "SHARED_OPERATOR_PARTIAL"
    assert imm["O2.3A"] == "CORE_ANCHOR_TARGET_ORIENTATION_NOT_IDENTIFIABLE"
    assert imm["O2.3C_first_attempt"] == "O2_3C_REST_PRODUCT_INCOMPATIBLE"
    assert imm["O3"] == "O3_NOT_READY"


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
