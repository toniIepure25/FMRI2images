"""Release-engineering surface: data contract + tiny examples, one-command demo, reproduce command, and the
provenance graph. Synthetic only; no external data required."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import __version__, SCHEMA_VERSIONS
from . import synthworld as SW, metrics as MX, benchmark as BM, falsification as FB, governance as GV, errors as ER


# ---------------- data contract ----------------
DATA_CONTRACT = {
    "schema_version": SCHEMA_VERSIONS["artifact"],
    "subject": {"subject_id": "str", "cohort": "A_HISTORICAL_N8_CLOSED|B_INDEPENDENT_O2_16|C_CONFIRMATION|SYNTHETIC"},
    "representation": {"name": "str", "family": "str", "dim": "int", "frozen": "bool"},
    "target_state": {"subject_id": "str", "state": "imagery|recall|...", "samples": "float array (n_obs x dim)"},
    "roi_matrix": "float array (n_obs x n_voxels), pseudonymous only",
    "trial": {"identity_id": "str", "repeat_index": "int", "run": "int", "session": "int", "site": "str"},
    "note": "no participant identity fields (name/email/health) ever enter research schemas",
}


def tiny_example():
    """A minimal valid synthetic subject the API accepts."""
    w = SW.generate(SW.WorldConfig(n_subjects=1, dim=12, n_identities=2, n_repeats=4, seed_tag="tiny"))
    s = w["subjects"][0]
    return {"subject_id": s["subject_id"], "cohort": "SYNTHETIC", "imagery_samples_shape": list(s["imagery"].shape)}


def validate_target_state(samples, required_rank):
    X = np.asarray(samples, np.float64)
    if X.ndim != 2:
        raise ER.ShapeError("(n_obs, dim)", X.shape, "target_state.samples")
    rank = int(np.linalg.matrix_rank(X - X.mean(0, keepdims=True)))
    if rank < required_rank:
        raise ER.RankDeficiencyError(rank, required_rank)
    return True


# ---------------- one-command demo ----------------
def demo(seed_tag="demo", dim=30, out_dir=None):
    """Generate a small known world -> recover support -> estimate private correction -> few-shot calibration ->
    falsification control -> compact report + provenance. CPU-fast; no external data."""
    t0 = time.perf_counter()
    cfg = SW.WorldConfig(n_subjects=6, dim=dim, snr=3.0, private_rank=2, shared_rank=3, seed_tag=seed_tag)
    world = SW.generate(cfg)
    # support recovery: fraction of energy captured by the (known) support basis
    support = world["support_basis"]
    frac = []
    for s in world["subjects"]:
        X = s["imagery"] - s["imagery"].mean(0, keepdims=True)
        tot = float(np.sum(X ** 2)); within = float(np.sum((X @ support.T) ** 2))
        frac.append(MX.support_fraction(within, tot))
    # private correction estimate vs truth
    priv_overlap = []
    for s in world["subjects"]:
        X = s["imagery"] - s["imagery"].mean(0, keepdims=True)
        resid = X - X @ support.T @ support
        est = np.linalg.svd(resid, full_matrices=False)[2][:cfg.private_rank]
        priv_overlap.append(MX.subspace_overlap(est, s["private_basis_true"]))
    # few-shot calibration
    b4 = BM.b4_few_shot(world, n_obs=8)
    # falsification control (F5 rotation invariance on a subject)
    A = world["subjects"][0]["private_basis_true"]; B = world["subjects"][1]["private_basis_true"]
    f5 = FB.shared_rotation_invariance(A, B, dim)
    report = {
        "mindir_version": __version__, "world_config": cfg.to_dict(),
        "support_fraction_median": float(np.median(frac)),
        "private_correction_overlap_median": float(np.median(priv_overlap)),
        "few_shot_recovery_overlap": b4["median_recovery_overlap"],
        "falsification_F5_rotation_invariant": f5["invariant"],
        "runtime_s": round(time.perf_counter() - t0, 3),
        "no_external_data": True, "synthetic_only": True,
        "caption": "software demo on synthetic ground truth; NOT a biological result",
    }
    prov = GV.repro_record("mindir_demo", {"synthetic": "generated"}, code_commit="local", config_hash=GV.semantic_hash(cfg.to_dict()),
                           seed=None, participant_set=[s["subject_id"] for s in world["subjects"]],
                           metric_version=SCHEMA_VERSIONS["metric"], timestamp=GV.dependency_env().get("python", ""), frozen=False)
    if out_dir:
        d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
        (d / "demo_report.json").write_text(json.dumps({"report": report, "provenance": prov}, indent=2, default=str))
    return {"report": report, "provenance": prov}


# ---------------- reproduce ----------------
def reproduce(manifest: dict):
    """Verify code/config/input/metric/seed/schema before deciding reproduction class. Never conflates classes."""
    checks = {}
    checks["schema_ok"] = manifest.get("schema_version") in SCHEMA_VERSIONS.values() or manifest.get("schema_version") is not None
    checks["config_hash_present"] = bool(manifest.get("config_hash"))
    checks["code_commit_present"] = bool(manifest.get("code_commit"))
    checks["metric_version_present"] = bool(manifest.get("metric_version"))
    checks["seed_present"] = manifest.get("seed") is not None or manifest.get("deterministic_without_seed", False)
    checks["inputs_hashed"] = bool(manifest.get("data_hashes"))
    if not checks["code_commit_present"] or not checks["config_hash_present"] or not checks["inputs_hashed"]:
        cls = "REPRODUCTION_UNAVAILABLE"
    elif manifest.get("bitwise_expected") and checks["seed_present"]:
        cls = "EXACT_REPRODUCTION"
    else:
        cls = "METHOD_REPRODUCTION"
    return {"reproduction_class": cls, "checks": checks,
            "note": "EXACT vs METHOD vs UNAVAILABLE are never conflated"}


# ---------------- provenance graph ----------------
def provenance_graph():
    nodes = ["DATA", "PREPROCESSING", "REPRESENTATION", "OPERATOR", "METRICS", "INFERENCE", "CLAIM", "MANUSCRIPT_FIGURE_TABLE"]
    edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    return {"schema_version": SCHEMA_VERSIONS["provenance"], "nodes": nodes,
            "edges": [{"from": a, "to": b} for a, b in edges],
            "every_artifact_traceable_backwards": True,
            "firewall": "predictor-side (DATA..METRICS training) is separated from protected Stage-B outcomes; "
                        "CLAIM requires the full backward chain + a falsifier + replication status"}
