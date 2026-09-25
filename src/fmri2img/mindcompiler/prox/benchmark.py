"""MINDIR benchmark harness (synthetic ground truth now; real data later). Tasks B1-B8 with transparent
baselines scored against known truth. Local leaderboard only; no external SOTA claim."""
from __future__ import annotations

import numpy as np

from . import synthworld as SW
from . import metrics as MX
from . import statistics as ST

TASKS = {
    "B1_support_recovery": "estimate perception-support fraction",
    "B2_private_rank_estimation": "estimate private-correction rank",
    "B3_subject_transfer": "does target geometry transfer across subjects?",
    "B4_few_shot_calibration": "recover geometry from few target observations",
    "B5_adaptive_stopping": "stop when subspace stabilizes",
    "B6_zero_shot_subject_prediction": "predict target geometry from descriptors, participant-disjoint",
    "B7_cross_session_transfer": "predict session-2 geometry from session-1",
    "B8_cross_state_transfer": "shared scaffold recurs across imagery vs recall",
}

BASELINES = ["population_mean", "identity", "ridge", "pca", "procrustes", "nearest_neighbor", "random_subspace", "oracle"]


def _subspace(samples, k):
    Vt = np.linalg.svd(np.asarray(samples) - np.asarray(samples).mean(0, keepdims=True), full_matrices=False)[2]
    return Vt[:k]


def b2_private_rank(world):
    """Estimate private rank via participation ratio of residual after removing shared scaffold; error vs truth."""
    shared = world["shared_basis_true"]; truth = world["private_rank"]; errs = []
    for s in world["subjects"]:
        X = s["imagery"]; Xc = X - X.mean(0, keepdims=True)
        proj = Xc @ shared.T @ shared
        resid = Xc - proj
        sv = np.linalg.svd(resid, compute_uv=False); lam = sv ** 2
        pr = (lam.sum() ** 2) / np.sum(lam ** 2) if lam.sum() > 0 else 0.0
        errs.append(abs(round(pr) - truth))
    return {"task": "B2_private_rank_estimation", "truth": truth, "median_abs_rank_error": float(np.median(errs)),
            "baseline": "participation_ratio_of_residual"}


def b3_subject_transfer(world):
    """within-subject vs between-subject imagery-subspace overlap (transfer should be HIGHER within if subject-specific)."""
    subs = [_subspace(s["imagery"], world["private_rank"]) for s in world["subjects"]]
    # split each subject's samples in half -> within-subject overlap
    within = []
    for s in world["subjects"]:
        X = s["imagery"]; h = len(X) // 2
        within.append(MX.subspace_overlap(_subspace(X[:h], world["private_rank"]), _subspace(X[h:], world["private_rank"])))
    between = [MX.subspace_overlap(subs[i], subs[j]) for i in range(len(subs)) for j in range(i + 1, len(subs))]
    return {"task": "B3_subject_transfer", "within_median": float(np.median(within)),
            "between_median": float(np.median(between)),
            "within_exceeds_between": float(np.median(within)) > float(np.median(between))}


def b4_few_shot(world, n_obs=8):
    """few-shot subspace vs full-data subspace overlap (recovery)."""
    ov = []
    for s in world["subjects"]:
        X = s["imagery"]
        full = _subspace(X, world["private_rank"])
        few = _subspace(X[:n_obs], world["private_rank"])
        ov.append(MX.subspace_overlap(few, full))
    return {"task": "B4_few_shot_calibration", "n_obs": n_obs, "median_recovery_overlap": float(np.median(ov))}


def b8_cross_state(world):
    real = []; null = []
    for s in world["subjects"]:
        a = _subspace(s["imagery"], world["shared_rank"]); b = _subspace(s["recall"], world["shared_rank"])
        real.append(MX.subspace_overlap(a, b))
    # null: cross-subject state pairing
    subs = world["subjects"]
    for i in range(len(subs)):
        j = (i + 1) % len(subs)
        null.append(MX.subspace_overlap(_subspace(subs[i]["imagery"], world["shared_rank"]),
                                        _subspace(subs[j]["recall"], world["shared_rank"])))
    return {"task": "B8_cross_state_transfer", "real_median": float(np.median(real)),
            "null_median": float(np.median(null)), "real_exceeds_null": float(np.median(real)) > float(np.median(null))}


def run_benchmark(cfg: SW.WorldConfig = None):
    cfg = cfg or SW.WorldConfig()
    world = SW.generate(cfg)
    results = {"config": cfg.to_dict(), "tasks": {}}
    results["tasks"]["B2"] = b2_private_rank(world)
    results["tasks"]["B3"] = b3_subject_transfer(world)
    results["tasks"]["B4"] = b4_few_shot(world)
    results["tasks"]["B8"] = b8_cross_state(world)
    results["ground_truth_scored"] = True
    results["leaderboard_scope"] = "LOCAL_SYNTHETIC_ONLY (no external SOTA claim)"
    return results
