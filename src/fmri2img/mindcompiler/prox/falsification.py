"""Canonical falsification battery F1-F10. Each is a label-destroying transform + a runner that checks a claimed
effect COLLAPSES toward the null under the transform (a claim that survives its own falsifier is not evidence).
Operates on synthetic fixtures / provided arrays only."""
from __future__ import annotations

import numpy as np

from . import statistics as ST
from . import metrics as MX

BATTERY = {
    "F1_IDENTITY_SHUFFLE": "shuffle identity labels within participant",
    "F2_PARTICIPANT_SHUFFLE": "permute participant labels (breaks subject-specific pairing)",
    "F3_STATE_LABEL_SHUFFLE": "shuffle state labels (imagery/recall)",
    "F4_TARGET_SUPPORT_RANDOMIZATION": "replace perception support with a random rank-matched subspace",
    "F5_RANDOM_ORTHOGONAL_ROTATION": "apply a shared random rotation (geometry-invariant metrics must NOT change)",
    "F6_RANK_MATCHED_RANDOM_SUBSPACE": "replace signal subspace with rank-matched random",
    "F7_NUISANCE_ONLY_PREDICTOR": "predict from nuisance-only descriptors",
    "F8_TEMPORAL_REPEAT_SHUFFLE": "shuffle repeat order",
    "F9_SCHEDULE_RANDOMIZATION": "randomize acquisition schedule",
    "F10_MEASUREMENT_NOISE_CONTROL": "inject matched measurement noise, no signal",
}


def shared_rotation_invariance(A, B, dim, tag="F5"):
    """F5: rotating both subspaces by the same orthogonal transform leaves overlap unchanged."""
    r = ST.rng("F5|%s|%d" % (tag, dim))
    Q = np.linalg.qr(r.standard_normal((dim, dim)))[0]
    before = MX.subspace_overlap(A, B)
    after = MX.subspace_overlap(A @ Q.T, B @ Q.T)
    return {"metric": "subspace_overlap", "before": before, "after": after,
            "invariant": abs(before - after) < 1e-9}


def participant_shuffle_collapses(pred_fn, descriptors, geometry, tag="F2", n_perm=300):
    """F2: a subject-specific prediction must collapse toward null when participant labels are permuted."""
    def median_align():
        N = descriptors.shape[0]
        al = []
        for i in range(N):
            tr = [j for j in range(N) if j != i]
            al.append(pred_fn(descriptors[tr], geometry[tr], descriptors[i], geometry[i]))
        return float(np.median(al))
    real = median_align()
    r = ST.rng("F2|%s" % tag); N = descriptors.shape[0]; null = []
    for _ in range(n_perm):
        perm = r.permutation(N)
        g = geometry[perm]
        al = []
        for i in range(N):
            tr = [j for j in range(N) if j != i]
            al.append(pred_fn(descriptors[tr], g[tr], descriptors[i], g[i]))
        null.append(float(np.median(al)))
    p = ST.permutation_p(real, null)
    return {"real": real, "null_median": float(np.median(null)), "perm_p": p, "collapses": p > 0.05}


def rank_matched_random_null(signal_subspace, dim, rank, tag="F6", n=200):
    """F6: overlap of the real signal subspace with rank-matched random subspaces (the null band)."""
    r = ST.rng("F6|%s|%d|%d" % (tag, dim, rank))
    ov = [MX.subspace_overlap(signal_subspace, np.linalg.qr(r.standard_normal((dim, rank)))[0].T) for _ in range(n)]
    return {"null_overlap_median": float(np.median(ov)), "null_overlap_p95": float(np.quantile(ov, 0.95))}


def battery_manifest():
    return {"battery": BATTERY, "rule": "every future claim must name which falsifiers could kill it; a claim "
            "that does not collapse under its label-destroying falsifier is not evidence",
            "participant_unit": True}
