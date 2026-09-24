"""M3 -- Cross-State Shared-Scaffold / Private-Correction Generalization.

Question: does the same shared-scaffold + private-correction structure appear in ANOTHER internally generated
state (delayed visual recall / memory reinstatement), not just imagery?

M3 is primarily a PROSPECTIVE PROTOCOL: a separate delayed-recall / memory-reinstatement task run as a
POST-REPLICATION block/session. It does NOT modify the O2.16 acquisition in any way and remains AWAITING
PI/ethics/site review. The cross-state geometry statistic decomposes each state's target geometry into a
shared-scaffold component (common across states) and a private-correction residual (state/participant-specific);
generalization means the shared scaffold recurs across states while private corrections stay subject-specific.
Here only a SYNTHETIC engineering check runs; nothing is discovered/supported/validated."""
from __future__ import annotations

import numpy as np

from . import common as CM

SCAFFOLD_DIM = 2
N_PERM_NULL = 1000


def cross_state_decompose(state_a, state_b):
    """state_a/state_b: (obs x feat) target-geometry samples for two internally generated states of one
    participant. Returns shared-scaffold overlap and private-correction fraction."""
    def sub(X):
        X = np.asarray(X, np.float64)
        Vt = np.linalg.svd(X - X.mean(0, keepdims=True), full_matrices=False)[2]
        return Vt[:SCAFFOLD_DIM]
    Sa, Sb = sub(state_a), sub(state_b)
    Qa = np.linalg.qr(Sa.T)[0]; Qb = np.linalg.qr(Sb.T)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    shared = float(np.mean(np.clip(s, 0, 1) ** 2))          # scaffold recurrence in [0,1]
    private = float(1.0 - shared)                            # private-correction fraction
    return {"shared_scaffold_overlap": shared, "private_correction_fraction": private}


def _synth_participant(name, shared_strength):
    r = CM.rng("M3|synth|%s|%.2f" % (name, shared_strength))
    feat = 24
    scaffold = np.linalg.qr(r.standard_normal((feat, SCAFFOLD_DIM)))[0]
    def state(tag):
        priv = np.linalg.qr(r.standard_normal((feat, SCAFFOLD_DIM)))[0]
        S = []
        for _ in range(8):
            c = r.standard_normal(SCAFFOLD_DIM)
            shared_part = shared_strength * (scaffold @ c)
            priv_part = (1 - shared_strength) * (priv @ r.standard_normal(SCAFFOLD_DIM))
            S.append(shared_part + priv_part + 0.1 * r.standard_normal(feat))
        return np.array(S)
    return state("imagery"), state("recall")


def simulate(n=CM.N_COHORT_B_PLANNED):
    """Engineering check: planted shared scaffold -> high shared overlap vs a no-shared null. NOT scientific."""
    rows = []
    for i in range(1, n + 1):
        name = "synB%03d" % i
        a, b = _synth_participant(name, shared_strength=0.7)
        an, bn = _synth_participant(name + "n", shared_strength=0.0)
        real = cross_state_decompose(a, b)["shared_scaffold_overlap"]
        null = cross_state_decompose(an, bn)["shared_scaffold_overlap"]
        rows.append({"participant": name, "shared_real": real, "shared_null": null,
                     "real_exceeds_null": real > null})
    n_exceed = sum(r["real_exceeds_null"] for r in rows)
    return {"moonshot": "M3", "engineering_simulation_only": True, "n": n,
            "n_real_exceeds_null": n_exceed, "frac": n_exceed / n,
            "protocol_status": "PROSPECTIVE_PROTOCOL_ONLY; separate post-replication block; O2.16 unmodified",
            "rows": rows, "caption": "synthetic engineering check; NOT discovered/supported/validated"}


def protocol_spec():
    return {
        "task": "delayed visual recall / memory reinstatement (cued recall of previously seen anchors after delay)",
        "relationship_to_o2_16": "SEPARATE post-replication block/session; DOES NOT modify O2.16 acquisition, "
                                  "endpoints, estimator, folds, or any frozen parameter",
        "cross_state_pairs": "imagery-state geometry (from O2.16) vs recall-state geometry (new block)",
        "primary_question": "does the shared-scaffold + private-correction decomposition recur across states?",
        "requires": ["PI review", "ethics amendment/approval for the added block", "site/session scheduling",
                     "stimulus permissions for recall cues", "participant burden review"],
        "status": "M3_CROSS_STATE_PROTOCOL_READY_FOR_HUMAN_REVIEW",
        "not_a_scientific_result": True,
    }


def support_rule_text():
    return ("M3 (only if the recall block is approved and run; post-unlock): median participant cross-state "
            "shared-scaffold overlap exceeds a no-shared permutation null with >=75% participants above null, AND "
            "private-correction remains subject-specific (does not transfer across participants). Any pass is a "
            "Cohort-B candidate requiring Cohort-C confirmation.")
