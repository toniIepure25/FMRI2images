"""Protocol invariants + balancing proofs. Proves the generated schedule satisfies the frozen design BEFORE
acquisition. No science; pure combinatorial verification."""
from __future__ import annotations

from collections import Counter

from . import config as C
from . import randomization as R


def prove_perception(seq):
    """Every anchor exactly 3x; no missing anchor; no 4th presentation; total 1536."""
    counts = Counter(t["anchor_id"] for t in seq)
    pres = Counter((t["anchor_id"], t["presentation"]) for t in seq)
    ok = {
        "total_trials_correct": len(seq) == C.PERCEPTION_TRIALS,
        "n_distinct_anchors": len(counts) == C.PERCEPTION_ANCHORS,
        "each_anchor_exactly_3": all(v == C.PERCEPTION_PRESENTATIONS for v in counts.values()),
        "no_missing_anchor": set(counts) == set(range(C.PERCEPTION_ANCHORS)),
        "no_fourth_presentation": all(0 <= p < C.PERCEPTION_PRESENTATIONS for (_, p) in pres),
        "no_duplicate_anchor_presentation": all(v == 1 for v in pres.values()),
    }
    return all(ok.values()), ok


def prove_imagery(seq):
    """Every identity exactly 8x; simple/nat 6/6; total 96."""
    counts = Counter(t["identity_id"] for t in seq)
    fam = Counter(t["family"] for t in seq)
    reps = Counter((t["identity_id"], t["repeat_index"]) for t in seq)
    ok = {
        "total_trials_correct": len(seq) == C.IMAGERY_TRIALS,
        "n_distinct_identities": len(counts) == C.IMAGERY_IDENTITIES,
        "each_identity_exactly_8": all(v == C.IMAGERY_REPEATS for v in counts.values()),
        "no_missing_identity": set(counts) == set(range(C.IMAGERY_IDENTITIES)),
        "simple_count_6": fam.get("simple", 0) == C.IMAGERY_SIMPLE * C.IMAGERY_REPEATS,
        "nat_count_6": fam.get("naturalistic", 0) == C.IMAGERY_NATURALISTIC * C.IMAGERY_REPEATS,
        "no_duplicate_identity_repeat": all(v == 1 for v in reps.values()),
    }
    return all(ok.values()), ok


def run_balance(seq, key):
    """Occurrence of key across runs (evenness diagnostic)."""
    runs = sorted(set(t["run"] for t in seq))
    return {r: dict(Counter(t[key] for t in seq if t["run"] == r)) for r in runs}


def prove_all(participant_id, session_id, n_runs=6):
    pseq, prec = R.perception_sequence(participant_id, session_id, n_runs)
    iseq, irec = R.imagery_sequence(participant_id, session_id, n_runs)
    pok, pdet = prove_perception(pseq)
    iok, idet = prove_imagery(iseq)
    return {
        "participant": participant_id, "session": session_id, "n_runs": n_runs,
        "perception_pass": pok, "perception_detail": pdet,
        "imagery_pass": iok, "imagery_detail": idet,
        "perception_seed": prec.seed, "imagery_seed": irec.seed,
        "all_pass": bool(pok and iok),
    }
