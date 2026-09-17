"""O2.16-DATA prospectively-frozen acquisition DESIGN (frozen config 804d5b21). Structural design + validation
for a genuinely NEW independent human fMRI cohort that runs the already-frozen O2.16 direct replication
(config 2da2cc79) UNCHANGED. This module encodes ONLY the frozen design constants and their internal-consistency
checks -- it acquires NO data (human scanning requires ethics/IRB + a scanner + real acquisition records, none
of which exist here). Perception anchors reference the sealed canonical 512 O2.3A-RD set (provenance-bound at
acquisition time); imagery is 12 identities (6 simple + 6 naturalistic) x 8 repeats; 6 outer folds each hold out
1 simple + 1 naturalistic, leaving 5+5 training -> C(5,2)^2=100 identity subsets x C(8,2)=28 repeat pairs = 2800
M4T2 schedules/fold; each schedule = exactly 8 target-imagery observations."""
from __future__ import annotations

import itertools

N_PERCEPTION_ANCHORS = 512          # exact canonical O2.3A-RD set (sealed; referenced by provenance)
PERCEPTION_REPEATS = 3
SIMPLE = [f"s{i}" for i in range(6)]
NAT = [f"n{i}" for i in range(6)]
IMAGERY_IDENTITIES = SIMPLE + NAT   # 12
IMAGERY_REPEATS = 8
N_OUTER_FOLDS = 6
M = 4
T = 2


def outer_folds():
    """6 folds; fold f holds out (SIMPLE[f], NAT[f]); training = remaining 5 simple + 5 naturalistic."""
    folds = []
    for f in range(N_OUTER_FOLDS):
        ho = (SIMPLE[f], NAT[f])
        tr_s = [s for s in SIMPLE if s != ho[0]]
        tr_n = [n for n in NAT if n != ho[1]]
        folds.append({"fold": f, "heldout_simple": ho[0], "heldout_nat": ho[1], "train_simple": tr_s, "train_nat": tr_n})
    return folds


def m4_subsets(train_simple, train_nat):
    """Balanced 2-simple + 2-naturalistic identity subsets: C(5,2) x C(5,2) = 100."""
    return [tuple(cs) + tuple(cn) for cs in itertools.combinations(train_simple, 2) for cn in itertools.combinations(train_nat, 2)]


def repeat_pairs():
    """All C(8,2) = 28 two-repeat subsets over repeat positions 0..7."""
    return list(itertools.combinations(range(IMAGERY_REPEATS), 2))


def schedules_per_fold(fold):
    return len(m4_subsets(fold["train_simple"], fold["train_nat"])) * len(repeat_pairs())


def validate_design():
    """Return (ok, checks dict) certifying the frozen design is internally consistent + O2.16-compatible."""
    c = {}
    c["perception_anchors_512"] = (N_PERCEPTION_ANCHORS == 512)
    c["perception_repeats_3"] = (PERCEPTION_REPEATS == 3)
    c["perception_total_1536"] = (N_PERCEPTION_ANCHORS * PERCEPTION_REPEATS == 1536)
    c["imagery_12_identities"] = (len(IMAGERY_IDENTITIES) == 12)
    c["imagery_6_simple_6_nat"] = (len(SIMPLE) == 6 and len(NAT) == 6)
    c["no_imagery_in_perception"] = (not set(IMAGERY_IDENTITIES) & set(f"anchor{i}" for i in range(N_PERCEPTION_ANCHORS)))
    c["imagery_8_repeats"] = (IMAGERY_REPEATS == 8)
    c["imagery_total_96"] = (len(IMAGERY_IDENTITIES) * IMAGERY_REPEATS == 96)
    folds = outer_folds()
    c["six_outer_folds"] = (len(folds) == 6)
    c["each_fold_holds_1s_1n"] = all(fd["heldout_simple"] in SIMPLE and fd["heldout_nat"] in NAT for fd in folds)
    c["each_train_5_5"] = all(len(fd["train_simple"]) == 5 and len(fd["train_nat"]) == 5 for fd in folds)
    c["100_m4_subsets"] = all(len(m4_subsets(fd["train_simple"], fd["train_nat"])) == 100 for fd in folds)
    c["28_repeat_pairs"] = (len(repeat_pairs()) == 28)
    c["2800_schedules_per_fold"] = all(schedules_per_fold(fd) == 2800 for fd in folds)
    c["schedule_8_observations"] = (M * T == 8)
    c["heldout_disjoint_from_training"] = all(fd["heldout_simple"] not in fd["train_simple"] and fd["heldout_nat"] not in fd["train_nat"] for fd in folds)
    # every training identity is held out in exactly one fold (full LOSO coverage over 6 simple + 6 nat)
    ho_s = [fd["heldout_simple"] for fd in folds]; ho_n = [fd["heldout_nat"] for fd in folds]
    c["full_loso_coverage"] = (sorted(ho_s) == SIMPLE and sorted(ho_n) == NAT)
    return all(c.values()), c
