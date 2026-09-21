# Reviewer FAQ

**Why only N = 8?** This is the publicly available imagery cohort (NSD-Imagery). We treat it as a
discovery/development cohort, use the participant as the inferential unit, and report the exact small-sample
inference (minimum one-sided p = 1/256 ≈ 0.004). Generality is deferred to an independent replication, which is
preregistered.

**Why is eight not a universal minimum?** Eight observations is the smallest burden we *prospectively tested*
that worked on average, under one *frozen estimator*, in *these two regions*, on *this cohort*. Finer grids and
other estimators were not the test, so it is not a demonstrated lower bound.

**Why isn't this just noise?** Group effects use exact participant-level sign-flip tests with multiplicity
control; the key predictor (repeat geometry → calibration success) is positive in all eight participants in both
regions. Controls rule out that it is a scalar reliability variable, that the anisotropy is a geometry artefact,
and that "no transfer" is a voxel-space comparability artefact.

**How is this different from Roy et al.?** Saha Roy et al. (2025) established that a vision→imagery
transformation predicts imagery activity. We independently reproduce/extend that framing and add: identity-held-
out mapping, cross-participant transfer, a perception-supported/target-specific decomposition, and a
minimal-calibration frontier with acquisition robustness and a geometric predictor. We make no bitwise-
replication claim.

**How is this different from MIRAGE?** MIRAGE (Kneeland et al., 2026) reconstructs images from brain activity
across vision and imagery. We study a brain-state→brain-state transformation and its calibration geometry — a
different target space and question.

**Why does cross-subject transfer matter?** If the imagery-specific geometry transferred across people, shared-
subject or hyperalignment approaches could supply it cheaply. It does not (under our test), which is why
per-person calibration is needed — and why its data cost is the interesting quantity.

**Why is independent replication essential?** All findings come from one small cohort analysed repeatedly.
Program-wide multiplicity is uncontrolled, so the surviving findings are preregistered as prospective hypotheses
for a fresh cohort rather than claimed as established.

**Why not analyze the eight people further?** The discovery program is deliberately closed: further analysis of
the same cohort would compound multiplicity and cannot establish generality. New evidence requires new people.

**Why 7T?** The discovery data are 7T high-resolution; matching field strength and ~1.8 mm resolution keeps the
replication comparable. 7T is a target where feasible, not a requirement secured.

**What happens if the replication fails?** The development-cohort calibration result does not generalise under
the frozen protocol — a clean, preregistered negative. The discovery findings remain honestly reported as
cohort-specific.

**What happens if it succeeds?** It supports external generalisation of the minimal-calibration result and its
operational quality monitor; the three secondary geometric endpoints are evaluated separately and cannot change
the primary conclusion.
