# Subject-specific geometry and calibration limits in human visual imagery

**Authors:** [Author name(s) and affiliation to be completed] · MSc Cognitive Science project

> **Cohort status.** All results are from a **discovery / development cohort of N = 8** participants (Natural
> Scenes Dataset). The independent replication described at the end is **prospectively frozen but not yet
> executed** — nothing here is an independent replication.

---

## Abstract

Mental imagery and perception engage overlapping visual cortex, and recent work frames imagery as a
transformation of perceptual activity. Rather than asking only whether perception predicts imagery, we ask
which components of imagery geometry perception fails to specify, how those components vary across participants,
and how much direct imagery data are required to recover them. In a discovery cohort of eight participants,
using a prospectively frozen analysis pipeline with the participant as the sole inferential unit, we find that
perception carries significant but incomplete information about the imagined ("target") neural state: a
low-dimensional component of target-state geometry lies outside the perceptual representation. Under the tested
representation and transfer procedure this missing component is **subject-specific** — it does not transfer
across participants and is not predictable from the tested perception phenotype, anatomy, or behaviour. A small
amount of direct imagery calibration recovers much of it: eight observations (four imagined identities each
measured twice) was the smallest prospectively tested common calibration burden, under the frozen estimator,
that recovered at least half of the native-oracle gap in both visual streams (ventral median total recovery
0.569, lateral 0.530) — not a universal minimum. This frontier was **not schedule-robust**: which measurements
are acquired matters, and repeat-to-repeat agreement of the outside-of-perception geometry predicted whether a
calibration generalised (median Spearman ρ ≈ 0.42 ventral, 0.48 lateral; all eight participants positive in
both streams). Exploratory follow-up on the same cohort indicates the repeat variation is structured but
distributed — not a discrete latent state, a scalar reliability variable, or a single dominant geometric drift
mode. An independent replication with three preregistered secondary endpoints is frozen but not yet executed;
these findings are discovery-cohort evidence and define prospective hypotheses for that replication.

---

## Introduction

Imagery and perception recruit overlapping visual cortex, but similarity of activation does not imply identical
representational geometry. Recent work frames imagery as a transformation of perceptual activity that predicts
imagery responses while occupying a distinct, lower-dimensional subspace (Saha Roy et al., 2025); more generally,
neural transformation functions can generalise to held-out stimuli (Ward, Isik & Chun, 2018). Whether such a
perception→imagery transformation is shared across people, and how much of the imagined state it leaves
unspecified, is not established. Methods for identifying shared representational structure across individuals
exist (hyperalignment: Haxby et al., 2011, 2020; the shared response model: Chen et al., 2015), and
vision-trained decoders can be adapted to reconstruct mental images (MIRAGE: Kneeland et al., 2026) — but the
latter concerns brain→image reconstruction rather than the brain-state→brain-state transformation studied here.

This project began as an independent method reproduction and extension of Saha Roy et al.; the original code and
random seeds are unavailable, so we make no claim of a bitwise replication. Building on transformation-based
analyses of perception and imagery, we jointly characterise identity-held-out perception-to-imagery mapping,
test its transfer across participants, decompose target imagery geometry into perception-supported and
target-specific components, and prospectively quantify the amount and acquisition robustness of the direct
participant-specific imagery calibration required to recover the missing geometry. We make no priority ("first")
claim; the contribution is this combination on a development cohort with a preregistered independent replication.

---

## Results

Throughout, the inferential unit is the participant (N = 8); measurement schedules, cross-validation folds,
identity subsets, and repeats are never treated as independent subjects. Group tests are exact participant-level
sign-flip tests (minimum attainable one-sided p = 1/256 ≈ 0.004) with multiplicity control within each
pre-declared analysis. Two visual-stream regions are analysed: the ventral and lateral streams.

### 1. Perception is informative but does not fully determine target-state geometry
Decomposing the imagined neural state relative to a perception-support subspace, a substantial fraction is
captured by the perception-derived mapping, but a real component of target-native geometry lies **outside**
perception support: the perception-support ceiling sits well below an oracle estimated from the participant's
own imagery data. Imagery is therefore not fully reconstructable from the perceptual representation in these
streams.

### 2. The missing geometry is low-complexity but subject-specific (under the tested representation)
The outside-of-perception residual is low-dimensional. Its orientation, however, is idiosyncratic to the
individual: transferring one participant's target-state geometry to others performed near chance, and the
orientation was not predictable from the tested perception phenotype, nor from the tested anatomical or
behavioural covariates. This is a failure of cross-subject transfer and of the tested predictors — not a
demonstration that the geometry is biologically unique to each person.

### 3. A small amount of direct imagery calibration recovers much of the missing geometry
Sweeping how many imagined identities (M) and how many repeats each (T) are used to calibrate, under a fixed
estimator, the smallest prospectively tested common burden that recovered at least half of the native-oracle gap
in **both** streams was **eight observations** — four imagined identities each measured twice:

| Stream | Median total recovery | Participants ≥ 0.5 | Median full-capacity fraction | Participants ≥ 0.5 |
|---|---|---|---|---|
| Ventral | **0.569** | 7 / 8 | 0.596 | 8 / 8 |
| Lateral | **0.530** | 6 / 8 | 0.591 | 8 / 8 |

Adding a perception-only prior improved recovery but did **not** reduce this eight-observation burden. This
figure is specific to the tested estimator, streams, and cohort — **not a universal minimum**.

### 4. The eight-observation frontier is fragile, and repeat geometry predicts success
Holding the eight-observation burden fixed, which particular measurements were acquired strongly affected
success: the fraction of eight-observation schedules meeting the joint sufficiency criterion (schedule coverage)
was only 0.656 (ventral) and 0.579 (lateral), and no schedule-robust minimal burden existed. What predicted
whether a calibration generalised was the **repeat-to-repeat agreement of the outside-of-perception geometry**:
its schedule-level correlation with the calibration robustness margin had median ρ 0.423 (ventral) and 0.477
(lateral), positive in all eight participants in both streams. The within-perception-support agreement was a
weaker predictor (median ρ 0.131 / 0.119). In short, acquisition success is governed by outside-support repeat
geometry, not merely by trial count.

### 5. Exploratory follow-up: the failure-predictive geometry is structured but distributed
*(Exploratory analyses on the same N = 8 cohort; they refine, not establish, the findings above.)* A single
coordinate-invariant perception↔imagery angular relationship survived a pipeline-triviality control in both
streams. However, there was no reproducible discrete two-state structure (a two-state model never out-predicted
a one-state model; held-out two-minus-one-state log-likelihood −0.113 ventral, −0.100 lateral, 0/8 positive),
and a scalar reliability variable did not predict the calibration margin (ρ 0.004 / 0.064). Repeat-level
perturbations of the outside-of-perception subspace were reproducibly anisotropic — a dominant, cross-half
reproducible direction beyond an isotropic norm/rank-matched baseline (centred anisotropy +0.093 ventral, +0.101
lateral; 8/8 participants) — but that dominant direction did **not** predict calibration failure (correlation of
the wrong sign, −0.128 / −0.099). Taken together, the failure-predictive repeat geometry is genuinely structured
but appears distributed, rather than reducible to a discrete state, a scalar quality variable, or a single
dominant mode.

---

## Methods

**Participants and data.** Eight participants from the Natural Scenes Dataset (NSD; Allen et al., 2022),
including the NSD-Imagery task, treated as a discovery/development cohort. The participant is the sole
inferential unit.

**Neural responses and voxel selection.** Single-trial responses used the NSD-Imagery B0 beta preparation,
corresponding to the public NSD `betas_fithrf` (b2-compatible) preparation; the public description is "similar
to b2," so we do not claim the released file is bitwise identical to the preparation in the original analysis. A
GLMdenoise/RR (b3-compatible) preparation was examined only as a measurement-preparation sensitivity branch.
Analyses used the native 1.8-mm functional grid (`func1pt8mm`) with no spatial resampling. Within each region,
eligible voxels were the intersection of the region mask, valid NSD-Imagery voxels, and voxels with finite
NSD-core noise-ceiling SNR (ncsnr); we retained voxels with ncsnr strictly greater than the region-specific 98th
percentile of that eligible distribution (computed separately per region, strict comparator, no threshold
adaptation after outcomes, and the same voxels used for the preparation-sensitivity comparison). This is a
per-region strict-percentile rule on an explicitly defined eligible set, not a generic "top-2%" selection.

**Regions.** Early visual regions (V1–hV4) came from the `prf-visualrois` atlas; the ventral, lateral, and
parietal regions came from the `streams` atlas. The two primary regions are the ventral and lateral streams;
parietal was reserved as a secondary region.

**Representation and calibration.** Per participant and region we froze a perception-support subspace and a
fixed two-dimensional outside-support component, and represented the perception→imagery transformation on this
frozen geometry (no Procrustes alignment). Calibration recovery was measured against a within-participant oracle
and a full-resource ceiling. All ranks and dimensions were frozen before any outcome was examined.

**Inference and reproducibility.** Group inference used exact participant-level sign-flip tests with
multiplicity control within each pre-declared analysis family; the minimum attainable one-sided p at N = 8 is
1/256 ≈ 0.004. Sufficiency and schedule-robustness criteria were pre-registered decision rules, not hypothesis
tests. Every analysis fixed its configuration before outcomes; reused estimators reproduce upstream quantities
to numerical tolerance. Full provenance (per-analysis configuration hashes and source bindings) is in the
technical supplement.

---

## Discussion

Four points follow. First, imagery is not a simple replay of perception geometry: a real component of the
imagined state lies outside the perceptual representation. Second, that component is participant-specific under
the tested representation and transfer procedure, and is not recovered by the tested perception, anatomy, or
behaviour predictors. Third, it is nonetheless low-complexity enough to be calibrated from a small amount of
direct imagery data — with eight observations the smallest prospectively tested common burden under the frozen
estimator, not a universal minimum. Fourth, calibration success depends on the geometry of the acquired repeats,
not only their number.

**Relation to prior work.** We do not claim these ideas as new. A vision→imagery transformation that predicts
imagery activity is established by Saha Roy et al. (2025), which this project extends. Stimulus-general neural
transformations that generalise to novel objects were shown by Ward, Isik & Chun (2018). Cross-subject shared
spaces are established by hyperalignment (Haxby et al., 2011, 2020) and the shared response model (Chen et al.,
2015), used here only as conceptual tools. Vision-trained decoding of imagery is established and examined by
MIRAGE (Kneeland et al., 2026), which targets brain→image reconstruction rather than the brain-state→brain-state
transformation studied here. The contribution is the combination: identity-held-out mapping, cross-participant
transfer, the perception-supported/target-specific decomposition, and the minimal-calibration frontier with its
acquisition robustness and geometric predictor.

**Alternative interpretations.** Two limitations bound every conclusion: the results are conditional on the
frozen estimator and representation, and the program tests the same eight participants repeatedly, so
program-wide multiplicity is uncontrolled. Apparent subject-specificity at N = 8 could partly reflect sampling,
and effects may be specific to this imagery task. We can exclude, by controls, that the failure-predictive
geometry is merely a scalar reliability variable, that the anisotropy is a projector-geometry artefact, and that
"no cross-subject transfer" is a comparability artefact of incomparable voxel spaces. Independent replication is
the decisive next step.

---

## Limitations

Small sample (N = 8); a discovery/development cohort analysed repeatedly; results conditional on the tested
estimator and representation; two primary regions only; an acquisition-schedule-fragile calibration frontier;
bounded trial-level data for the exploratory analyses; task specificity; fMRI spatial/temporal limits; no causal
or mechanistic claim (the geometric quantities are model quantities, not identified neural processes); no bitwise
reproduction of the original analysis (original code/seeds unavailable); and, critically, no independent
replication yet.

---

## Data and code availability

The Natural Scenes Dataset is publicly available; no raw response files are redistributed here. Analysis
configurations, provenance, and result tables are maintained under version control with per-analysis
configuration hashes; a technical supplement records the full analysis ledger and source bindings.

---

## References

See `REFERENCES.md` (primary-source verified). Key works: Saha Roy et al. (2025); Ward, Isik & Chun (2018);
Haxby et al. (2011, 2020); Chen et al. (2015); Kneeland et al. (2026, MIRAGE); Allen et al. (2022, NSD).
