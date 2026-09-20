# Abstract (structured, ~300 words)

**Background.** Mental imagery and perception engage overlapping visual cortex, and recent work frames imagery
as a transformation of perceptual activity (Saha Roy et al., 2025). Rather than asking only whether perception
predicts imagery, we ask which components of imagery geometry perception fails to specify, how those components
vary across participants, and how much direct imagery data are required to recover them. We studied this in a
**discovery/development cohort** of **N = 8** participants from the Natural Scenes Dataset, treating the
participant as the sole inferential unit and using prospective, pre-outcome freezes throughout.

**Methods.** Within ventral and lateral visual streams we constructed a perception-support subspace and a
perception→imagery operator, decomposed the target-imagery state into within-support and outside-support
components, and quantified how much target geometry lies outside perception support. We tested whether the
missing geometry transfers across participants or is predictable from perception, anatomy, or behaviour; how
little direct imagery calibration recovers it (an M identities × T repeats grid under a fixed direct-SVD +
two-dimensional outside estimator); whether the resulting frontier is robust to acquisition-schedule
composition; and whether repeat-to-repeat geometric agreement predicts calibration generalization. Four
exploratory mechanistic analyses (invariants, latent states, angular reliability, geometric drift) probed the
structure of the repeat variation. Inference used participant-level exact sign-flip tests (minimum attainable
one-sided p = 1/256 ≈ 0.0039) with Holm correction.

**Results.** Perception carried significant but incomplete information about target-state geometry; the missing
component was low-dimensional (under the frozen representation) yet **subject-specific under the tested
representation and transfer procedure** — it did not transfer across participants and was not predictable from
the tested perception phenotype, anatomy, or behaviour. The **smallest prospectively tested common calibration
burden under the frozen estimator** that closed at least half of the native-oracle gap in both streams was **8
observations (4 identities × 2 repeats)** (ventral median total recovery 0.569, lateral 0.530) — not a universal
minimum, and this frontier was **not schedule-robust**.
Repeat-to-repeat outside-support geometric agreement strongly predicted calibration robustness (median
Spearman ρ 0.42 ventral, 0.48 lateral; 8/8 participants both streams). Mechanistically, the repeat variation
was structured but **distributed**: it was not a discrete two-state process, not a scalar reliability variable,
not a repeat-stable angular scaffold, and not reducible to a single dominant geometric drift mode (which,
although reproducibly anisotropic, did not predict calibration failure).

**Interpretation.** In this discovery cohort, imagery is not a simple replay of perception geometry: a
low-complexity but participant-specific target-state component governs minimal calibration, whose success
depends on distributed repeat geometry. A genuinely independent replication is preregistered and frozen but
**not yet executed**; these findings are discovery-cohort evidence pending that replication.
