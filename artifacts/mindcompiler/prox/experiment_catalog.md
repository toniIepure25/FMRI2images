# Prospective high-value experiment catalog (MINDIR-PROX / P4)

Twelve prospective experiments for Cohort B/C. **None runs on historical N=8.** Each names its falsifier(s),
required cohort, and minimum evidence. Ranked qualitatively (value / falsifiability / feasibility / N-realism /
novelty / new-data dependence / ambiguity risk) — see `innovation_value_matrix.csv`. No hype; no fake scores.

### E1 — Test-retest private-geometry fingerprint
*Is the subject-specific correction stable across days/sessions?* Same subject, same state, different session.
Test within-subject cross-session similarity > between-subject. Falsifier: **F2 participant shuffle** (within
must exceed between). Cohort B (2 sessions) → C. **High value** (a stable fingerprint reframes the private
component as trait, not noise).

### E2 — Active acquisition / information gain (shadow first)
Choose next identity/repeat by expected information gain vs fixed/random. Metrics: obs used, recovery, regret,
false-stop. **Shadow mode only** first (never changes acquisition). Falsifier: **F9 schedule randomization**.
Cohort B.

### E3 — Cross-session zero-shot
Train private correction on Session 1; predict Session 2 imagery geometry with no Session-2 calibration.
Distinguishes stable individual geometry from session nuisance. Falsifier: **F2**, cross-session null. Cohort B.

### E4 — Cross-state private geometry (M3 extension)
Compare perception→imagery vs perception→recall (later: other justified states). Do private corrections share
orientation/dimensionality/reliability/subject ordering? Falsifier: **F3 state-label shuffle**. Cohort B (recall
block, human-approved) → C.

### E5 — Behavioral relevance
Do neural geometry metrics predict imagery vividness / recall accuracy / confidence (minimal preregistered
behavioral variables)? Falsifier: **F2**, behavior-shuffle null. Cohort B. **High value** (cognitive grounding).

### E6 — Multi-site generalization (3T vs 7T / two facilities)
Which MINDIR quantities are representation-level vs scanner/site dependent? Falsifier: site-swap null. Cohort C
(multi-site); not executable until feasible.

### E7 — Private-geometry stability vs state noise (variance components)
Hierarchical subject/session/run/repeat/identity model; estimate variance components prospectively. Participant
remains the unit. Falsifier: **F8/F10**. Cohort B (multi-session).

### E8 — Calibration / learning curve
Recovery vs 1..N observations with predeclared schedules; fit saturating models; is there a subject-specific
sample-complexity parameter (a phenotype candidate)? Falsifier: **F9**. Cohort B.

### E9 — Predictive geometry quality index (predictor-side)
A trust score for a participant's current calibration WITHOUT held-out outcomes (stability / condition number /
subspace convergence / identity diversity / uncertainty). M1 is the preregistered starting point. Falsifier:
must predict held-out failure better than chance without peeking. Cohort B.

### E10 — Cross-representation robustness
Do conclusions survive multiple pre-frozen representation families (voxel-native / PCA / SRM-like / anatomical)?
Freeze representations before outcomes. Falsifier: representation-swap. Cohort B. **High value** (separates
biology from representation artifact).

### E11 — Measurement-error-aware model
Separate latent geometry from measurement uncertainty; compare naive vs uncertainty-aware geometry (no large
model search). Falsifier: **F10**. Cohort B.

### E12 — Neural-geometry fingerprint identification (privacy-aware, scientific only)
Can subject identity be discriminated from private geometry across sessions above chance? Demonstrates persistent
subject structure. **Not an identification product; not for biometric use.** Falsifier: cross-session
participant shuffle. Cohort B/C.

## Discovery-stop discipline
Every positive on Cohort B is a candidate requiring an independent **Cohort C** confirmation (auto-templated). No
experiment reopens historical N=8.
