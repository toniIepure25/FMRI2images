# Phase 18 — Mock editorial decision (internal readiness only)

Decisions are an **internal readiness assessment**, not a prediction of any journal's action.

## Initial verdicts (on the P1 manuscript, before P2 hardening)
| Reviewer | Verdict | Principal blockers |
|---|---|---|
| A — systems/comp neuro | **MAJOR_REVISION** | conceptual delta over Roy/hyperalignment not stated; constructs' neural meaning; subject-specificity reach |
| B — statistics/methods | **MINOR_REVISION** | no critical flaw; "minimum" language, cross-gate multiplicity, inferential-unit signposting |
| C — fMRI/reproducibility | **MAJOR_REVISION** | beta version + voxel-selection disclosure; cross-subject comparability statement; task provenance |

## Changes applied in P2 (wording/structure/caveats only; no result changed)
- Novelty sentence added (`novelty_statement.md`); representation-level framing reinforced.
- "Subject-specific under the tested representation and transfer procedure" throughout (Abstract, Results 2,
  claims matrix).
- "8 observations" bounded everywhere as the smallest prospectively tested common burden under the frozen
  estimator on the development cohort; not a universal minimum (Abstract, Result 3, claims matrix).
- Methods: statistical-scope-and-multiplicity paragraph; voxel-selection disclosure placeholder; cross-subject
  invariance statement.
- Discussion: alternative-interpretations and replication-value paragraphs.
- Limitations: task specificity, schedule fragility, no-physical-cause added (now 14 items).
- Figure plan: participant-level rendering mandate + N=8 legend note for F4/F5.

## Post-revision verdicts
| Reviewer | Verdict | Remaining blockers |
|---|---|---|
| A — systems/comp neuro | **MINOR_REVISION** | wants the novelty sentence in the Introduction/Discussion once the human literature check confirms it |
| B — statistics/methods | **ACCEPTABLE_FOR_SUBMISSION (minor)** | none critical; ensure every figure caption names the inferential unit |
| C — fMRI/reproducibility | **MINOR_REVISION** | human must fill the exact NSD beta version + voxel-selection rule from the upstream config (placeholder present) |

## Editor synthesis
**Overall: MINOR_REVISION for internal readiness — no CRITICAL flaw remains.** The science is unchanged and
well-controlled; the manuscript is honest about N = 8, discovery status, estimator dependence, and pending
replication. Two remaining items are **human-only** and cannot be resolved by this editing gate without new
work or external facts:
1. **Fill the exact beta version + voxel-selection rule** in Methods §4 from the sealed upstream config.
2. **Verify the literature citations** in `literature_positioning_matrix.csv` against primary sources and
   confirm the novelty sentence.
Neither is a scientific-result change. With these two human steps done, the package is submission-ready.
