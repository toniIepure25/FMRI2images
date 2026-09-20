# Revision log — P2 adversarial hardening

Every change is wording / structure / caveat / framing only. **No scientific result, number, classification,
statistical test, or gate status was altered.** Source of each change in brackets.

## Manuscript files revised (`paper/mindir/`)
- **ABSTRACT.md** — "subject-specific" → "subject-specific under the tested representation and transfer
  procedure"; "8 observations" bounded as "smallest prospectively tested common calibration burden under the
  frozen estimator … not a universal minimum." [Reviewer A/B; Phases 7,8,12]
- **RESULTS.md** — added **Question / Status (prospective vs exploratory / decision-rule vs test)** headers to
  Results 1–4; Result 2 subject-specific qualifier + "not biological uniqueness"; Result 3 "8 observations"
  bounded to frozen estimator + development cohort; Result 4 inferential-unit statement (schedules are
  within-participant repeated measures). [Reviewers A/B; Phases 7,8,9,14]
- **METHODS.md** — new "Statistical scope and multiplicity" paragraph (participant N; within-gate families; no
  cross-gate correction; sufficiency rules ≠ tests); §4 renamed "Neural response estimation and voxel selection"
  with beta-version/voxel-rule disclosure placeholder; §7 explicit cross-subject invariant-quantity statement.
  [Reviewers B/C; Phases 6,15]
- **DISCUSSION.md** — added "Alternative interpretations we cannot exclude" and "What independent O2.16 will
  test prospectively." [Phases 3,15,17]
- **LIMITATIONS.md** — added items 12 (schedule fragility), 13 (task specificity), 14 (no physical cause for
  repeat geometry). [Phase 16]
- **claims_matrix.csv** — downgraded SUPPORTED_WORDING for C2 (low-dimensional → under frozen D=2
  representation), C3 (subject-specific → under tested representation/transfer; strengthened FORBIDDEN_OVERCLAIM
  to include "biologically unique"), C6 (8-observation → under frozen estimator on development cohort; not a
  demonstrated lower bound). [Phase 2 audit]
- **FIGURE_PLAN.md** — added participant-level rendering mandate + "how to read N=8" legend note for F4/F5 +
  Exploratory heading for F6. [Phase 11]

## Not changed (immutable)
O2.9/O2.10/O2.11/O2.12/O2.13/O2.14/O2.15 statuses and numbers; X1/X2/X3/X4 statuses and numbers; O2.16 /
O2.16-DATA / O2.16-SEC statuses; O3_NOT_READY; discovery-stop. `evidence_table.csv` (values) unchanged.
`MANUSCRIPT.md`, `METHODS.md` numbering, `replication_box.md`, `reviewer_risk_register.md`, `TITLE_OPTIONS.md`
substance unchanged (title recommendation unchanged: Title 1).

## Human-only items surfaced (cannot be resolved in this editing gate)
1. Fill exact NSD beta version + voxel-selection rule in Methods §4 (from the sealed upstream config).
2. Verify literature citations in `literature_positioning_matrix.csv` against primary sources; confirm novelty
   sentence.
