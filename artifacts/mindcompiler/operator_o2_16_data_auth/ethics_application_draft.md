# Ethics / IRB application — DRAFT (not submitted; not approved)

> **Status:** DRAFT support material only. This document has **not** been submitted to any ethics committee and carries **no** approval. All `[PLACEHOLDER — …]` fields require completion by the responsible human investigator/institution. Do not treat any field as institutionally endorsed.

**A. Study title.** Independent replication of a minimal target-imagery calibration and a training-only calibration-quality monitor for fMRI mental-imagery decoding.

**B. Scientific rationale.** Prior work (development cohort, NSD 7T, N=8) established a minimal 8-observation (4 identities × 2 repeats, "M4T2") target-imagery calibration and a repeat-to-repeat geometry-stability signal that predicts held-out calibration success. These findings have not been independently replicated; the frozen protocol tests them prospectively in a new cohort.

**C. Primary hypotheses.** (1) The frozen M4T2 direct estimator again meets the operational sufficiency criterion in independent participants; (2) a repeat-A vs repeat-B geometry-agreement monitor computed from the 8 calibration observations prospectively predicts held-out calibration success. Analysis is preregistered/frozen (O2.16 config `2da2cc79`) before acquisition.

**D. Why independent replication is necessary.** All prior evidence comes from the same 8 NSD subjects; generalization requires a non-overlapping cohort, ideally a different site/scanner.

**E. Participant count.** Planned confirmatory **N=12** (analysis floor N=8); plus an optional **engineering pilot ≤2** (excluded from confirmatory analysis).

**F. Recruitment criteria.** Healthy adults, MRI-safe, normal or corrected-to-normal vision, able to perform perception and imagery tasks, no overlap with the development cohort. No exclusion based on task/scientific performance.

**G. MRI procedures.** [PLACEHOLDER — site] 7T (preferred) functional + structural MRI; parameters per `scanner_site_requirements.md` once a facility confirms a sequence.

**H. Perception task.** Continuous-recognition viewing of 512 canonical anchor images × 3 presentations (3 s image, 1 s gap), central fixation, button response.

**I. Imagery task.** Cued mental imagery of 12 identities (6 simple + 6 naturalistic) × 8 repeats; ~4 s trials with a 2-alternative decision (structure recovered from sealed NSD-Imagery provenance; exact full run/session layout to be finalized — see `imagery_task_exact_replay_spec.json`).

**J. Approximate participant burden.** Task acquisition lower bounds: perception ≈ 1536×4 s, imagery ≈ 96×4 s (see `session_plan.csv`). **Total session time is deliberately left TBD** until the exact run/session structure and site setup times are confirmed — no fabricated total.

**K. Foreseeable risks.** Standard MRI risks; possible mild fatigue/discomfort from lying still and sustained attention. [PLACEHOLDER — institutional MRI risk language.]

**L. Incidental-finding handling.** [PLACEHOLDER — INSTITUTION-SPECIFIC procedure and clinical referral pathway.]

**M. Compensation.** [PLACEHOLDER — human decision, per institutional policy.]

**N. Withdrawal procedure.** Participants may withdraw at any time without penalty; data-withdrawal handling per [PLACEHOLDER — institutional policy].

**O. Pseudonymization.** Participants assigned pseudonymous IDs (`REP-NNN`); identity↔ID mapping held only in a secured recruitment store separate from the research dataset (see `data_flow_and_access_matrix.csv`).

**P. Storage / access controls.** [PLACEHOLDER — institutional secure storage + role-based access.]

**Q. Retention / deletion policy.** [PLACEHOLDER — institutional retention schedule.]

**R. Publication / data-sharing intention.** Aggregate results and analysis code shared openly; raw data sharing per [PLACEHOLDER — institutional/consent terms]; no raw data in the code repository.

**S. Preregistration statement.** The held-out outcome analysis is **frozen before acquisition** (O2.16 config `2da2cc79`); Stage A (calibration + quality monitor) is sealed and server-verified before any held-out imagery outcome is opened in Stage B.
