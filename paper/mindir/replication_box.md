# Box — Independent replication already preregistered (not yet executed)

> **No independent participant has yet been acquired.** The independent cohort experiment (O2.16) and its
> secondary geometric endpoints (O2.16-SEC) are **prospectively frozen** but **not executed**. Nothing in this
> manuscript is an independent replication result.

## O2.16 — primary independent replication (frozen config `2da2cc791054…`)
A genuinely independent N ≥ 8 cohort, run through the frozen O2.16 protocol **unchanged**:
- **Stage A** (no held-out): fit `W_target`, 6 outer folds, all M4T2 fits, and the operational 1-vs-1
  calibration-quality monitors `Q_MON_IN` / `Q_MON_OUT`; hash and freeze; commit / push / server-verify.
- **Stage B** (open held-out): total recovery / full-capacity fraction → M4T2 operational replication →
  schedule margins → participant-level Q associations → exact four-test Holm → seal.
- Primary criterion: the frozen 75% operational criterion and the primary four-test family, unchanged.
- Current status: **`O2_16_INDEPENDENT_REPLICATION_DATA_UNAVAILABLE`** (no eligible independent cohort exists).
- Human enablement: `O2_16_DATA_AWAITING_HUMAN_ACQUISITION_AUTHORIZATION` (acquisition protocol `804d5b21fc0c…`
  and a pre-scanning readiness package prepared; no ethics/scanner/recruitment records fabricated).

## O2.16-SEC — three preregistered secondary geometric endpoints (frozen config `37708168…`)
Exactly **3 findings × 2 ROIs = 6 tests**, Holm @ 0.05, participant-level sign-flip, evaluated **only** on the
independent cohort and **only after** the primary O2.16 result is frozen (primary-first; secondary can never
modify primary status):
- **S1 — X1 `B_angles`**: donor-mean leave-one-subject-out `E_SHARED_angle` + independent pipeline-triviality
  control. Support: median > 0, ≥ 75% participants > 0, Holm-reject, triviality passes.
- **S2 — O2.15 `Q_OUT → margin`**: participant Spearman(Q_OUT, robustness margin), Fisher-z (the mechanistic
  O2.15 quantity, **not** the operational Q_MON). Support: median ρ > 0, ≥ 75% > 0, Holm-reject.
- **S3 — X4 `ANISO_CENTRED`**: centred top-fraction vs isotropic null (never the ratio) + ≥ 5/6-fold cross-half
  reproducibility + isotropic norm/rank-matched triviality control. S3 tractability frozen: preferred all-100
  subsets + ≥ 5000 nulls, deterministic fallback = historical 20-of-100 + 1000 nulls, chosen on an outcome-
  independent Stage-A feasibility check.

**Permanently excluded from the positive family** (immutable negatives; reference-only): X1 spectral/support/
compression, X2 K2−K1, X3 angular scaffold, X3 scale→margin, X4-H2 MODE_GAP→margin.

**Preregistration status:** `O2_16_SECONDARY_ENDPOINTS_PREREGISTERED_AWAITING_INDEPENDENT_COHORT` — not
evidence.
