# Actual-site engineering pilot checklist (NON-CONFIRMATORY)

Every pilot is `NON_CONFIRMATORY` and must never be pooled with the confirmatory cohort. Neural effect size is
**not** a go/no-go criterion. Do not mark any item true until it has actually happened.

## Before scan
- [ ] Site config populated (candidate → real values by the MRI physicist).
- [ ] Protocol version + hash recorded.
- [ ] Stimulus hashes verified.
- [ ] Stimulus permissions cleared for pilot display.
- [ ] MR operator approval to run.
- [ ] Ethics / pilot authorization as required by the institution.
- [ ] Response-device test (button mapping).
- [ ] Display test (resolution, refresh, fullscreen).
- [ ] Trigger test (key/code, ITI, 100+ pulses; see the trigger validation procedure).

## During scan
- [ ] Trigger log captured (`*_scansync.tsv`).
- [ ] Frame-timing log captured.
- [ ] Response log captured.
- [ ] Operator notes / protocol-deviation notes recorded live.

## After scan
- [ ] DICOM received.
- [ ] BIDS conversion run (`dcm2niix`, pinned).
- [ ] Official `bids-validator` run.
- [ ] Events ↔ BOLD volume reconstruction checked.
- [ ] Trigger ↔ volume alignment checked.
- [ ] QC report produced (PASS / WARN / FAIL).
- [ ] Protocol deviations documented and versioned.

Pilot outputs carry the marker `PILOT_ONLY_DO_NOT_POOL_WITH_CONFIRMATORY`.
