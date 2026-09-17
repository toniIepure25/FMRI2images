# HUMAN_ACTION_CHECKLIST — O2.16-DATA independent replication

> These steps require **real human/institutional action** and cannot be performed or certified by the agent. Ordered by dependency. Nothing below is complete until a human confirms it with real evidence.

1. [ ] Identify PI / responsible investigator.
2. [ ] Identify MRI facility (7T preferred).
3. [ ] Confirm scanner-sequence feasibility against `scanner_site_requirements.md` (fill the `TBD_BY_MRI_FACILITY` column).
4. [ ] Resolve exact imagery-task replay provenance (full run/session layout, inter-run rest, dummy volumes, naturalistic-family identities) — clears `O2_16_DATA_IMAGERY_TASK_PROVENANCE_BLOCKER`.
5. [ ] Confirm stimulus-use permissions (perception anchors + imagery simple/naturalistic sets) — resolve `stimulus_permissions_action_items.csv`.
6. [ ] Finalize participant information / consent (`participant_information_consent_draft.md`) with real institutional details.
7. [ ] Finalize data-protection plan (`data_flow_and_access_matrix.csv`) with real roles/storage.
8. [ ] Submit ethics/IRB package (`ethics_application_draft.md`).
9. [ ] Obtain formal ethics approval.
10. [ ] Obtain MRI facility / site approval.
11. [ ] Book pilot scanner time.
12. [ ] Run ≤2 engineering pilot subjects (`pilot_execution_protocol.md`).
13. [ ] Certify actual-site BIDS/event logging on real hardware.
14. [ ] Certify scanner compatibility (all representation-critical rows compatible).
15. [ ] Begin confirmatory recruitment (target N=12).
16. [ ] Collect planned N=12.
17. [ ] Freeze the replication cohort manifest **before** opening any held-out outcome.
18. [ ] Resume the frozen **O2.16** (config `2da2cc79`) UNCHANGED — Stage A (calibration + Q_MON, no held-out) → server-verify → Stage B (evaluate + seal).

**Items the agent CANNOT generate (require real external evidence):** ethics approval, consent signatures, scanner booking, sequence confirmation, human recruitment/screening, acquired MRI data, pilot completion, real-site BIDS validation, confirmatory cohort acquisition. These remain explicitly incomplete.
