# Data flow (real site) — identity vs research separation

Generic data-flow spec. It does not assume any specific hospital IT system; the exact systems are
`TBD_SITE_OPERATOR`. It marks where personal/identifiable data (PHI) may exist and where research
pseudonymization begins.

```
[SCANNER]
   |  (raw acquisition; may carry PHI in DICOM headers)   <-- PHI ZONE
   v
[DICOM export]  (site PACS / console)                     <-- PHI ZONE
   |  pseudonymization step (strip identifiers; assign sub-<pseudonym>)
   v
[Pseudonymized acquisition directory]                     <-- RESEARCH ZONE begins
   |  dcm2niix (pinned)
   v
[BIDS conversion]  (events from the experiment stack)
   |  official bids-validator
   v
[Validated BIDS dataset]  (pseudonymous only)
   |  frozen preprocessing pipeline
   v
[Stage-A predictor-side artifacts]   analysis/stage_a/    (NO held-out outcomes)
   |  (human STAGE_B_RELEASE.json required)
   v
[Stage-B protected outcomes]         analysis/stage_b_protected/
```

## Key points
- **PHI zone:** scanner + raw DICOM (headers may contain identifiers). Pseudonymization happens **before** data
  enter the research zone; the research dataset uses `sub-<pseudonym>` only.
- **Identity mapping** (pseudonym ↔ person) is stored **outside** the research dataset per site policy, never in
  BIDS scientific output.
- **No cloud upload** of identifiable participant information is implemented or assumed.
- **Stage-A/Stage-B firewall:** Stage-A code cannot read `analysis/stage_b_protected/`; Stage B stays locked
  until a human-created `STAGE_B_RELEASE.json` with full certification exists.
- Every concrete system (PACS, storage, network) is `TBD_SITE_OPERATOR` until the facility specifies it.
