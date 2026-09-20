# Reviewer C — fMRI / Reproducibility

Scope: preprocessing, ROI construction, ncsnr/voxel selection, native-space analyses, trial/beta estimation,
cross-subject comparability, task provenance, original-code unavailability, replication limitations, data/code
reproducibility, whether another lab could reproduce the pipeline. Severity: CRITICAL / MAJOR / MODERATE / MINOR.

## Summary judgement
Reproducibility engineering is a strength: prospective config SHAs, server-verified commits, committed source
injected to a fixed compute node, per-gate hashes/provenance/tests, and estimator replay to <=1e-10. The gaps
are in *methodological disclosure in the manuscript itself*, not in the underlying pipeline.

## Major concerns
- **[MAJOR] Beta/voxel-selection disclosure.** The manuscript must state, in Methods, the NSD beta version used,
  the ncsnr/reliability-based voxel selection (if any), and how `W_target` rank and the D=2 outside basis were
  chosen and frozen. A reader cannot presently tell whether voxel selection could induce the outside-support
  residual. Add the exact selection rule and note it was frozen before outcomes.
- **[MAJOR] Cross-subject comparability.** Native-space analyses are not directly comparable across participants;
  the manuscript says cross-subject comparisons used coordinate-invariant quantities only — good, but Methods
  should state this explicitly and note that O2.10 transfer is defined on invariant/donor-mean quantities, not
  raw voxel spaces, so "no transfer" is not an artifact of incomparable spaces.
- **[MODERATE] Task provenance.** The imagery-task field-level provenance is only partly certified
  (`O2_16_DATA_IMAGERY_TASK_PROVENANCE_BLOCKER`). This is honestly flagged, but the manuscript should say which
  imagery-task parameters are certain (from sealed NSD-Imagery provenance) vs unresolved.
- **[MODERATE] Roy reproduction.** Correctly labeled as independent method reproduction/extension with
  `ORIGINAL_CODE_REPRODUCTION_UNAVAILABLE`; ensure this appears in Methods and Limitations, not only in the
  supplement.

## Minor concerns
- **[MINOR]** State smoothing/spatial resolution and single-trial noise as bounds on geometric resolution.
- **[MINOR]** Provide the exact artifact paths + commit SHAs for each figure's source (already in
  `evidence_table.csv`; reference it from figure captions).
- **[MINOR]** Note NSD is public and no raw betas are redistributed.

## What would cause rejection
- Inability of an independent group to reconstruct the pipeline from Methods + repo (currently borderline: repo
  is strong, Methods disclosure needs the beta/voxel-selection specifics).
- Any cross-subject claim that could be a native-space comparability artifact.

## What would change my confidence
- A Methods "Neural response estimation and voxel selection" paragraph with the frozen beta version + selection
  rule.
- Explicit statement that cross-subject analyses use invariant quantities.
- A short reproducibility statement pointing to the per-gate provenance/hashes.
