# Revision log — P3 (Methods closure + literature verification)

Wording/citation/disclosure changes only. **No scientific result, number, classification, statistical test, or
gate status altered.** Every Methods fact verified against a sealed artifact; every central citation verified
against a primary source.

## Manuscript files revised (`paper/mindir/`)
- **METHODS.md §3 (ROI):** exact atlas sources — `prf-visualrois` (V1=1,2; V2=3,4; V3=5,6; hV4=7) and `streams`
  (ventral=5, lateral=6, parietal=7); note that label 5 = V3 (prf) vs ventral (streams); primary = ventral +
  lateral; parietal reserved; V1 reference-only. [verified: `roi_selection_manifest.json` b123072]
- **METHODS.md §4 (response estimation + voxel selection):** resolved both P2 placeholders — B0 = `nsdimagery
  betas_fithrf` (b2-compatible `betas_fithrf`; "similar to b2" ≠ bitwise; GLMdenoise_RR = B1 sensitivity only);
  native `func1pt8mm` 1.8-mm grid, no resampling, HDF5 reversed-axis layout note; eligible = ROI mask ∩ valid
  NSD-Imagery ∩ finite NSD-core ncsnr; strict > ROI-specific 98th percentile; per-ROI; identical voxels B0/B1;
  no threshold adaptation; subj01 QC example (ventral 7604/153, lateral 7799/156) as example only. [verified:
  `beta_version_evidence.json` 3be9303, `roi_selection_manifest.json` b123072, `subj01_spatial_alignment.json`
  b7ec53e, `final_method_concordance.csv` 03b8fa3]
- **ABSTRACT.md:** added the verified short-form framing sentence + Saha Roy et al. (2025) anchor in Background.
- **MANUSCRIPT.md (Introduction outline):** added verified prior-work anchors (Saha Roy 2025; Ward, Isik & Chun
  2018; Haxby 2011/2020; Chen 2015; MIRAGE/Kneeland 2026), the short-form framing sentence, and a pointer to the
  verified citation audit.
- **DISCUSSION.md:** added a "Relation to prior work" paragraph with the certified combination-of-contributions
  statement and the four differentiation statements (Roy / MIRAGE / hyperalignment-SRM / Ward). No priority
  claim.
- **review/literature_positioning_matrix.csv:** replaced with the verified enriched matrix (identical to
  `p3/updated_literature_positioning_matrix.csv`).
- **review/novelty_statement.md:** P3 verification note added; superseded the "human must verify" caveat.

## Not changed (immutable)
All O2 and X1–X4 statuses and numbers; O2.16 / O2.16-DATA / O2.16-SEC statuses; O3_NOT_READY; discovery-stop.
`evidence_table.csv` values unchanged. `claims_matrix.csv` scientific rows unchanged.

## Remaining human-verification items (small)
- Confirm the high-confidence DOIs for Haxby et al. 2011 (10.1016/j.neuron.2011.08.026) and Allen et al. 2022
  (10.1038/s41593-021-00962-x) at submission.
- Decide the final citation style / reference list formatting.
