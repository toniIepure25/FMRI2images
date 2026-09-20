# Method-exactness audit (P3)

Searched the manuscript for every term the frozen pipeline constrains. Each occurrence checked for consistency
with the sealed evidence (`beta_version_evidence.json`, `roi_selection_manifest.json`,
`subj01_spatial_alignment.json`, `final_method_concordance.csv`).

| Term | Hits | Location(s) | Consistent with frozen pipeline? |
|---|---|---|---|
| `GLMdenoise` | 1 | METHODS §4 | YES — named only as the **B1 (b3-compatible) sensitivity branch**, explicitly not the primary. |
| `RR` | 1 | METHODS §4 (GLMdenoise/RR) | YES — sensitivity branch. |
| `b2` | 1 | METHODS §4 | YES — B0 = betas_fithrf = b2-compatible (primary). |
| `b3` | 1 | METHODS §4 | YES — b3 = GLMdenoise_RR = B1 sensitivity. |
| `fithrf` | 3 | METHODS §4 (beta prep; ncsnr path) | YES — primary preparation + ncsnr source path. |
| `ncsnr` | 2 | METHODS §4 | YES — finite NSD-core ncsnr eligibility + strict >98th-pct rule. |
| 98th percentile | present | METHODS §4 | YES — strict `>` ROI-specific 98th percentile (not "top 2%"). |
| `voxel` | 9 | METHODS §4, LIMITATIONS | YES — eligible-set + strict-percentile rule; voxel-selection flagged as a bounded interpretation. |
| `top 2%` | 1 | METHODS §4 | YES — appears **only** in the negation ("not a generic 'top 2% of voxels'"). |
| `resampl` | 1 | METHODS §4 | YES — "no spatial resampling"; HDF5 reversed axes = layout convention. |

## Findings
- **No inconsistency found.** GLMdenoise/RR never presented as primary; the beta preparation is B0/`betas_fithrf`
  (b2-compatible); the voxel rule is the exact frozen strict->98th-percentile intersection rule, never
  simplified to "top 2%"; spatial handling is native `func1pt8mm` with no resampling and a correctly described
  reversed-axis HDF5 layout.
- subj01 counts (7604/153 ventral; 7799/156 lateral) appear only as implementation/QC examples in Methods, not
  as cohort-wide counts.

**Status: PASS — all Methods-exactness terms consistent with the frozen pipeline.**
