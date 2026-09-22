# MRI sequence requirements

## Frozen scientific targets (do NOT change to fit a site)
- Whole-brain functional imaging.
- ~1.8 mm isotropic voxel target.
- TR ~1.6 s target.
- Structural T1w ≤ 1 mm; T2w recommended.
- These are analysis-relevant targets; the frozen O2.16 scientific config, estimator, endpoints, folds, and
  design are unchanged regardless of the sequence a site can offer.

## Negotiable site parameters (MRI physicist fills; no fabricated values)
| Field | Value |
|---|---|
| Sequence name | `TBD_SITE_OPERATOR` |
| TR | `TBD_SITE_OPERATOR` |
| TE | `TBD_SITE_OPERATOR` |
| Flip angle | `TBD_SITE_OPERATOR` |
| Voxel size | `TBD_SITE_OPERATOR` |
| Matrix / FOV | `TBD_SITE_OPERATOR` |
| Slices | `TBD_SITE_OPERATOR` |
| Multiband factor | `TBD_SITE_OPERATOR` |
| Partial Fourier | `TBD_SITE_OPERATOR` |
| Phase-encoding direction | `TBD_SITE_OPERATOR` |
| Bandwidth | `TBD_SITE_OPERATOR` |
| Echo spacing | `TBD_SITE_OPERATOR` |
| Distortion correction | `TBD_SITE_OPERATOR` |
| Dummy scans | `TBD_SITE_OPERATOR` |
| Reference scans | `TBD_SITE_OPERATOR` |
| Shim method | `TBD_SITE_OPERATOR` |

## Deviation policy
Any material deviation from the targets is `REVIEW_REQUIRED` and decided by the PI + MRI physicist. If a site's
best feasible acquisition differs from the targets, that is documented as a site constraint — the scientific
design is not tuned to the site, and no outcome is used to choose parameters.
