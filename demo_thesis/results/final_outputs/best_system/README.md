# Final Best System

This directory contains the frozen final retrieval system outputs for thesis/report/presentation.

## Best System

The production retrieval system is the fixed tri-expert fusion built from:

- tri/main results dir: `experimental_results/V35_legacy_teacher_distill/subj01`
- legacy results dir: `experimental_results/N1v28a_dual_head/subj01`

Frozen fusion recipe:

- compact score: `csls`
- legacy score: `csls`
- family: `normalized_weighted`
- normalization: `zscore`
- shortlist_k: `150`
- alpha / beta / gamma: `0.3 / 0.0 / 0.7`
- expected SHARED1000 R@1: `77.2%`

## Final Files

- `final_metrics_summary.json`: compact final metrics bundle for VAL and SHARED1000
- `final_metrics_summary.csv`: report-friendly table of the main systems/metrics
- `per_query_val_predictions.csv`: per-query frozen best-system predictions on VAL
- `per_query_shared1000_predictions.csv`: per-query frozen best-system predictions on SHARED1000

## Qualitatives And Reconstructions

- `qualitatives/val/`
- `qualitatives/shared1000/`
- `qualitatives/contact_sheets/`
- `reconstructions/val/`
- `reconstructions/shared1000/`

## Regeneration

Metrics bundle:

```bash
python3 scripts/evaluation/export_final_best_system_bundle.py   --tri-results-dir experimental_results/V35_legacy_teacher_distill/subj01   --legacy-results-dir experimental_results/N1v28a_dual_head/subj01
```

Qualitatives + retrieval reconstructions:

```bash
python3 scripts/evaluation/export_final_qualitatives.py   --tri-results-dir experimental_results/V35_legacy_teacher_distill/subj01   --legacy-results-dir experimental_results/N1v28a_dual_head/subj01   --stimuli-hdf5 /path/to/nsd_stimuli.hdf5
```

## Notes

- This package freezes the validated fixed tri-expert system exactly.
- No new training is introduced by these export scripts.
- Retrieval-based reconstruction uses the fused top-1 retrieved image as the final reconstruction.
- Optional generative export is intentionally not part of the default pipeline.
