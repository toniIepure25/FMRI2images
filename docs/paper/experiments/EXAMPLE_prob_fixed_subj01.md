# Experiment: Probabilistic reconstruction (example)

## Goal

Demonstrate a probabilistic reconstruction run that produces:

- reconstructions
- a manifest with sample-level identity
- evaluation outputs (CSV/JSON/fig) with strict matching

## Setup

- **Git commit**: (fill from manifest)
- **Subject**: `subj01`

## Command

Use the repo’s one-click runner (if available in your workflow) or call the scripts directly.

## Inputs

- Index: `data/indices/nsd_index/subject=subj01/index.parquet`
- CLIP cache: `outputs/clip_cache/clip.parquet`
- (Optional) preprocessing: `checkpoints/preproc/...`

## Outputs

- Recon dir: `outputs/recon/subj01/<run_name>/`
- Manifest: `outputs/recon/subj01/<run_name>/manifest.json`
- Reports:
  - `outputs/reports/subj01/<run_name>/recon_eval_matched.csv`
  - `outputs/reports/subj01/<run_name>/recon_eval_summary.json`

## Expected results

- Non-empty matched CSV.
- If gallery is large, Retrieval@10 may be 0.0 with large ranks.

## Notes

This card is a template/example—replace with your real run and the exact commands you used.
