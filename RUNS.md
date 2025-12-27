# Stage 3/4 Reconstruction & Evaluation Commands

Paper-grade commands (outputs under `outputs/recon/<subject>/<run_name>/`).

## Deterministic (DET_FIXED, K=1)
```bash
python3 scripts/run_reconstruct_and_eval.py \
  --subject subj01 \
  --encoder two_stage \
  --ckpt checkpoints/two_stage/subj01/det_fixed/two_stage_best.pt \
  --clip-cache outputs/clip_cache/clip.parquet \
  --index-root data/indices/nsd_index \
  --output-dir outputs/recon/subj01/det_fixed \
  --report-dir outputs/reports/subj01/det_fixed \
  --trial-csv trial_results.csv \
  --guidance-scale 7.5 \
  --steps 50
```

## Probabilistic (PROB_FIXED, fixed_k K=4)
```bash
python3 scripts/run_reconstruct_and_eval.py \
  --subject subj01 \
  --encoder prob \
  --ckpt checkpoints/two_stage/subj01/prob_fixed/two_stage_best.pt \
  --clip-cache outputs/clip_cache/clip.parquet \
  --index-root data/indices/nsd_index \
  --output-dir outputs/recon/subj01/prob_fixed \
  --report-dir outputs/reports/subj01/prob_fixed \
  --probabilistic \
  --sampling-policy fixed_k --k-base 4 --k-set 1 2 4 \
  --selection-rule likelihood \
  --trial-csv trial_results.csv \
  --guidance-scale 7.5 \
  --steps 50
```

## Probabilistic (PROB_ADAPT_Q, adaptive_quantile K_set [1,2,4,8,16], budget enforced)
```bash
python3 scripts/run_reconstruct_and_eval.py \
  --subject subj01 \
  --encoder prob \
  --ckpt checkpoints/two_stage/subj01/prob_adapt_q_run2/two_stage_best.pt \
  --clip-cache outputs/clip_cache/clip.parquet \
  --index-root data/indices/nsd_index \
  --output-dir outputs/recon/subj01/prob_adapt_q \
  --report-dir outputs/reports/subj01/prob_adapt_q \
  --probabilistic \
  --sampling-policy adaptive_quantile --k-base 4 --k-set 1 2 4 8 16 \
  --adaptive-quantile 0.0 0.5 1.0 \
  --selection-rule likelihood \
  --trial-csv trial_results.csv \
  --guidance-scale 7.5 \
  --steps 50
```

## Orchestrated (all runs + compute vs quality)
```bash
python3 scripts/run_stage34_recon_eval.py \
  --subject subj01 \
  --index-root data/indices/nsd_index \
  --clip-cache outputs/clip_cache/clip.parquet \
  --limit 64
```
(Add `--force` to overwrite existing run directories.)
