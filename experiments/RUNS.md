# RUNS: Probabilistic fMRI→CLIP Decoder (Adaptive Diffusion)

This sheet enumerates the **paper-ready run matrix** with config files that map 1:1 to the ablations. All configs live under `configs/experiments/probabilistic/` and inherit `configs/base.yaml`, keeping deterministic baselines intact. Target feature branch for this work: `feat/probabilistic-clip-decoder` (create when publishing).

## Naming conventions
- **Run name**: `{stage}_{variant}_env[R{0|1}_P{0|1}]_K{base}_sel[{rule}]_unc[{diag|scalar}]`
- **Stages**:
  - `stage1`: sanity/plumbing (DET_FIXED, PROB_FIXED, PROB_ADAPT_Q)
  - `stage2_env`: env grid reliability×PCA (4 combos) applied to each stage1 recipe
  - `stage2_uncertainty`: diag vs scalar logvar
  - `stage2_selection`: likelihood vs cosine (oracle = upper bound only)
  - `stage3_compute`: compute curve K_base ∈ {2,4,8,16}
- **Seeds**: default 42 (see `experiments/SEEDS.md`); include in run folder name if changed.

## Core commands (examples)
Replace `<SUBJ>` and checkpoint paths as needed. Diffusion/galleries are unchanged from existing scripts.

**Train (two-stage encoder, deterministic or probabilistic head — paper mode, unique outputs)**
```bash
# DET_FIXED
python scripts/train_two_stage.py \
  --config configs/experiments/probabilistic/stage1/det_fixed.yaml \
  --subject subj01 \
  --index-root data/indices/nsd_index \
  --clip-cache outputs/clip_cache/clip.parquet \
  --output-dir checkpoints/two_stage/subj01/det_fixed \
  --batch-size 4 \
  --num-workers 0 \
  --seed 42 \
  --paper-mode

# PROB_FIXED
python scripts/train_two_stage.py \
  --config configs/experiments/probabilistic/stage1/prob_fixed.yaml \
  --subject subj01 \
  --index-root data/indices/nsd_index \
  --clip-cache outputs/clip_cache/clip.parquet \
  --output-dir checkpoints/two_stage/subj01/prob_fixed \
  --batch-size 4 \
  --num-workers 0 \
  --seed 42 \
  --paper-mode

# PROB_ADAPT_Q
python scripts/train_two_stage.py \
  --config configs/experiments/probabilistic/stage1/prob_adapt_q.yaml \
  --subject subj01 \
  --index-root data/indices/nsd_index \
  --clip-cache outputs/clip_cache/clip.parquet \
  --output-dir checkpoints/two_stage/subj01/prob_adapt_q \
  --batch-size 4 \
  --num-workers 0 \
  --seed 42 \
  --paper-mode
```

**Reconstruct + evaluate (paper assets + calibration)**
```bash
# DET_FIXED reconstruction/eval
python scripts/run_reconstruct_and_eval.py \
  --subject subj01 \
  --encoder det \
  --ckpt checkpoints/two_stage/subj01/det_fixed/two_stage_best.pt \
  --clip-cache outputs/clip_cache/clip.parquet \
  --index-root data/indices/nsd_index \
  --output-dir outputs/recon/subj01/det_fixed \
  --trial-csv trial_results.csv \
  --allow-oracle

# PROB_FIXED reconstruction/eval
python scripts/run_reconstruct_and_eval.py \
  --subject subj01 \
  --encoder prob \
  --ckpt checkpoints/two_stage/subj01/prob_fixed/two_stage_best.pt \
  --clip-cache outputs/clip_cache/clip.parquet \
  --index-root data/indices/nsd_index \
  --output-dir outputs/recon/subj01/prob_fixed \
  --probabilistic \
  --sampling-policy fixed_k \
  --k-base 8 \
  --selection-rule likelihood \
  --trial-csv trial_results.csv \
  --allow-oracle

# PROB_ADAPT_Q reconstruction/eval
python scripts/run_reconstruct_and_eval.py \
  --subject subj01 \
  --encoder prob \
  --ckpt checkpoints/two_stage/subj01/prob_adapt_q/two_stage_best.pt \
  --clip-cache outputs/clip_cache/clip.parquet \
  --index-root data/indices/nsd_index \
  --output-dir outputs/recon/subj01/prob_adapt_q \
  --probabilistic \
  --sampling-policy adaptive_quantile \
  --adaptive-quantile 0.0 0.10 0.30 0.60 0.85 1.0 \
  --k-set 16 8 4 2 1 \
  --k-base 8 \
  --selection-rule likelihood \
  --trial-csv trial_results.csv \
  --allow-oracle
```

Swap the `--config` path for the desired recipe below. For env-grid or selection/uncertainty runs, point to the specific YAML in that subfolder.

## Stage 1 (sanity)
- Deterministic fixed K: `configs/experiments/probabilistic/stage1/det_fixed.yaml`
- Probabilistic fixed K: `configs/experiments/probabilistic/stage1/prob_fixed.yaml`
- Probabilistic adaptive quantile (main): `configs/experiments/probabilistic/stage1/prob_adapt_q.yaml`

**Oracle upper bound (do not report as main metric)**
```bash
python scripts/run_reconstruct_and_eval.py \
  --subject <SUBJ> --encoder prob \
  --ckpt checkpoints/two_stage/<SUBJ>/prob_adapt_q/best.pt \
  --clip-cache outputs/clip_cache/clip.parquet \
  --index-root data/indices/nsd_index \
  --output-dir outputs/recon/<SUBJ>/prob_adapt_q_oracle \
  --report-dir outputs/reports/<SUBJ>/prob_adapt_q_oracle \
  --probabilistic --selection-rule oracle --allow-oracle \
  --sampling-policy adaptive_quantile --adaptive-quantile 0.2 0.6 0.9 --k-set 1 2 4 --k-base 2
```

## Stage 2 (env grid: reliability × PCA)
Use the recipe matching the Stage 1 variant:
- Det: `env_R{0|1}_P{0|1}_det_fixed.yaml`
- Prob fixed: `env_R{0|1}_P{0|1}_prob_fixed.yaml`
- Prob adapt Q: `env_R{0|1}_P{0|1}_prob_adapt_q.yaml`

## Stage 2 (uncertainty parameterization)
- Diagonal (default): `stage2_uncertainty/prob_diag.yaml`
- Scalar logvar: `stage2_uncertainty/prob_scalar.yaml`

## Stage 2 (selection rule ablation)
- Likelihood (main): `stage2_selection/selection_likelihood.yaml`
- Cosine-to-mean: `stage2_selection/selection_cosine.yaml`
- Oracle (upper bound only): `stage2_selection/selection_oracle.yaml`

## Stage 3 (compute curve)
- K_base=2: `stage3_compute/compute_k2.yaml`
- K_base=4: `stage3_compute/compute_k4.yaml`
- K_base=8: `stage3_compute/compute_k8.yaml`
- K_base=16: `stage3_compute/compute_k16.yaml`

## Smoke / CI
- Tiny fast recipe: `configs/experiments/probabilistic/stage1/smoke_tiny.yaml`

## Logging & artifacts (already wired in codebase)
- Manifest/env/git captured via existing manifest utilities (no change required).
- Per-run outputs should include: config dump, git commit, CUDA/torch/GPU info, dataset split IDs, diffusion settings, per-trial CSV/Parquet for uncertainty & selection metrics, and calibration plots (produced by eval stage).
