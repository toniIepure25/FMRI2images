# NeuroBridge-OT: Implementation Report

## Files Added

### Core Architecture (`src/fmri2img/models/neurobridge_ot/`)
- `__init__.py` — Package exports.
- `roi_tokenizer.py` — 4 tokenizer variants (Fixed, Learned, SubjectAdaptive, HyperAdapter).
- `subject_fingerprint.py` — Subject fingerprint computation module.
- `optimal_transport.py` — Sinkhorn OT alignment with canonical token bank.
- `semantic_transformer.py` — Pre-norm transformer with stochastic depth.
- `decoding_heads.py` — CLIP, Token, vMF, Calibration heads.
- `teacher_distillation.py` — Teacher registry + 3 distillation loss types.
- `hyper_adapter.py` — HyperNetwork (FiLM/LoRA) + AdaptationController.
- `losses.py` — Composite loss with 9 configurable components.
- `model.py` — Top-level NeuroBridgeOTModel integrating all modules.

### Dataset (`src/fmri2img/data/`)
- `neurobridge_dataset.py` — Multi-subject dataset with protocol support.

### Factory Integration
- `src/fmri2img/models/unified_model.py` — Added `neurobridge_ot` branch to `create_model()`.

### Scripts (`scripts/neurobridge_ot/`)
- `train_neurobridge_ot.py` — Full training entrypoint.
- `eval_neurobridge_ot.py` — Evaluation with CSLS support.
- `run_loso_neurobridge_ot.py` — LOSO orchestration.
- `run_fewshot_adaptation.py` — Few-shot adaptation with data-efficiency curves.
- `export_teacher_predictions.py` — Teacher artifact export.
- `run_ablation_suite.py` — Ablation runner.
- `audit_neurobridge_data.py` — Data availability checker.
- `summarize_neurobridge_results.py` — Results aggregation.

### Configs (`configs/experiments/neurobridge_ot/`)
- 10 YAML configs covering full system, ablations, protocols, and smoke test.

### Tests (`tests/`)
- `test_neurobridge_ot.py` — Unit tests for all modules.
- `test_neurobridge_ot_integration.py` — Integration/smoke tests.

### Documentation (`research_paper/docs/neurobridge_ot/`)
- 8 documentation files covering overview through run commands.

## Integration Points

1. **Model factory**: `create_model({"type": "neurobridge_ot", ...})` returns `NeuroBridgeOTModel`.
2. **ROI indices**: Reuses existing `build_roi_index()` from `fmri2img.data.roi_utils`.
3. **CLIP embeddings**: Same parquet-based loading pattern.
4. **Evaluation**: Compatible with existing `compute_retrieval_metrics`.

## What Is Fully Implemented

- All architecture modules with full feature flags.
- All 4 tokenizer variants.
- Differentiable Sinkhorn OT with 4 alignment modes.
- 9-component configurable loss.
- HyperNetwork with FiLM and LoRA generation.
- Subject adversarial with gradient reversal.
- vMF uncertainty with softplus kappa.
- Teacher registry with nsdId validation.
- LOSO and few-shot orchestration scripts.
- Balanced subject sampling.
- SHARED1000 exclusion enforcement.

## What Is Not Yet Run (Pending Compute)

- Full 8-subject multi-subject training on H100 (~12-18 hours).
- 8-fold LOSO evaluation (~48-64 hours).
- Few-shot data-efficiency curves across all 8 targets (~24-36 hours).
- Full ablation suite (~160-240 hours).
- Teacher prediction export (requires V61a/V62a/V66a checkpoints; coverage-limited).

## Known Limitations / Blockers

1. **Teacher artifacts**: V61a/V62a/V66a checkpoints must exist on the pod. Teacher coverage is limited to subj01/02/05/07 (V66a) and subj01-only (V61a/V62a). Run teacher experiments as "teacher-coverage-limited."
2. **Token cache**: Rich token targets (257×768) require `make token-clip-cache` to have been run.
3. **8-subject data preparation**: All 8 subjects need `make index` and `make preextract`. The 4 legacy subjects (subj01/02/05/07) likely already have features; subj03/04/06/08 need preparation before first use.
4. **Memory**: Full 8-subject system with d_model=768 may require 50-60GB for multi-subject training on H100.
