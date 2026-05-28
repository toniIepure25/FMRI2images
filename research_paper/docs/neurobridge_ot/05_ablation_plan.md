# NeuroBridge-OT: Ablation Plan

## Scientific Hypotheses and Ablation Tests

### Hypothesis 1: Learned ROI tokens outperform fixed summaries

**Ablation**: `neurobridge_ot_roi_summary_baseline.yaml` vs `neurobridge_ot_full.yaml`

**Difference**: Tokenizer type (`fixed_summary` vs `learned`).

**Expected outcome**: Learned tokenizer captures richer per-voxel information than mean/std/max.

**Metrics**: R@1 (val), R@1 (SHARED1000), all subjects.

### Hypothesis 2: OT alignment improves cross-subject transfer

**Ablation**: `neurobridge_ot_no_ot.yaml` vs `neurobridge_ot_full.yaml`

**Difference**: `alignment_mode: identity` vs `alignment_mode: ot`.

**Expected outcome**: OT alignment creates more consistent cross-subject representations, improving LOSO and few-shot performance more than seen-subject performance.

**Metrics**: LOSO R@1, seen-subject R@1, transport matrix entropy.

### Hypothesis 3: Teacher distillation transfers expert knowledge

**Ablation**: `neurobridge_ot_no_teacher.yaml` vs `neurobridge_ot_full.yaml`

**Difference**: `w_teacher: 0.0` vs `w_teacher: 0.3`.

**Expected outcome**: Teacher predictions guide the shared model toward high-performing representations, especially for subjects where teachers are strong.

**Prerequisite**: Teacher predictions must be exported first.

**Metrics**: Seen-subject R@1 (with/without teacher), cosine similarity to teacher predictions.

### Hypothesis 4: Hyper-adapters improve new-subject data efficiency

**Ablation**: `neurobridge_ot_no_hyperadapter.yaml` + few-shot curves vs full + few-shot curves.

**Difference**: `use_hyper_adapter: false` vs `use_hyper_adapter: true`.

**Expected outcome**: Fingerprint-generated adapters provide better initialization for adaptation, showing improved R@1 at low N (10-100 samples).

**Metrics**: Data-efficiency curves (R@1 vs N), adaptation speed (epochs to convergence).

### Hypothesis 5: vMF uncertainty enables selective prediction

**Test**: Compute R@1 on full test set vs top-50%/top-25% by kappa.

**Expected outcome**: Selecting high-kappa predictions yields higher R@1 (reliability).

**Metrics**: R@1 at coverage levels (100%, 75%, 50%, 25%), AUROC of kappa as quality predictor.

### Hypothesis 6: Subject adversarial improves invariance

**Ablation**: `neurobridge_ot_no_adversarial.yaml` vs `neurobridge_ot_full.yaml`

**Difference**: `w_adversarial: 0.0` vs `w_adversarial: 0.1`.

**Expected outcome**: Adversarial training reduces subject-identifiable information in the semantic space, improving LOSO generalization without excessively hurting seen-subject performance.

**Metrics**: LOSO R@1, seen-subject R@1, subject classifier accuracy on shared embeddings.

## Ablation Priority Order

1. Full system (baseline for comparisons).
2. No OT (cheapest architectural change).
3. ROI summary baseline (tests tokenizer).
4. No adversarial (tests invariance training).
5. No teacher (tests distillation).
6. No hyper-adapter (tests adaptation mechanism).
7. LOSO (tests generalization).
8. Few-shot curves (tests data efficiency).

## Statistical Validation

- All comparisons use paired t-tests across subjects.
- Report Cohen's d effect sizes.
- Bootstrap 95% CIs on SHARED1000 metrics.
- Minimum 3 seeds for each key result.
