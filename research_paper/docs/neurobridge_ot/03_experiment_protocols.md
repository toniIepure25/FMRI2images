# NeuroBridge-OT: Experiment Protocols (8-Subject Primary)

## Taxonomy

Every experiment is labeled with exactly one taxonomy label. These labels determine
how results are framed, compared, and cited in publications.

| # | Label | Definition |
|---|---|---|
| 1 | `single_subject_same_subject` | Train on subj X, evaluate on subj X held-out images |
| 2 | `subject_specific_replication` | Independent per-subject training, compare across all 8 |
| 3 | `multi_subject_seen_subject` | Joint training on 8 subjects, evaluate on seen subjects |
| 4 | `strict_zero_shot` | Target has NO learned parameters, NO calibration, NO labels |
| 5 | `unsupervised_calibrated_zero_shot` | Target has NO labels but unsupervised fingerprint/statistics |
| 6 | `loso_strict_zero_shot` | 8-fold LOSO, target evaluated without any target-specific info |
| 7 | `loso_unsupervised_calibrated` | 8-fold LOSO, target fingerprint from unsupervised statistics |
| 8 | `supervised_few_shot_adaptation` | Pretrain + adapt with N labeled target samples |
| 9 | `data_efficiency_transfer` | Compare adaptation vs scratch at various N |

## Required Manifest Fields

Every metrics file and run manifest MUST include:
- taxonomy_label
- training_subjects
- source_subjects
- target_subject
- evaluation_subjects
- target_subject_seen_during_training: true/false
- target_calibration_fmri_used: true/false
- target_labels_used: true/false
- target_adapter_fitted: true/false
- target_learned_parameters_used: true/false
- split_by_image: true/false
- exclude_shared1000: true/false
- gallery_size
- csls_k
- seed
- command
- timestamp

---

## Protocol A — 8-Subject Data Audit

**Goal:** Confirm all 8 subjects are usable and comparable before training.

**Steps:**
1. Run audit for all 8 subjects.
2. Verify fMRI features, index, ROI indices, CLIP cache, SHARED1000 counts.
3. Document any missing data or blockers.

**Command:**
```bash
python3 scripts/neurobridge_ot/audit_neurobridge_data.py \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
    --check-rois --check-teachers --check-shared1000
```

**No model training required.**

---

## Protocol B — Single-Subject NeuroBridge-OT

**Taxonomy:** `single_subject_same_subject`

**Goal:** Establish per-subject upper bounds for NeuroBridge-OT architecture.

**Method:**
- Train NeuroBridge-OT on one subject only.
- Evaluate on same subject's SHARED1000.
- Repeat for all 8 subjects independently.

**Config:** `neurobridge_ot_8subj_subject_specific.yaml`

**Command:**
```bash
for subj in subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08; do
    python3 scripts/neurobridge_ot/train_neurobridge_ot.py \
        --config configs/experiments/neurobridge_ot/neurobridge_ot_8subj_subject_specific.yaml \
        --subjects $subj --seed 42
done
```

---

## Protocol C — 8-Subject Multi-Subject Seen-Subject Training

**Taxonomy:** `multi_subject_seen_subject`

**Goal:** Test whether shared training works across all 8 seen subjects.

**Method:**
- Train jointly on all 8 subjects.
- Evaluate each subject on its own SHARED1000.

**Config:** `neurobridge_ot_8subj_full.yaml`

**Command:**
```bash
python3 scripts/neurobridge_ot/train_neurobridge_ot.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_8subj_full.yaml \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
    --seed 42
```

---

## Protocol D — 8-Fold LOSO Strict Zero-Shot

**Taxonomy:** `loso_strict_zero_shot`

**Goal:** Strict subject-level generalization — no target information at all.

**Method:**
- For each target subject: train on 7 other subjects.
- Evaluate target subject without calibration and without learned target parameters.
- Report mean and per-fold R@1.

**Config:** `neurobridge_ot_8subj_loso.yaml`

**Command:**
```bash
python3 scripts/neurobridge_ot/run_loso_neurobridge_ot.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_8subj_loso.yaml \
    --all-8-folds --mode strict --seed 42
```

---

## Protocol E — 8-Fold LOSO Unsupervised Calibrated

**Taxonomy:** `loso_unsupervised_calibrated`

**Goal:** Evaluate whether unsupervised calibration helps zero-shot performance.

**Method:**
- For each target: train on 7 other subjects.
- Compute target fingerprint from unsupervised target fMRI statistics only.
- No target labels, no supervised adaptation.
- Evaluate target SHARED1000.

**Config:** `neurobridge_ot_8subj_loso.yaml` with `--mode unsupervised_calibrated`

**Command:**
```bash
python3 scripts/neurobridge_ot/run_loso_neurobridge_ot.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_8subj_loso.yaml \
    --all-8-folds --mode unsupervised_calibrated --seed 42
```

---

## Protocol F — 8-Subject Few-Shot Adaptation

**Taxonomy:** `supervised_few_shot_adaptation` / `data_efficiency_transfer`

**Goal:** Measure data-efficiency transfer across all 8 subjects.

**Method:**
- For each target subject: pretrain on other 7 subjects.
- Adapt with 10, 25, 50, 100, 250, 500, 1000, full target examples.
- Compare to scratch with same sample count.
- Test adaptation modes: adapter-only, head-only, full fine-tune.

**Config:** `neurobridge_ot_8subj_fewshot.yaml`

**Command:**
```bash
python3 scripts/neurobridge_ot/run_fewshot_adaptation.py \
    --config configs/experiments/neurobridge_ot/neurobridge_ot_8subj_fewshot.yaml \
    --all-target-subjects \
    --few-shot-sizes 10 25 50 100 250 500 1000 \
    --seed 42
```

---

## Protocol G — Ablation Suite

**Taxonomy:** varies per ablation

**Goal:** Controlled ablations isolating each component's contribution.

**Ablations:**
1. Fixed ROI summary baseline (no learned tokenization)
2. Learned ROI tokenizer, no OT
3. Learned tokenizer + OT (no anatomical prior)
4. Learned tokenizer + OT + anatomical prior
5. No teacher distillation
6. With teacher distillation (teacher-coverage-limited)
7. No vMF uncertainty head
8. With vMF uncertainty head
9. No adversarial loss
10. With adversarial loss
11. No hyper-adapter
12. With hyper-adapter

**Config:** Use corresponding ablation YAML files.

**Command:**
```bash
python3 scripts/neurobridge_ot/run_ablation_suite.py \
    --configs-dir configs/experiments/neurobridge_ot/ \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
    --limit-batches 100 --seed 42
```

---

## Split Hygiene Rules

1. SHARED1000 is **always excluded** from training/validation.
2. Splits are by unique `nsdId` (image), never by trial.
3. Validation images are never used for hyperparameter selection in final evaluation.
4. Teacher predictions for distillation use **training images only**.
5. SHARED1000 is used **only for final paper-grade evaluation**.
6. Gallery size must be reported in every manifest.
7. Metrics are comparable ONLY if gallery nsdIds match.

## Evaluation Gallery

- For seen-subject eval: gallery = all unique validation embeddings for that subject.
- For SHARED1000 eval: gallery = 1000 SHARED1000 image CLIP embeddings (identical across subjects).
- CSLS k selected on validation, frozen for SHARED1000.
- Gallery size variation: if any subject has <1000 valid SHARED1000 entries, document why.

---

## Legacy 4-Subject Comparison

The previous 4-subject setup (subj01/subj02/subj05/subj07) is preserved for:
- Direct comparison against V66a.
- Continuity with generalization workstream results.
- NOT for publication-level claims.

**Config:** `neurobridge_ot_4subj_legacy_v66a_comparison.yaml`

All publication-level generalization claims should preferably use 8-subject evidence.
