# NeuroBridge-OT: Experiment Protocols

## Taxonomy

Every experiment is labeled with exactly one taxonomy label:

| # | Label | Definition |
|---|---|---|
| 1 | `single_subject_same_subject` | Train on subj X, evaluate on subj X |
| 2 | `subject_specific_replication` | Independent per-subject training |
| 3 | `multi_subject_seen_subject` | Joint training on N subjects, evaluate on training subjects |
| 4 | `true_unseen_subject_zero_shot` | Target subject has NO learned parameters or calibration |
| 5 | `loso_generalization` | Leave-one-out: train on N-1, eval on held-out |
| 6 | `few_shot_adaptation` | Pretrain + adapt with N labeled target samples |
| 7 | `data_efficiency_transfer` | Compare adaptation vs scratch at various N |

## Protocol 1: Single-Subject

- Train NeuroBridge-OT on one subject only.
- Eval on same subject's validation/SHARED1000.
- Establishes per-subject upper bound for the architecture.
- Config: `neurobridge_ot_full.yaml` with `subjects: ["subj01"]`.

## Protocol 2: Subject-Specific Replication

- Run Protocol 1 independently for each of subj01/02/05/07.
- Compare against V61a/V62a baselines (if checkpoints available).

## Protocol 3: Multi-Subject Seen-Subject

- Train jointly on subj01/02/05/07.
- Evaluate each subject on its own SHARED1000.
- Tests whether shared model matches per-subject models on seen subjects.
- Config: `neurobridge_ot_seen_subject_multisubject.yaml`.

## Protocol 4: True Zero-Shot

- Train on subj01/02/05 (no target subject data at all).
- Evaluate on subj07 SHARED1000.
- Target subject has NO learned parameters, NO adapter, NO calibration.
- Only subject fingerprint (unsupervised ROI stats) is allowed.
- Config: `neurobridge_ot_loso.yaml` (single fold).

## Protocol 5: LOSO Generalization

- 4-fold LOSO (or 8-fold with all NSD subjects if data available).
- Each fold trains on N-1 subjects, evaluates on held-out.
- Report mean and per-fold R@1.
- Script: `run_loso_neurobridge_ot.py`.
- Config: `neurobridge_ot_loso.yaml`.

## Protocol 6: Few-Shot Adaptation

- Pretrain on source subjects.
- Adapt to target with N labeled samples: 10, 25, 50, 100, 250, 500, 1000.
- Test adaptation modes: adapter-only, head-only, full fine-tune.
- Script: `run_fewshot_adaptation.py`.
- Config: `neurobridge_ot_fewshot.yaml`.

## Protocol 7: Data Efficiency Transfer

- Same as Protocol 6 but also trains from scratch with same N samples.
- Generates data-efficiency curves showing transfer benefit.
- Critical comparison: adapted model should outperform scratch at low N.

## Split Hygiene Rules

1. SHARED1000 is **always excluded** from training/validation.
2. Splits are by unique `nsdId` (image), never by trial.
3. Validation images are never used for hyperparameter selection in final evaluation.
4. Teacher predictions for distillation use **training images only**.
5. SHARED1000 is used **only for final paper-grade evaluation**.

## Evaluation Gallery

- For seen-subject eval: gallery = all unique validation embeddings for that subject.
- For SHARED1000 eval: gallery = all ~982 SHARED1000 image embeddings.
- CSLS k is selected on validation, then frozen for SHARED1000.
