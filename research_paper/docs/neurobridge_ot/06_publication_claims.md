# NeuroBridge-OT: Publication Claims

## Claim-Evidence Matrix

| Claim | Required Evidence | Status |
|---|---|---|
| Learned ROI tokens outperform fixed summaries | Ablation: full vs roi_summary_baseline, p<0.05 | Pending experiment |
| OT alignment improves LOSO generalization | Ablation: full vs no_ot on LOSO protocol | Pending experiment |
| Teacher distillation improves seen-subject retrieval | Ablation: full vs no_teacher on seen subjects | Pending experiment |
| Hyper-adapters improve few-shot data efficiency | Data-efficiency curves: adapter vs scratch | Pending experiment |
| vMF kappa correlates with retrieval quality | Selective prediction analysis | Pending experiment |
| Subject adversarial training improves invariance | LOSO + subject classifier accuracy | Pending experiment |
| Architecture supports true zero-shot transfer | LOSO R@1 > random baseline | Pending experiment |

## Allowed Wording

### When results exist and are significant:
- "NeuroBridge-OT achieves X% R@1 on LOSO generalization."
- "Optimal-transport alignment improves LOSO R@1 by Y percentage points (p<0.05, Cohen's d=Z)."
- "Few-shot adaptation with 100 samples recovers W% of subject-specific performance."

### When results are pending:
- "We evaluate NeuroBridge-OT on X protocol; results are pending."
- "The architecture supports Y evaluation regime; experiments are in progress."

### When results are not significant:
- "While the difference was not statistically significant (p=X), the trend suggests..."
- "OT alignment did not significantly improve LOSO generalization in this setting."

## Forbidden Overclaims

1. **Never claim**: "NeuroBridge-OT solves subject-independent brain decoding."
2. **Never claim**: "This is the first architecture to..." (unless verified with literature search).
3. **Never claim**: Zero-shot performance matches subject-specific models (extremely unlikely).
4. **Never report** SHARED1000 metrics from a model that used SHARED1000 data for training/distillation.
5. **Never label** an experiment as "zero-shot" if the target subject had ANY learned parameters or labeled targets.
6. **Never present** ablation results without specifying which subjects/seeds were used.
7. **Never extrapolate** from 4-subject results to "all subjects" without qualification.

## Classification Rules

| Condition | Label |
|---|---|
| Target subject is in training set, has per-subject parameters | `multi_subject_seen_subject` |
| Target subject is NOT in training set, has NO parameters | `true_unseen_subject_zero_shot` |
| Target subject is NOT in training, gets N labeled samples for adaptation | `few_shot_adaptation` |
| Target subject has unsupervised calibration only (no image labels) | `true_unseen_subject_zero_shot` |
| Target subject has labeled calibration data | `few_shot_adaptation` |
| SHARED1000 predictions used for teacher → training | INVALID (benchmark contamination) |

## Appropriate Novelty Claims

To our knowledge, this combination of the following has not been evaluated together in fMRI-to-CLIP retrieval:
- Differentiable optimal-transport ROI alignment.
- Hyper-network subject adapters from anatomical fingerprints.
- Subject-adversarial gradient reversal for brain representation invariance.
- vMF uncertainty estimation with confidence-aware retrieval.

**Important**: These are architectural contributions. Whether they improve performance is an empirical question answered by the experiments above.
