# Best-Performing System Brief for Gemini Deep Research

Date: 2026-04-22  
Project: fMRI-to-image decoding on NSD  
Purpose: provide a repository-grounded, research-ready English brief about the strongest version in this codebase, so an external research agent can investigate why performance stopped below 90%+ R@1 and what the most credible remedies are.

## 1. Executive Summary

The strongest reportable system currently present in this repository is **not a single newly trained checkpoint**. It is a **frozen inference-time retrieval system** that combines:

- the **V35** trainable compact/triple-head pipeline: `experimental_results/V35_legacy_teacher_distill/subj01`
- the **legacy N1v28a** dual-head retrieval expert: `experimental_results/N1v28a_dual_head/subj01`

The canonical frozen fusion rule is:

- compact score: `csls`
- legacy score: `csls`
- family: `normalized_weighted`
- normalization: `zscore`
- shortlist size: `K=150`
- weights: `alpha=0.3`, `beta=0.0`, `gamma=0.7`

Final exported metrics from the repository bundle:

- **Validation R@1 = 77.6%**
- **SHARED1000 R@1 = 77.2%**

The repository itself frames **MindEye at 93.2% R@1** on the same NSD benchmark as the main reference target. Relative to that headline reference, the final frozen system is **16.0 percentage points lower**.

The most important repository-level conclusion is that the remaining gap is **not explained by shortlist recall alone**. The evidence instead points to a combination of:

- **single-subject data scale limitations**
- **representation ceiling in the current compact expert**
- **decision errors inside already-good shortlists**
- **train/eval mismatch in learned reranking or gating**
- **subject-related nuisance variance and incomplete cross-subject transfer**

## 2. Critical Interpretation Rule

Gemini should keep the following distinction explicit:

- **Best overall practical system**: the frozen compact+legacy fusion described above, at **77.2% SHARED1000 R@1**
- **Best single legacy expert**: `N1v28a_dual_head`, at **70.3% CSLS R@1**
- **Best trained V35-family pipeline before the final frozen export**: weaker than the final frozen system; the repository states that the strongest deployed result comes from explicit fixed fusion at inference time, not from a fully unified training wave

In other words, the best result in this repo is a **composite system**, not a clean single-model endpoint.

## 3. Canonical Evidence Sources

Gemini should treat the following repository files as the primary sources for this brief:

- `docs/thesis/results/final_outputs/best_system/README.md`
- `docs/thesis/results/final_outputs/best_system/final_metrics_summary.json`
- `docs/thesis/results/final_outputs/best_system/final_metrics_summary.csv`
- `docs/EXPERIMENT_CONTEXT.md`
- `configs/experiments/V35_legacy_teacher_distill.yaml`
- `configs/experiments/N1v28a_dual_head.yaml`
- `scripts/evaluation/final_best_system.py`

Useful supporting artifacts:

- `docs/thesis/results/final_outputs/best_system/per_query_val_predictions.csv`
- `docs/thesis/results/final_outputs/best_system/per_query_shared1000_predictions.csv`
- `docs/thesis/results/final_outputs/best_system/qualitatives/...`
- `docs/thesis/results/final_outputs/best_system/reconstructions/...`

## 4. Exact Definition of the Best System

### 4.1 What is frozen

The exported final system is defined in `docs/thesis/results/final_outputs/best_system/README.md` and verified by `scripts/evaluation/final_best_system.py`.

The script hard-codes the approved production recipe:

- `compact_score = csls`
- `legacy_score = csls`
- `family = normalized_weighted`
- `normalization = zscore`
- `shortlist_k = 150`
- `alpha = 0.3`
- `beta = 0.0`
- `gamma = 0.7`
- expected SHARED1000 R@1 = `0.772`

### 4.2 Practical meaning of the weights

Although the tooling still calls the system a "tri-expert" fusion, the frozen best setting gives the rerank channel **zero weight** (`beta=0.0`). Operationally, the best practical system is therefore:

- **compact V35 expert**
- plus **legacy N1v28a expert**
- with fixed z-scored weighted fusion

This matters because it means the final gain does **not** come from a successful learned reranker or gate.

### 4.3 Final metrics table

Canonical metrics from `final_metrics_summary.json`:

| Split | System | R@1 | R@5 | R@10 | MedR | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| VAL | compact raw | 42.0% | 71.9% | 81.0% | 2 | 0.5546 |
| VAL | compact CSLS | 52.8% | 83.1% | 91.1% | 1 | 0.6642 |
| VAL | rerank-only | 51.8% | 83.1% | 92.1% | 1 | 0.6544 |
| VAL | legacy CSLS | 70.1% | 91.7% | 96.7% | 1 | 0.7934 |
| VAL | fixed two-expert fusion | 63.9% | 90.3% | 96.1% | 1 | 0.7537 |
| VAL | **fixed final fusion** | **77.6%** | **96.1%** | **98.6%** | **1** | **0.8547** |
| SHARED1000 | compact raw | 42.1% | 73.2% | 83.6% | 2 | 0.5548 |
| SHARED1000 | compact CSLS | 51.8% | 81.3% | 90.3% | 1 | 0.6484 |
| SHARED1000 | rerank-only | 51.6% | 84.5% | 92.2% | 1 | 0.6592 |
| SHARED1000 | legacy CSLS | 70.1% | 94.1% | 97.4% | 1 | 0.8029 |
| SHARED1000 | fixed two-expert fusion | 64.3% | 90.8% | 95.1% | 1 | 0.7528 |
| SHARED1000 | **fixed final fusion** | **77.2%** | **94.4%** | **97.5%** | **1** | **0.8477** |

Net gains of the final frozen system on SHARED1000:

- over compact CSLS: **+25.4 pp**
- over legacy CSLS: **+7.1 pp**
- over the earlier fixed two-expert fusion: **+12.9 pp**

## 5. Technical Description of the Two Main Components

### 5.1 V35: trainable compact/triple-head base

Config: `configs/experiments/V35_legacy_teacher_distill.yaml`

Purpose:

- keep the V33b/V32 triple-head architecture
- keep the compact shortlist head, rerank head, and regression head
- replace earlier rerank-teacher distillation with **legacy N1v28a teacher distillation**
- checkpoint on **`fused_r@1`**

Data protocol:

- subject: `subj01`
- ROI: `nsdgeneral`
- image-level split: `train 90% / val 10%`
- `exclude_shared1000: true`
- `average_repetitions: false`
- `zscore_mode: per_session`
- CLIP target embedding column: `fused`
- token cache: `outputs/clip_cache/tokens_ViT-L-14_projected.h5`
- rerank cache method: PCA
- rerank cache path: `outputs/rerank_cache/pca_trainonly_dim2048_seed42.npz`

Model:

- type: `vmf_triple`
- encoder: residual MLP with hidden dims `[8192, 8192, 4096, 2048]`
- dropout: `0.25`
- compact retrieval head: vMF, `retrieval_dim=768`
- rerank head: enabled, `rerank_dim=2048`
- token target structure: `257 x 768`
- kappa mode: `softplus`

Losses:

- `vmf_nce` enabled
- `regression_mse` enabled, weight `0.5`
- `rerank_softclip` enabled, weight `1.0`
- `legacy_teacher_distill` enabled, weight `0.15`
- `softclip` enabled, weight `1.0`
- `kappa_reg` enabled
- `shortlist_teacher_distill` disabled

Training:

- batch size `16`
- gradient accumulation `16`
- effective mixed precision: `bf16`
- epochs `100`
- optimizer: AdamW, lr `3e-5`
- EMA enabled
- MixCo enabled

Evaluation:

- CSLS enabled with `k=10`
- SHARED1000 evaluation enabled
- checkpoint metric: **`fused_r@1`**
- two-stage retrieval evaluation enabled
- fusion during training/eval enabled

Interpretation:

- V35 is a **trainable retrieval-first base** with three functional branches
- it is important because it supplies the compact expert used in the final frozen system
- however, the final best score still requires the external legacy expert at inference time

### 5.2 N1v28a: legacy dual-head expert

Config: `configs/experiments/N1v28a_dual_head.yaml`

Purpose:

- implement a MindEye-style dual-head architecture
- combine an L2-normalized contrastive head with an unnormalized regression head
- test whether explicit regression improves absolute placement in embedding space

Data protocol:

- subject: `subj01`
- ROI: `nsdgeneral`
- image-level split with `exclude_shared1000: true`
- token cache: `outputs/clip_cache/tokens_ViT-L-14_projected.h5`

Model:

- type: `vmf`
- residual MLP encoder with `[8192, 8192, 4096, 2048]`
- decoder output is token-space: `257 x 768 = 197,376`
- dual heads:
  - normalized contrastive `mu_head`
  - unnormalized `regression_head`
- `kappa_head` with `softplus`

Losses:

- `vmf_nce`
- `regression_mse`
- `softclip`
- `kappa_reg`

Training:

- batch size `16`
- gradient accumulation `16`
- epochs `300`
- mixed precision `bf16`
- checkpoint metric: `csls_r@1`

Validated result from `docs/EXPERIMENT_CONTEXT.md`:

- **Raw R@1 (shared1000): 56.0%**
- **CSLS R@1 (shared1000): 70.3%**

Repository interpretation:

- dual-head regression helped raw accuracy
- but the system still hit an approximate **70% CSLS ceiling**
- the repo attributes that ceiling mainly to **single-subject data scale**, not merely architecture

## 6. Evaluation Protocol That Matters for This Brief

Gemini should not collapse multiple evaluation regimes into one. The repository uses at least three distinct notions of success:

- **raw retrieval**: ordinary cosine ranking
- **CSLS retrieval**: hubness-corrected retrieval, often materially higher than raw
- **frozen fusion retrieval**: inference-time combination of expert scores after shortlist generation

The final reportable number is the third one:

- **77.2% on SHARED1000 for the frozen final fusion**

This is different from:

- **70.3% CSLS** for the best single legacy expert
- **51.8% compact CSLS** for the compact V35 expert in the final export bundle

This distinction is central to diagnosing the gap to 90%+.

## 7. Experimental Lineage Relevant to the Best System

Gemini does not need the full project history. The following milestones are the most relevant chain of evidence.

### 7.1 V26a: token-target breakthrough

Repository finding:

- token targets were a major jump
- `N1v26a` reached **52.8% raw** and **69.6% CSLS**

Meaning:

- richer targets substantially improved retrieval geometry
- after this point, simple loss-level tweaks gave only small or negative returns

### 7.2 V27a: ViT-bigG did not solve the bottleneck

Repository finding:

- raw R@1 improved from **52.8% to 54.2%**
- CSLS R@1 regressed from **69.6% to 67.2%**
- hub fraction increased from about **6.0% to 8.4%**

Meaning:

- stronger image backbone quality alone was **not enough**
- higher-dimensional targets increased hubness
- fidelity improved, but discriminability worsened

### 7.3 V28a: dual-head regression helped, but the ceiling persisted

Repository finding:

- **56.0% raw**
- **70.3% CSLS**

Repository conclusion:

- dual-head regression improved absolute prediction quality
- but it did **not** remove the main ceiling
- the repo explicitly interprets the remaining bottleneck as **single-subject data volume**

### 7.4 V30 wave: compact retrieval and reranking were separated

Repository finding:

- shortlist recall was already near saturation
- for `V30e`, shortlist recall@100 on SHARED1000 was **99.8%**
- rerank-only performance improved, and oracle rerank reached **62.7%**
- compact+rerank fixed fusion reached **51.5% SHARED1000 R@1**

Meaning:

- the rerank space captured useful local information
- but rerank-score replacement was too destructive
- fusion was better than replacement

### 7.5 V32: stronger compact+rerank fusion, but still not enough

Repository finding:

- V32 fused SHARED1000 R@1 = **57.0%**
- oracle rerank on SHARED1000 = **74.1%**

Meaning:

- strong shortlist-local information existed
- but learned use of that information remained incomplete

### 7.6 V34 and V35: adding the legacy expert changed the game

Repository findings:

- V34 tri-expert fused SHARED1000 R@1 = **75.3%**
- later, re-running fixed tri-fusion on the stronger V35 base produced **77.2% SHARED1000 R@1**

Meaning:

- the largest final gain came from **legacy + compact complementarity**
- not from a new monolithic architecture

### 7.7 V37 and V39/V40: learned decision layers failed to replace fixed fusion

Repository findings:

- V37 learned global gate: **89.6% VAL** but only **67.1% SHARED1000**
- V39 reranker underperformed the fixed baseline
- V40 identified train/eval mismatch from in-sample train expert predictions

Meaning:

- the project found real decision-layer headroom
- but the available learned resolvers were not robust enough
- fixed fusion remained the most trustworthy reportable endpoint

## 8. Evidence-Backed Bottlenecks

This section is the most important input for Gemini. Each point below is grounded in explicit repository evidence.

### 8.1 Bottleneck A: the 77.2% ceiling is not a shortlist-recall failure

Repository evidence from `docs/EXPERIMENT_CONTEXT.md`:

- the post-V35 audit says the **77.2% SHARED1000 ceiling is not a shortlist recall failure**
- union-oracle shortlist recall reaches about **100% by K=50**
- compact and legacy top-1 predictions disagree on **55.1%** of SHARED1000 queries

Interpretation:

- the correct image is usually already present in the candidate set
- many remaining failures happen **inside** the shortlist
- the unresolved problem is candidate ranking or decision integration, not just candidate retrieval

### 8.2 Bottleneck B: representation ceiling in the compact expert

Repository evidence:

- compact raw performance in the final bundle is only **42.1%** on SHARED1000
- compact CSLS performance is **51.8%**
- V28a raised the best single-expert CSLS to **70.3%**, but further gains stalled
- V27a bigG improved raw fidelity but worsened CSLS because of hubness

Interpretation:

- the compact expert is not individually strong enough
- external fusion compensates for this weakness
- the repo repeatedly suggests that the current compact representation is still below the discriminative quality needed for 90%+

### 8.3 Bottleneck C: single-subject data scale

Repository evidence:

- the repo repeatedly states that `subj01` has about **9K unique images**
- it contrasts this with MindEye's approximate **70K** unique-image scale across subjects
- after V28a, the repo explicitly identifies **data volume** as the primary remaining bottleneck

Interpretation:

- even after target-space and loss improvements, single-subject training likely limits generalization
- the best final system still depends on a frozen legacy expert rather than a strong cross-subject unified student

### 8.4 Bottleneck D: hubness and metric mismatch remain structurally important

Repository evidence:

- compact raw vs compact CSLS gaps remain large
- V27a showed better cosine fidelity but worse CSLS because hubness increased
- earlier sections of the repo repeatedly describe hubness as a dominant issue in retrieval geometry

Interpretation:

- the system can get "close" in embedding space without ranking correctly at top-1
- better target fidelity alone does not guarantee higher retrieval accuracy
- 90%+ may require a representation that is intrinsically less hub-prone, not merely better post-hoc correction

### 8.5 Bottleneck E: learned reranking or gating has unresolved train/eval mismatch

Repository evidence:

- V37 strongly overfit validation
- V39 initially had degenerate features due to a masking bug
- V40 concluded that the primary blocker was **train/eval distribution mismatch**
- train caches were based on overly easy in-sample predictions instead of true out-of-fold expert predictions

Interpretation:

- there may still be real headroom in a learned shortlist resolver
- but the repo does not yet contain a validated resolver that beats fixed fusion robustly

### 8.6 Bottleneck F: subject-related nuisance variance is still unresolved

Repository evidence:

- the post-freeze SCFR line was introduced to test subject-invariant factorization
- the repo describes `V44a_all8_scfr_smoke` as an engineering success but a **scientific negative result**
- retrieval remained near-random and subject disentanglement was weak

Interpretation:

- subject nuisance variance remains a plausible limitation
- but the first explicit factorization attempt did not yet validate a clean solution

## 9. What the Repository Already Believes vs. What Still Needs External Research

### 9.1 Strong internal conclusions

These conclusions are already strongly supported by repo evidence:

- the final best result is a **frozen compact+legacy fusion**, not a single-model win
- shortlist recall is **not** the main remaining failure mode
- explicit decision models were **not** robustly solved in the validated runs
- single-subject scale is a serious bottleneck
- higher-capacity targets or backbones alone do not automatically fix retrieval

### 9.2 Open questions that require external comparison or deeper synthesis

These are the right places for Gemini Deep Research to add value:

- Why do MindEye-style systems reach 90%+ while this project stalls at 77.2%?
- Which specific ingredients matter most: multi-subject scale, loss mix, backbone choice, subject alignment, data repetition handling, or evaluation protocol?
- Which remedies are highest-ROI for this exact codebase?
- Can the final fixed compact+legacy complementarity be distilled into a single student, or is the compact representation itself fundamentally too weak?
- Would a better query-aware shortlist resolver plausibly close most of the remaining 16 pp, or is the larger gain more likely to come from representation learning and data scale?

## 10. Questions Gemini Should Answer

Gemini should answer the following as explicitly as possible.

1. Compared with MindEye and MindEye2, which missing ingredients in this repository are most likely responsible for the gap from **77.2%** to **90%+**?
2. Of the bottlenecks suggested by the repository, which are most strongly supported by external literature?
3. How much improvement is realistically attributable to:
   - multi-subject pretraining
   - better subject alignment
   - a stronger compact retrieval representation
   - a more robust shortlist resolver
   - different target spaces
   - different losses or training curricula
4. Does the final system's dependence on the legacy expert suggest that the current compact head is underfit, mis-specified, or both?
5. Is the large compact raw vs compact CSLS gap a sign that hubness remains a first-order blocker even at the final stage?
6. What is the most credible path to a **single deployable model** that preserves or exceeds the current frozen fusion result?
7. What is the most credible path specifically toward **90%+ R@1**, and which parts would likely require new data rather than just new architecture?

## 11. Recommended Research Angles for Gemini

Gemini should prioritize the following lines of investigation.

### 11.1 Literature comparison

Compare this repository's final system against:

- MindEye
- MindEye2
- Brain Diffuser
- any NSD retrieval-first methods with strong SHARED1000 performance

For each, extract:

- training data scale
- subject count
- target space
- loss composition
- whether they use unified training or post-hoc fusion
- how they handle repeated presentations
- whether they rely on explicit retrieval correction such as CSLS

### 11.2 Failure-mode mapping

Map each repository bottleneck to the literature:

- shortlist ranking vs shortlist recall
- hubness
- subject variability
- data scale
- regression-vs-contrastive balance
- train/eval mismatch in rerankers

### 11.3 Intervention ranking

Rank potential remedies by:

- expected R@1 gain
- engineering cost in this repo
- scientific plausibility
- compatibility with the current frozen best-system findings

## 12. Proposed Remedy Space to Evaluate

These are not validated fixes. They are candidate directions that Gemini should assess critically.

### 12.1 Highest-priority candidate remedies

- **True multi-subject pretraining or pooled training** that is actually validated end-to-end
- **A stronger compact retrieval student** so the system does not depend on external legacy fusion
- **A leakage-safe shortlist resolver** trained only on true out-of-fold expert predictions
- **Representation changes that reduce hubness intrinsically**, not just through CSLS at evaluation time
- **Better subject alignment or nuisance suppression** if literature supports it more strongly than the failed SCFR attempt

### 12.2 Candidate remedies to be treated cautiously

- simply increasing image-backbone dimensionality
- more complex learned fusion without strong train/val/test hygiene
- making the rerank branch more central without evidence that rerank replacement transfers
- claiming that diffusion-side improvements will fix retrieval

## 13. Constraints and Non-Claims

Gemini should respect the following repository-grounded constraints:

- Do **not** describe the final system as a single end-to-end trained model.
- Do **not** claim that V39/V40 solved reranking; they did not.
- Do **not** claim that SCFR improved retrieval; the repo describes it as a negative scientific result.
- Do **not** assume that newer roadmap configs beyond the frozen final system are validated just because config files exist.
- Do **not** treat qualitative diffusion exports as evidence for retrieval gains.

## 14. High-Value Files for Close Reading

If Gemini can reason over provided excerpts or manually uploaded content, prioritize these:

- `docs/EXPERIMENT_CONTEXT.md`
- `docs/thesis/results/final_outputs/best_system/final_metrics_summary.json`
- `docs/thesis/results/final_outputs/best_system/README.md`
- `configs/experiments/V35_legacy_teacher_distill.yaml`
- `configs/experiments/N1v28a_dual_head.yaml`
- `scripts/evaluation/final_best_system.py`

Then use these for query-level failure inspection:

- `docs/thesis/results/final_outputs/best_system/per_query_shared1000_predictions.csv`
- `docs/thesis/results/final_outputs/best_system/per_query_val_predictions.csv`

Useful note on the per-query files:

- they include the compact, legacy, rerank, and fused top-1 predictions
- they also include ground-truth ranks under each expert and under the frozen fused system
- example columns: `compact_gt_rank`, `legacy_gt_rank`, `rerank_gt_rank`, `fused_gt_rank`

## 15. Ready-to-Paste Prompt for Gemini Deep Research

```text
I need a deep technical research analysis of the best-performing system in my fMRI-to-image decoding repository.

Please treat the following as the canonical fact pattern:

1. The best reportable result is NOT a single trained checkpoint. It is a frozen inference-time retrieval system that combines:
   - a V35 compact/triple-head pipeline
   - a legacy N1v28a dual-head retrieval expert

2. The frozen fusion recipe is:
   - compact_score = csls
   - legacy_score = csls
   - family = normalized_weighted
   - normalization = zscore
   - shortlist_k = 150
   - alpha = 0.3
   - beta = 0.0
   - gamma = 0.7

3. Final exported metrics are:
   - Validation R@1 = 77.6%
   - SHARED1000 R@1 = 77.2%

4. Comparator target from the repository:
   - MindEye = 93.2% R@1 on NSD

5. Important repository evidence:
   - the 77.2% ceiling is explicitly described as NOT a shortlist recall failure
   - union-oracle shortlist recall is essentially saturated (~100% by K=50)
   - compact and legacy top-1 predictions disagree on 55.1% of SHARED1000 queries
   - learned gating/reranking attempts did not robustly beat fixed fusion
   - the repository repeatedly identifies single-subject data scale as a likely core bottleneck
   - higher-capacity targets/backbones alone did not solve the problem

6. Important nuance:
   - the final best system is effectively compact + legacy fusion, because beta = 0.0 and the rerank branch has zero weight in the best frozen setting

Your task:

- explain the most likely reasons this system stopped at 77.2% instead of 90%+
- compare it against MindEye / MindEye2 and related NSD systems
- distinguish evidence-backed conclusions from hypotheses
- rank the most credible remedies by expected gain and implementation realism
- say which changes likely require more data versus which are plausibly achievable through architecture/training changes alone
- tell me what the highest-ROI next step would be if I wanted to turn this codebase into a 90%+ system

Please be very explicit about:
- subject-count / data-scale effects
- target-space choice
- hubness and ranking geometry
- shortlist-decision errors vs shortlist-recall errors
- why the legacy expert is still necessary
- whether a single unified student seems realistic from the current evidence
```

## 16. Bottom Line

The repository's strongest result is a **77.2% SHARED1000 R@1 frozen compact+legacy fusion system**. The key research question is no longer "can the shortlist find the correct image?" but rather:

**Why does the system still fail to rank the correct image first once the right candidates are already present, and which combination of data scale, representation learning, and shortlist decision modeling is needed to close the remaining 16-point gap to 90%+?**
