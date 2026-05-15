# Neural Manifold Alignment for Explainable fMRI-to-Image Decoding

**Subtitle:** Geometry, hubness, uncertainty, reliability, and semantic interpretability beyond top-1 retrieval.

Final semantic manifold archive SHA256:

`1c13d11a6ac0b6d6e3bc9d7c35b807e948be500f1f96bfd67477004a46e16359`

## Executive Summary

This project decodes NSD visual-stimulus-related fMRI activity into CLIP ViT-L/14 768-D embedding space. It evaluates whether decoded embeddings retrieve the correct image and whether they preserve geometry, local neighborhoods, semantic concept structure, reliability signals, and interpretable CLIP-language readouts.

This is not mind reading, dream decoding, or consciousness decoding. It is an analysis of measured fMRI responses to observed visual stimuli, decoded into a multimodal CLIP embedding space.

The central thesis is: **R@1 is not enough.** A neural decoder should also be evaluated by whether it preserves semantic manifold structure, reduces high-dimensional hubness, provides useful reliability signals, and yields interpretable semantic readouts.

The final semantic reports show that V62a preserves image identity and CLIP-language structure:

- Validation final semantic report: `20260515_V62a_val_full_semantic`
- Official SHARED1000 final semantic report: `20260515_V62a_shared1000_full_semantic`

## Key Scientific Findings

1. The decoder preserves not only image identity but also CLIP-language semantics. Validation semantic probe agreement@5 is `0.8989`; SHARED1000 agreement@5 is `0.8330`.
2. CLIP Text Probe shows strong agreement between decoded and target concept distributions. Text-score Spearman is `0.5706` on validation and `0.4614` on SHARED1000.
3. Semantic Axes show interpretable concept dimensions are preserved. Mean axis Spearman is `0.7643` on validation and `0.6730` on SHARED1000, with all 10 axes built.
4. SHARED1000 is harder than validation but still preserves meaningful language-space and axis-level structure.
5. CSLS improves retrieval and reduces hubness. On SHARED1000, CSLS improves R@1 from `0.399` to `0.483` and improves hubness Gini from `0.3861` to `0.2265`.
6. Counterfactual latent edits reveal semantic robustness margins in CLIP space. Robustness margin means are `0.5167` on validation and `0.5083` on SHARED1000.
7. Interpolation paths are smooth in CLIP text-probe distribution space. Mean smoothness is `0.9998768` on validation and `0.9998487` on SHARED1000.
8. Remaining unavailable claims are input-dependent: repeat stability needs per-trial predictions; ROI-axis mapping needs ROI-masked inference; SHARED1000 kappa uncertainty is unavailable because kappas are missing.

## Scientific Motivation

The decoder maps fMRI activity evoked by visual stimuli to CLIP image embeddings. Retrieval tests whether the predicted vector is close to the correct image vector, but retrieval alone can miss important scientific structure.

CLIP is a multimodal semantic manifold: images and text are embedded in a shared space where semantic relationships can be probed through image neighbors, text concepts, and text-defined directions. This makes the decoded embeddings useful as scientific objects, not just as retrieval keys.

The framework evaluates:

- representational similarity analysis (RSA), using RDMs from `1 - cosine_similarity(normalized_embeddings)`
- local neighborhood preservation with random baselines and lift-over-random
- high-dimensional retrieval hubness and CSLS correction
- uncertainty and reliability through kappa, margins, density, and selective decoding
- language-space interpretability through CLIP text probes
- semantic axis preservation through text-defined CLIP directions
- counterfactual latent edits and spherical interpolation in decoded CLIP space

## Framework Architecture

| Module | Status | Evidence | Output files | Notes |
|---|---|---|---|---|
| Retrieval evaluation | Implemented | Validation and SHARED1000 final reports | `retrieval_metrics.json`, retrieval figures | Cosine and CSLS metrics, 1-indexed user-facing ranks. |
| Hubness analysis | Implemented | CSLS reduces Gini/skewness/max hubs | `hubness_metrics.json`, hubness figures | Supports C02. |
| RSA | Implemented and hardened | Validation rho `0.5832`, SHARED1000 rho `0.3890` | `rsa_metrics.json`, RDM figures | Includes RDM error metrics and CIs where available. |
| Neighborhood preservation | Implemented and hardened | Validation lift@10 `514.22x`, SHARED1000 lift@10 `39.98x` | `neighborhood_metrics.json`, figures | Supports C04. |
| Kappa uncertainty | Implemented | Validation AUROC `0.5727` | `uncertainty_metrics.json` | Weak standalone signal; SHARED1000 kappas missing. |
| Manifold density | Implemented | C07 not supported | `manifold_density_metrics.json` | Density alone does not separate correct/incorrect. |
| Reliability-aware selective decoding | Implemented | Validation 20% coverage accuracy `0.9722` | `reliability_metrics.json` | Supported on validation, not on SHARED1000. |
| NMAS | Implemented | Validation `0.5307`, SHARED1000 `0.4119` | `nmas_metrics.json` | Heuristic aggregate score. |
| Semantic error vector field | Implemented | C11 supported in final reports | `vector_field_metrics.json` | Needs future null baseline for stronger inference. |
| CLIP Text Probe | Implemented and run | C09 supported in both final reports | `semantic_probe_metrics.json`, CSVs, figures | Interprets decoded embeddings relative to CLIP language concepts. |
| Semantic Axes | Implemented and run | C10 supported in both final reports | `semantic_axes_metrics.json`, CSVs, figures | 10 axes built in both final reports. |
| Counterfactual latent editing | Implemented and run | C12 supported in both final reports | `counterfactual_metrics.json`, cases, CSV, figures | Latent-space edits only; not brain manipulation. |
| Spherical interpolation | Implemented and run | C13 supported in both final reports | `interpolation_metrics.json`, cases, CSV, figures | Latent-space trajectories only; not real neural trajectories. |
| Claim testing | Implemented and hardened | C01-C14 evaluated | `claim_tests.json` | Split-specific statuses. |
| Report generation | Implemented and hardened | Final semantic reports transferred locally | `summary.md`, metrics, figures | Diagnostic reports are explicitly excluded. |
| Repeat stability | Implemented path, unavailable | C05 unavailable | repeat-stability outputs | Requires per-trial predictions. |
| ROI axes | Planned/advanced | C14 unavailable | ROI-axis outputs | Requires ROI masks and masked inference/export. |

## Final Report Table

| Report | Status | Protocol | N | Gallery | Dim | CSLS R@1 | RSA Spearman | C09 | C10 | C12 | C13 | Claim counts |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| `20260515_V62a_val_full_semantic` | valid final semantic report | validation, full 10k gallery | 900 | 10000 | 768 | 0.9189 | 0.5832 | supported | supported | supported | supported | 9 supported, 1 partial, 2 not supported, 2 unavailable |
| `20260515_V62a_shared1000_full_semantic` | valid final semantic report | official SHARED1000 target-embeddings 1k gallery | 1000 | 1000 | 768 | 0.4830 | 0.3890 | supported | supported | supported | supported | 8 supported, 1 partial, 2 not supported, 3 unavailable |
| `20260514_V62a_shared1000_subj01_hardened_with_ids` | valid different benchmark | SHARED1000 queries over full 10k gallery | 1000 | 10000 | 768 | not extracted locally | not extracted locally | unavailable | unavailable | unavailable | unavailable | 3 supported, 2 partial, 7 unavailable in cluster log |
| `20260514_V61a_shared1000_mctta_token_full` | valid token-space report | 197376-D token-space retrieval | 1000 | 10000 inferred | 197376 | 0.1650 | 0.1658 | unavailable | unavailable | unavailable | unavailable | not a 768-D CLIP claim table |

## V62a Validation Final Semantic Report

Report: `20260515_V62a_val_full_semantic`

This is a valid 768-D CLIP manifold report over a full 10k gallery. It supersedes the pre-semantic hardened validation report for C09-C13 conclusions.

### Retrieval

| Metric | Cosine | CSLS |
|---|---:|---:|
| R@1 | 0.9177777777777778 | 0.9188888888888889 |
| R@5 | 0.9344444444444444 | 0.9444444444444444 |
| R@10 | 0.9466666666666667 | 0.9477777777777778 |
| Median rank | 1.0 | 1.0 |
| Mean rank | 11.407777777777778 | 7.9 |
| MRR | 0.9260650745486166 | 0.928962536964556 |

CSLS improves R@1 by `+0.0011111111111110628`, R@5 by `+0.010000000000000009`, and mean rank by `-3.507777777777777`.

### Geometry, Neighborhoods, And Hubness

| Metric | Value |
|---|---:|
| RSA Spearman | 0.5831830506873594 |
| RSA Pearson | 0.6187697649002075 |
| Mean absolute RDM error | 0.5711466670036316 |
| Median absolute RDM error | 0.5743107795715332 |
| 95th percentile absolute RDM error | 0.6855753242969512 |
| NeighborhoodOverlap@10 | 0.5142222222222221 |
| Random baseline@10 | 0.001 |
| Lift@10 | 514.2222222222222x |
| Trustworthiness | 0.9512245461968469 |
| Continuity | 0.9588342440801457 |
| Hubness Gini cosine -> CSLS | 0.6001102666666667 -> 0.5173008 |
| Hubness skewness cosine -> CSLS | 1.4734349264567659 -> 0.8937531369213694 |
| Max hub cosine -> CSLS | 9 -> 6 |
| NMAS | 0.5306517203072261 |

### Semantic Probe, Axes, Counterfactuals, And Interpolation

| Metric | Value |
|---|---:|
| C09 agreement@1 | 0.5066666666666667 |
| C09 agreement@5 | 0.8988888888888888 |
| C09 agreement@10 | 0.9766666666666667 |
| C09 score-vector Spearman | 0.570574712643678 |
| C09 score-vector Pearson | 0.6635606211258306 |
| C09 JS divergence mean | 0.00908694788813591 |
| C10 axes built | 10 |
| C10 mean axis Spearman | 0.7642887032370822 |
| C10 mean axis Pearson | 0.8118015170097351 |
| C10 mean preservation | 0.8384965181892845 |
| C10 mean axis MAE | 0.037282809242606164 |
| C10 best axis | `indoor_vs_outdoor` |
| C10 weakest axis | `building_vs_natural_landscape` |
| C12 robustness margin mean | 0.5166666666666667 |
| C12 robustness margin median | 0.5 |
| C12 transition rate | 1.0 |
| C12 transitions observed | 240 |
| C13 mean smoothness | 0.9998768267270733 |
| C13 abrupt transition rate | 0.0175 |
| C13 mean semantic velocity | 0.00010009621973949834 |
| C13 top-concept transition count | 22 |

## V62a SHARED1000 Final Semantic Report

Report: `20260515_V62a_shared1000_full_semantic`

This is the valid official SHARED1000 target-gallery reproduction with full semantic modules. It uses `shared1000_ground_truth.npy` as the 1000-candidate gallery and `target_indices=np.arange(1000)`. It exactly reproduces `shared1000_metrics.json`.

### Retrieval

| Metric | Cosine | CSLS |
|---|---:|---:|
| R@1 | 0.399 | 0.483 |
| R@5 | 0.732 | 0.802 |
| R@10 | 0.837 | 0.897 |
| R@20 | 0.922 | 0.958 |
| R@50 | 0.972 | 0.989 |
| R@100 | 0.989 | 0.996 |
| Median rank | 2.0 | 2.0 |
| Mean rank | 8.33 | 5.025 |
| MRR | 0.5452108744517535 | 0.6207708890365495 |

CSLS improves R@1 by `+0.084`, R@5 by `+0.070`, R@10 by `+0.060`, and MRR by `+0.075560014584796`.

### Geometry, Neighborhoods, And Hubness

| Metric | Value |
|---|---:|
| RSA Spearman | 0.3890413340912553 |
| RSA Spearman CI | [0.38664180575607254, 0.39178909364578646] |
| RSA Pearson | 0.4608447253704071 |
| Mean absolute RDM error | 0.5272711515426636 |
| Median absolute RDM error | 0.5331650376319885 |
| 95th percentile absolute RDM error | 0.6823657304048538 |
| NeighborhoodOverlap@10 | 0.39980000000000004 |
| Random baseline@10 | 0.01 |
| Lift@10 | 39.980000000000004x |
| Trustworthiness | 0.8862625698324023 |
| Continuity | 0.8923893346876587 |
| Hubness Gini cosine -> CSLS | 0.3860652 -> 0.2264522 |
| Hubness skewness cosine -> CSLS | 1.1786611866791903 -> 0.5973609821229598 |
| Max hub cosine -> CSLS | 46 -> 26 |
| NMAS | 0.41187460775327206 |

### Semantic Probe, Axes, Counterfactuals, And Interpolation

| Metric | Value |
|---|---:|
| C09 agreement@1 | 0.454 |
| C09 agreement@5 | 0.833 |
| C09 agreement@10 | 0.952 |
| C09 score-vector Spearman | 0.46142561576354674 |
| C09 score-vector Pearson | 0.5657484461818821 |
| C09 JS divergence mean | 0.0137467160820961 |
| C10 axes built | 10 |
| C10 mean axis Spearman | 0.6730118350118351 |
| C10 mean axis Pearson | 0.7322260797023773 |
| C10 mean preservation | 0.7968396524523687 |
| C10 mean axis MAE | 0.04696349613368511 |
| C10 best axis | `indoor_vs_outdoor` |
| C10 weakest axis | `water_vs_land` |
| C12 robustness margin mean | 0.5083333333333333 |
| C12 robustness margin median | 0.5 |
| C12 transition rate | 1.0 |
| C12 transitions observed | 240 |
| C13 mean smoothness | 0.9998487128650547 |
| C13 abrupt transition rate | 0.0075000000000000015 |
| C13 mean semantic velocity | 0.00012371759785310133 |
| C13 top-concept transition count | 26 |

## SHARED1000 Protocol Finding

The official saved SHARED1000 benchmark is reproduced exactly by:

```python
gallery = shared1000_ground_truth.npy
target_indices = np.arange(1000)
```

It is not reproduced by filtering `clip.parquet` by `nsdId`, and it is not the full 10k gallery protocol.

The valid official reports are:

- pre-semantic benchmark: `20260514_V62a_shared1000_target_embeddings_hardened`
- final semantic report: `20260515_V62a_shared1000_full_semantic`

The valid but different protocol is:

- `20260514_V62a_shared1000_subj01_hardened_with_ids`: 1000 shared queries over a full 10k gallery

The excluded diagnostic reports are:

- `20260514_V62a_shared1000_subj01_hardened`: no target IDs, diagonal fallback
- `20260514_V62a_shared1000_1k_gallery_hardened`: misconfigured/wrong 1k attempt, loaded a 10k-style gallery and did not reproduce `shared1000_metrics.json`

## V61a Token-Space Status

The V61a MC-TTA token-space reports are valid as token-space/rank/hubness reports, but they are 197376-D token-space analyses, not 768-D CLIP manifold reports.

Visible metrics:

| Metric | Cosine | CSLS |
|---|---:|---:|
| R@1 | 0.092 | 0.165 |
| R@5 | 0.374 | 0.581 |
| R@10 | 0.586 | 0.752 |
| Median rank | 7 | 3 |
| MRR | 0.23534784790856872 | 0.3457259237133666 |
| Hubness Gini | 0.6797156 | 0.521355 |
| Hubness skewness | 2.4042223857330307 | 1.1212628988777504 |

RSA Spearman is `0.16584571769615267`. This supports CSLS ranking and hubness benefits in token-space, but it must not be mixed with V62a 768-D CLIP conclusions.

Fusion remains rank-level unless a true fused 768-D embedding export is created. Fusion RSA, neighborhood preservation, semantic probes, and semantic axes are unavailable without that export.

## Scientific Claim Table

| Claim ID | Claim | Current status | Evidence | Limitations | Next action |
|---|---|---|---|---|---|
| C01 | CSLS improves retrieval compared to cosine. | Supported overall; partial on easy validation. | Validation R@1 gain `+0.0011`; SHARED1000 R@1 gain `+0.0840`; token-space R@1 gain `+0.0730`. | Effect size is protocol-dependent and near ceiling on validation. | Report split-specific gains. |
| C02 | CSLS reduces hubness. | Supported. | Validation Gini `0.6001 -> 0.5173`; SHARED1000 Gini `0.3861 -> 0.2265`; token-space Gini `0.6797 -> 0.5214`. | Hubness severity depends on gallery/protocol. | Keep Gini, skewness, and max-hub metrics together. |
| C03 | Decoded fMRI embeddings preserve representational geometry. | Supported on validation; partial on SHARED1000. | RSA Spearman `0.5832` validation, `0.3890` SHARED1000. | Harder split weakens geometry preservation. | Add confidence intervals and null baselines where possible. |
| C04 | Decoded embeddings preserve local semantic neighborhoods. | Supported. | Validation Overlap@10 `0.5142` with `514.22x` lift; SHARED1000 Overlap@10 `0.3998` with `39.98x` lift. | Lift depends on gallery size. | Always report observed overlap with random baseline. |
| C05 | Repeated presentations produce stable decoded embeddings. | Unavailable. | No per-trial prediction export in final reports. | Image-averaged predictions are insufficient. | Export per-trial `z_pred` and rerun repeat stability. |
| C06 | vMF kappa provides directional uncertainty. | Not supported on validation; unavailable on SHARED1000. | Validation AUROC `0.5727`; SHARED1000 kappas missing. | Kappa alone is weak; no SHARED1000 kappa artifact. | Use kappa only as part of composite reliability; export shared kappas if possible. |
| C07 | Reliable predictions tend to be on-manifold/high-margin/low-entropy. | Not supported for density alone. | Final claim tests reject density separation. | Simple KNN density is not enough. | Test composite reliability features. |
| C08 | Reliability-aware selective decoding improves accuracy at lower coverage. | Supported on validation; not supported on SHARED1000. | Validation selective accuracy at 20% coverage `0.9722`; SHARED1000 final C08 not supported. | Does not transfer cleanly across protocols. | Improve reliability scoring and keep coverage curves split-specific. |
| C09 | CLIP text probes interpret decoded brain embeddings. | Supported. | Agreement@5 `0.8989` validation, `0.8330` SHARED1000; score-vector Spearman `0.5706` and `0.4614`. | CLIP-language probe, not literal thought decoding. | Expand concept library and report calibration. |
| C10 | Semantic axes reveal structured concept preservation/distortion. | Supported. | 10 axes built; mean axis Spearman `0.7643` validation, `0.6730` SHARED1000. | Axis validity depends on CLIP text concept quality. | Add axis-specific qualitative examples. |
| C11 | Semantic error vectors are structured rather than random. | Supported. | Final claim tests support structured error vectors in both reports. | Needs stronger permutation/null baseline. | Add null vector-field tests. |
| C12 | Counterfactual latent edits reveal semantic robustness margins. | Supported. | Robustness margin mean `0.5167` validation, `0.5083` SHARED1000; transition rate `1.0`. | Latent-space edits only; not neural manipulation or causal control. | Curate qualitative grids and axis-specific margins. |
| C13 | Decoded embeddings form continuous semantic trajectories under interpolation. | Supported. | Mean smoothness `0.9998768` validation, `0.9998487` SHARED1000; abrupt transition rates `0.0175` and `0.0075`. | Latent-space path, not a real neural trajectory. | Compare same-concept and different-concept paths in figures. |
| C14 | ROI ablations map cortical regions to semantic axes. | Unavailable. | No ROI-masked inference/export in final reports. | Requires ROI masks and rerunnable inference. | Audit ROI assets and implement ROI-masked prediction export. |

## What Is Supported

The final reports support the core thesis: V62a preserves retrieval identity, representational geometry, local neighborhood structure, CLIP-language concept distributions, and semantic axis structure.

CSLS is especially important on harder protocols. It is marginal on easy validation because retrieval is already near ceiling, but it substantially improves SHARED1000 ranking and reduces hubness.

The semantic probe and semantic axes move the project beyond R@1: decoded embeddings can be compared to CLIP language concepts, and interpretable concept directions are preserved even on the harder SHARED1000 benchmark.

Counterfactual latent editing and spherical interpolation provide additional CLIP-space explanations: robustness margins are measurable, and interpolation paths are smooth in text-probe distribution space.

## What Is Weak Or Not Supported

Kappa alone is weak as standalone uncertainty on validation and unavailable on SHARED1000.

Simple manifold density alone does not separate correct from incorrect predictions reliably.

Reliability-aware selective decoding is promising on validation, but the final SHARED1000 claim test does not support it.

Token-space V61a results are useful evidence for CSLS ranking and hubness behavior, but they are not 768-D CLIP manifold evidence.

## What Is Still Missing

- repeat stability from per-trial predictions
- SHARED1000 kappa uncertainty, because `shared1000_kappas.npy` is missing
- ROI-to-semantic-axis mapping through ROI masks and ROI-masked inference/export
- true fused 768-D embeddings for fusion manifold analysis
- stronger null/permutation baselines for semantic error vector structure and counterfactual transitions

## Thesis/Paper Narrative

We built a high-performing fMRI-to-CLIP decoder and then used it as a scientific instrument to analyze the geometry, hubness, reliability, and semantic interpretability of decoded brain representations in CLIP space.

The strongest narrative is not that the model reads minds. It does not. The scientific contribution is that decoded visual-stimulus-related fMRI embeddings can be evaluated as points on a multimodal semantic manifold. This enables retrieval, RSA, neighborhood preservation, hubness correction, language-probe interpretation, semantic-axis analysis, counterfactual latent editing, and interpolation analysis within one coherent framework.

## Scientific Honesty

- This is visual-stimulus-related fMRI-to-CLIP analysis.
- It is not mind reading.
- It is not dream decoding.
- It is not consciousness decoding.
- Counterfactual edits do not manipulate brain activity.
- Interpolations are latent-space paths, not real neural trajectories.
- Modules without required artifacts are marked unavailable.
- Negative findings such as weak kappa and weak density separation are scientifically useful.

## Best Figures For Thesis Or Presentation

| Purpose | Figure |
|---|---|
| Core retrieval comparison | `20260515_V62a_val_full_semantic/figures/figure_retrieval_comparison.png` |
| Hard benchmark retrieval comparison | `20260515_V62a_shared1000_full_semantic/figures/figure_retrieval_comparison.png` |
| Neighborhood preservation | `20260515_V62a_val_full_semantic/figures/figure_neighborhood_overlap_vs_k.png` |
| Hubness reduction | `20260515_V62a_shared1000_full_semantic/figures/figure_hubness_lorenz.png` |
| Text-probe headline | `20260515_V62a_val_full_semantic/figures/figure_semantic_probe_agreement_bar.png` |
| Text-probe score alignment | `20260515_V62a_shared1000_full_semantic/figures/figure_semantic_probe_score_correlation.png` |
| Semantic axes headline | `20260515_V62a_val_full_semantic/figures/figure_semantic_axes_preservation_bar.png` |
| Semantic axis projections | `20260515_V62a_shared1000_full_semantic/figures/figure_semantic_axes_projection_scatter.png` |
| Counterfactual margins | `20260515_V62a_val_full_semantic/figures/figure_counterfactual_robustness_margin.png` |
| Counterfactual axis sensitivity | `20260515_V62a_shared1000_full_semantic/figures/figure_counterfactual_axis_sensitivity.png` |
| Interpolation smoothness | `20260515_V62a_val_full_semantic/figures/figure_interpolation_semantic_velocity.png` |
| Interpolation concepts | `20260515_V62a_shared1000_full_semantic/figures/figure_interpolation_top_concept_paths.png` |
| Aggregate manifold score | `20260515_V62a_val_full_semantic/figures/figure_nmas_radar.png` |

## Reproducibility

Inspect final metrics:

```bash
python3 - <<'PY'
import json
from pathlib import Path

for report in [
    Path("20260515_V62a_val_full_semantic"),
    Path("20260515_V62a_shared1000_full_semantic"),
]:
    print("\\n==", report, "==")
    for name in [
        "retrieval_metrics.json",
        "rsa_metrics.json",
        "neighborhood_metrics.json",
        "hubness_metrics.json",
        "semantic_probe_metrics.json",
        "semantic_axes_metrics.json",
        "counterfactual_metrics.json",
        "interpolation_metrics.json",
        "nmas_metrics.json",
        "claim_tests.json",
    ]:
        path = report / name
        print(name, "exists=", path.exists())
        if path.exists():
            data = json.loads(path.read_text())
            print(json.dumps(data, indent=2)[:1200])
PY
```

Rerun V62a validation hardened baseline:

```bash
PYTHONPATH=/home/jovyan/work/FMRI2images/src \
python3 scripts/run_manifold_analysis.py \
  --config /tmp/manifold_V62a_val.yaml \
  --output-dir reports/manifold_analysis/20260514_V62a_val_subj01_hardened
```

Rerun official SHARED1000 target-embeddings benchmark:

```bash
PYTHONPATH=/home/jovyan/work/FMRI2images/src \
python3 scripts/run_manifold_analysis.py \
  --config configs/manifold_analysis_V62a_shared1000_target_embeddings.template.yaml \
  --output-dir reports/manifold_analysis/20260514_V62a_shared1000_target_embeddings_hardened
```

Rerun full semantic reports after the CLIP text embedding cache exists:

```bash
PYTHONPATH=/home/jovyan/work/FMRI2images/src \
python3 scripts/run_manifold_analysis.py \
  --config configs/manifold_analysis.yaml \
  --output-dir reports/manifold_analysis/20260515_V62a_val_full_semantic
```

Use the exact cluster configs for the transferred final reports when regenerating publication artifacts.
