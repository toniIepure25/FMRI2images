# Neural Manifold Alignment Comparison Summary

Updated: 2026-05-15.

Final semantic manifold archive SHA256:

`1c13d11a6ac0b6d6e3bc9d7c35b807e948be500f1f96bfd67477004a46e16359`

This summary separates validation, official SHARED1000 target-gallery, full-10k SHARED1000, and V61a token-space protocols. The scientific framing is visual-stimulus-related fMRI-to-CLIP analysis. It is not mind reading, dream decoding, or consciousness decoding.

## Final Comparison Table

| report_dir | status | model | protocol | N | prediction_dim | gallery_size | cosine_R@1 | csls_R@1 | cosine_R@5 | csls_R@5 | cosine_R@10 | csls_R@10 | median_rank_cosine | median_rank_csls | MRR_cosine | MRR_csls | CSLS_R@1_improvement | RSA_Spearman | NeighborhoodOverlap@10 | NeighborhoodLift@10 | Hubness_Gini_cosine | Hubness_Gini_CSLS | Kappa_AUROC | SelectiveAcc20 | NMAS | C09 | C10 | C12 | C13 | claim_counts | caveats |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| `20260515_V62a_val_full_semantic` | valid final semantic report | V62a CLS retrieval, subj01 | validation, full 10k gallery | 900 | 768 | 10000 | 0.9178 | 0.9189 | 0.9344 | 0.9444 | 0.9467 | 0.9478 | 1.0 | 1.0 | 0.9261 | 0.9290 | +0.0011 | 0.5832 | 0.5142 | 514.22x | 0.6001 | 0.5173 | 0.5727 | 0.9722 | 0.5307 | supported | supported | supported | supported | 9 supported, 1 partial, 2 not supported, 2 unavailable | Strongest/easiest protocol; CSLS R@1 near ceiling, but semantic probe/axes/counterfactual/interpolation all support CLIP-space interpretability. |
| `20260515_V62a_shared1000_full_semantic` | valid final semantic report | V62a CLS retrieval, subj01 | official SHARED1000 target-embeddings 1k gallery | 1000 | 768 | 1000 | 0.3990 | 0.4830 | 0.7320 | 0.8020 | 0.8370 | 0.8970 | 2.0 | 2.0 | 0.5452 | 0.6208 | +0.0840 | 0.3890 | 0.3998 | 39.98x | 0.3861 | 0.2265 | unavailable | not supported | 0.4119 | supported | supported | supported | supported | 8 supported, 1 partial, 2 not supported, 3 unavailable | Exactly reproduces `shared1000_metrics.json` using `shared1000_ground_truth.npy` as the 1000-image gallery. Kappas are missing, so C06 is unavailable. |
| `20260514_V62a_shared1000_subj01_hardened_with_ids` | valid different benchmark | V62a CLS retrieval, subj01 | SHARED1000 queries over full 10k gallery | 1000 | 768 | 10000 | not extracted locally | not extracted locally | not extracted locally | not extracted locally | not extracted locally | not extracted locally | not extracted locally | not extracted locally | not extracted locally | not extracted locally | not extracted locally | not extracted locally | not extracted locally | not extracted locally | not extracted locally | not extracted locally | unavailable | not extracted locally | not extracted locally | unavailable | unavailable | unavailable | unavailable | 3 supported, 2 partial, 7 unavailable in cluster log | Valid target-ID mapping, but not comparable to `shared1000_metrics.json`; harder 1000-query over 10k-gallery protocol. |
| `20260514_V61a_shared1000_mctta_token_full` | valid token-space/rank/hubness report | V61a MC-TTA token-space | SHARED1000-like token-space full gallery | 1000 | 197376 | 10000 inferred | 0.0920 | 0.1650 | 0.3740 | 0.5810 | 0.5860 | 0.7520 | 7 | 3 | 0.2353 | 0.3457 | +0.0730 | 0.1658 | unavailable locally | unavailable locally | 0.6797 | 0.5214 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | no final C01-C14 table locally | Valid token-space analysis, not a 768-D CLIP manifold report; do not mix with V62a 768-D claims. |

## Final Semantic Metrics

| Metric | V62a validation full semantic | V62a SHARED1000 full semantic |
|---|---:|---:|
| Semantic probe agreement@1 | 0.5067 | 0.4540 |
| Semantic probe agreement@5 | 0.8989 | 0.8330 |
| Semantic probe agreement@10 | 0.9767 | 0.9520 |
| Text-score Spearman | 0.5706 | 0.4614 |
| Text-score Pearson | 0.6636 | 0.5657 |
| JS divergence mean | 0.0091 | 0.01375 |
| Semantic axes built | 10 | 10 |
| Mean axis Spearman | 0.7643 | 0.6730 |
| Mean axis Pearson | 0.8118 | 0.7322 |
| Mean preservation | 0.8385 | 0.7968 |
| Mean axis MAE | 0.0373 | 0.04696 |
| Best axis | `indoor_vs_outdoor` | `indoor_vs_outdoor` |
| Weakest axis | `building_vs_natural_landscape` | `water_vs_land` |
| Counterfactual robustness margin mean | 0.5167 | 0.5083 |
| Counterfactual transition rate | 1.0 | 1.0 |
| Counterfactual transitions observed | 240 | 240 |
| Interpolation mean smoothness | 0.9998768 | 0.9998487 |
| Interpolation abrupt transition rate | 0.0175 | 0.0075 |
| Interpolation mean semantic velocity | 0.0001001 | 0.0001237 |
| Interpolation top-concept transitions | 22 | 26 |

## Key Scientific Findings

1. The decoder preserves not only image identity but also CLIP-language semantics: final validation has semantic probe agreement@5 `0.8989`, and final SHARED1000 has agreement@5 `0.8330`.
2. CLIP Text Probe shows strong agreement between decoded and target concept distributions: score-vector Spearman is `0.5706` on validation and `0.4614` on SHARED1000.
3. Semantic Axes show interpretable concept dimensions are preserved: mean axis Spearman is `0.7643` on validation and `0.6730` on SHARED1000, with all 10 axes built in both final reports.
4. SHARED1000 is harder than validation but still preserves meaningful language-space and axis-level structure.
5. CSLS improves retrieval and reduces hubness. On SHARED1000, R@1 improves from `0.399` to `0.483`, and hubness Gini improves from `0.3861` to `0.2265`.
6. Counterfactual latent edits reveal semantic robustness margins in CLIP space: robustness margin means are `0.5167` and `0.5083`, with transition rate `1.0` in both final reports.
7. Interpolation paths are smooth in CLIP text-probe distribution space: mean smoothness is above `0.9998` in both reports, with low abrupt transition rates.
8. Remaining unavailable claims are narrow and input-dependent: repeat stability needs per-trial predictions; ROI-axis mapping needs ROI-masked inference; SHARED1000 kappa uncertainty is unavailable because kappas are missing.

## Claim Table

| Claim | Validation final status | SHARED1000 final status | Evidence |
|---|---|---|---|
| C01 CSLS improves retrieval | partially supported | supported | Validation R@1 gain `+0.0011`; SHARED1000 R@1 gain `+0.0840`. |
| C02 CSLS reduces hubness | supported | supported | Validation Gini `0.6001 -> 0.5173`; SHARED1000 Gini `0.3861 -> 0.2265`. |
| C03 Geometry preservation | supported | partially supported | RSA Spearman `0.5832` validation, `0.3890` SHARED1000. |
| C04 Neighborhood preservation | supported | supported | Overlap@10 `0.5142` and `0.3998`, both far above random baselines. |
| C05 Repeat stability | unavailable | unavailable | Requires per-trial predictions, not image-averaged embeddings. |
| C06 Kappa uncertainty | not supported | unavailable | Validation AUROC `0.5727`; SHARED1000 kappas missing. |
| C07 Density/reliability separation | not supported | not supported | Density alone does not separate correct and incorrect predictions. |
| C08 Selective decoding | supported | not supported | Validation selective accuracy at 20% coverage `0.9722`; SHARED1000 final claim test not supported. |
| C09 CLIP text probes | supported | supported | Agreement@5 `0.8989` validation, `0.8330` SHARED1000. |
| C10 Semantic axes | supported | supported | Mean axis Spearman `0.7643` validation, `0.6730` SHARED1000. |
| C11 Structured error vectors | supported | supported | Error-vector structure supported in both final claim tests. |
| C12 Counterfactual robustness margins | supported | supported | Robustness margin means `0.5167` and `0.5083`; transition rate `1.0`. |
| C13 Smooth latent trajectories | supported | supported | Mean smoothness `0.9998768` and `0.9998487`; low abrupt transition rates. |
| C14 ROI-axis mapping | unavailable | unavailable | Requires ROI masks plus ROI-masked inference/export. |

## Diagnostic / Excluded Reports

| report_dir | status | reason |
|---|---|---|
| `20260514_V62a_shared1000_subj01_hardened` | invalid/diagnostic | Ran before `shared1000_nsd_ids.npy` existed and used diagonal fallback without valid target IDs. |
| `20260514_V62a_shared1000_1k_gallery_hardened` | diagnostic/misconfigured | Loaded a 10k-style `clip.parquet` gallery and produced `cosine R@1=0.116`, `CSLS R@1=0.134`; not the official SHARED1000 target-embedding benchmark. |

## Best Thesis Figures

- `20260515_V62a_val_full_semantic/figures/figure_retrieval_comparison.png`
- `20260515_V62a_shared1000_full_semantic/figures/figure_retrieval_comparison.png`
- `20260515_V62a_val_full_semantic/figures/figure_neighborhood_overlap_vs_k.png`
- `20260515_V62a_shared1000_full_semantic/figures/figure_hubness_lorenz.png`
- `20260515_V62a_val_full_semantic/figures/figure_semantic_probe_agreement_bar.png`
- `20260515_V62a_shared1000_full_semantic/figures/figure_semantic_probe_score_correlation.png`
- `20260515_V62a_val_full_semantic/figures/figure_semantic_axes_preservation_bar.png`
- `20260515_V62a_shared1000_full_semantic/figures/figure_semantic_axes_projection_scatter.png`
- `20260515_V62a_val_full_semantic/figures/figure_counterfactual_robustness_margin.png`
- `20260515_V62a_shared1000_full_semantic/figures/figure_counterfactual_axis_sensitivity.png`
- `20260515_V62a_val_full_semantic/figures/figure_interpolation_semantic_velocity.png`
- `20260515_V62a_shared1000_full_semantic/figures/figure_interpolation_top_concept_paths.png`
- `20260515_V62a_val_full_semantic/figures/figure_nmas_radar.png`

## Scientific Caveats

- The evidence is from visual-stimulus-related fMRI responses to observed images.
- The semantic probe and semantic axes interpret decoded CLIP embeddings relative to CLIP language concepts; they do not decode thoughts.
- Counterfactual edits do not manipulate brain activity; they perturb decoded CLIP latent vectors.
- Interpolation paths are latent-space paths, not real neural trajectories.
- V61a token-space metrics are separate from 768-D CLIP manifold metrics.
- Fusion remains rank-level unless true fused 768-D embeddings are exported.
