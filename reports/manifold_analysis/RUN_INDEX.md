# Manifold Analysis Run Index

Updated: 2026-05-15.

Final semantic manifold archive SHA256:

`1c13d11a6ac0b6d6e3bc9d7c35b807e948be500f1f96bfd67477004a46e16359`

This index separates final semantic reports, benchmark-only reports, diagnostic reports, and token-space reports. Do not mix protocols or embedding spaces.

| Report directory | Model/run | Split/protocol | Status | N | Gallery size | Prediction dim | Cosine R@1 | CSLS R@1 | RSA Spearman | C09 | C10 | C12 | C13 | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
| `20260515_V62a_val_full_semantic/` | V62a CLS retrieval, subj01 | validation, full 10k gallery | valid final semantic report | 900 | 10000 | 768 | 0.9178 | 0.9189 | 0.5832 | supported | supported | supported | supported | Final validation report with text probes, semantic axes, counterfactuals, interpolation, kappa, reliability, NMAS, and claim tests. Claim counts: 9 supported, 1 partial, 2 not supported, 2 unavailable. |
| `20260515_V62a_shared1000_full_semantic/` | V62a CLS retrieval, subj01 | official SHARED1000 target-embeddings 1k gallery | valid final semantic report | 1000 | 1000 | 768 | 0.3990 | 0.4830 | 0.3890 | supported | supported | supported | supported | Final SHARED1000 report. It exactly reproduces `shared1000_metrics.json` using `shared1000_ground_truth.npy` as the 1000-image gallery. Claim counts: 8 supported, 1 partial, 2 not supported, 3 unavailable. |
| `20260515_V62a_val_text_probe/` | V62a CLS retrieval, subj01 | validation, text-probe extension | valid semantic probe report | 900 | 10000 | 768 | 0.9178 | 0.9189 | 0.5832 | supported | not included | not included | not included | Intermediate C09 report superseded by `20260515_V62a_val_full_semantic/`. |
| `20260515_V62a_shared1000_text_probe/` | V62a CLS retrieval, subj01 | SHARED1000 target-embeddings, text-probe extension | valid semantic probe report | 1000 | 1000 | 768 | 0.3990 | 0.4830 | 0.3890 | supported | not included | not included | not included | Intermediate C09 report superseded by `20260515_V62a_shared1000_full_semantic/`. |
| `20260514_V62a_val_subj01_hardened/` | V62a CLS retrieval, subj01 | validation, full 10k gallery | valid hardened benchmark report | 900 | 10000 | 768 | 0.9178 | 0.9189 | 0.5832 | unavailable | unavailable | unavailable | unavailable | Valid pre-semantic hardened report. NeighborhoodOverlap@10 `0.5142`, lift@10 `514.22x`. |
| `20260514_V62a_shared1000_target_embeddings_hardened/` | V62a CLS retrieval, subj01 | official SHARED1000 target-embeddings 1k gallery | valid official benchmark reproduction | 1000 | 1000 | 768 | 0.3990 | 0.4830 | 0.3890 | unavailable | unavailable | unavailable | unavailable | Valid pre-semantic report. Cosine R@5/R@10 `0.732/0.837`; CSLS R@5/R@10 `0.802/0.897`; NeighborhoodOverlap@10 `0.3998`; Gini `0.3861 -> 0.2265`; NMAS `0.4119`. |
| `20260514_V62a_shared1000_subj01_hardened_with_ids/` | V62a CLS retrieval, subj01 | SHARED1000 queries over full 10k gallery | valid different benchmark | 1000 | 10000 | 768 | not extracted locally | not extracted locally | not extracted locally | unavailable | unavailable | unavailable | unavailable | Valid target-ID mapping, but not comparable to `shared1000_metrics.json`; harder 1000-query over 10k-gallery protocol. Cluster log: 3 supported, 2 partial, 7 unavailable. |
| `20260514_V62a_shared1000_subj01_hardened/` | V62a CLS retrieval, subj01 | SHARED1000 full 10k gallery attempt | invalid/diagnostic | 1000 | 10000 | 768 | excluded | excluded | excluded | excluded | excluded | excluded | excluded | Ran before `shared1000_nsd_ids.npy` existed and used diagonal fallback. Exclude retrieval, hubness, neighborhood, and claim conclusions. |
| `20260514_V62a_shared1000_1k_gallery_hardened/` | V62a CLS retrieval, subj01 | attempted SHARED1000 target-ID parquet gallery | diagnostic/misconfigured | 1000 | expected 1000, but loaded 10k-style gallery | 768 | 0.1160 | 0.1340 | not indexed | excluded | excluded | excluded | excluded | Did not reproduce `shared1000_metrics.json`; treat as diagnostic unless regenerated after confirmed final gallery shape `(1000, 768)`. |
| `20260514_V61a_shared1000_mctta_token/` | V61a MC-TTA token-space | SHARED1000-like token-space full gallery | valid token-space report | 1000 | 10000 inferred | 197376 | 0.0920 | 0.1650 | 0.1658 | unavailable | unavailable | unavailable | unavailable | Valid 197376-D token-space report. Not a 768-D CLIP manifold report. |
| `20260514_V61a_shared1000_mctta_token_full/` | V61a MC-TTA token-space | SHARED1000-like token-space full gallery | valid token-space report | 1000 | 10000 inferred | 197376 | 0.0920 | 0.1650 | 0.1658 | unavailable | unavailable | unavailable | unavailable | Same visible retrieval/hubness/RSA metrics as token report, with additional figures. Do not mix with V62a 768-D CLIP analysis. |

## Final Full-Semantic Metrics

| Metric | Validation `20260515_V62a_val_full_semantic` | SHARED1000 `20260515_V62a_shared1000_full_semantic` |
|---|---:|---:|
| Cosine R@1/R@5/R@10 | 0.9178 / 0.9344 / 0.9467 | 0.3990 / 0.7320 / 0.8370 |
| CSLS R@1/R@5/R@10 | 0.9189 / 0.9444 / 0.9478 | 0.4830 / 0.8020 / 0.8970 |
| Cosine MRR | 0.9260650745486166 | 0.5452108744517535 |
| CSLS MRR | 0.928962536964556 | 0.6207708890365495 |
| RSA Spearman | 0.5831830506873594 | 0.3890413340912553 |
| RSA Pearson | 0.6187697649002075 | 0.4608447253704071 |
| NeighborhoodOverlap@10 | 0.5142222222222221 | 0.39980000000000004 |
| Random baseline@10 | 0.001 | 0.01 |
| Lift@10 | 514.2222222222222x | 39.980000000000004x |
| Hubness Gini cosine -> CSLS | 0.6001102666666667 -> 0.5173008 | 0.3860652 -> 0.2264522 |
| Semantic probe agreement@1/@5/@10 | 0.5067 / 0.8989 / 0.9767 | 0.4540 / 0.8330 / 0.9520 |
| Text-score Spearman / Pearson | 0.5706 / 0.6636 | 0.4614 / 0.5657 |
| Semantic axes mean Spearman / Pearson | 0.7643 / 0.8118 | 0.6730 / 0.7322 |
| Semantic axes mean preservation | 0.8385 | 0.7968 |
| Counterfactual robustness margin mean | 0.5167 | 0.5083 |
| Counterfactual transition rate | 1.0 | 1.0 |
| Interpolation mean smoothness | 0.9998768 | 0.9998487 |
| Interpolation abrupt transition rate | 0.0175 | 0.0075 |
| NMAS | 0.5306517203072261 | 0.41187460775327206 |

## SHARED1000 Protocol Finding

The official saved benchmark `experimental_results/V62a_cls_retrieval_768d/subj01/metrics/shared1000_metrics.json` is reproduced exactly by using:

```python
gallery = shared1000_ground_truth.npy
target_indices = np.arange(1000)
```

It is not reproduced by filtering `clip.parquet` by `nsdId`, and it is not the full 10k gallery protocol.

## Caveats

- The evidence is from visual-stimulus-related fMRI responses to observed images.
- Text probes and semantic axes interpret decoded CLIP embeddings relative to CLIP language concepts; they do not decode thoughts, dreams, or consciousness.
- Counterfactual edits operate on decoded CLIP latent vectors and do not manipulate brain activity.
- Interpolation paths are latent-space paths, not real neural trajectories.
- SHARED1000 kappa uncertainty remains unavailable because `shared1000_kappas.npy` is missing.
- Repeat stability requires per-trial predictions.
- ROI-axis mapping requires ROI masks and ROI-masked inference/export.
