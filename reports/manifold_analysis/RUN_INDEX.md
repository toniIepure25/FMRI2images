# Manifold Analysis Run Index

Updated: 2026-05-15.

This index separates validation, SHARED1000 full-gallery, SHARED1000 target-gallery, and token-space reports. It also marks diagnostic reports that must not be used for final scientific claims.

| Report directory | Model/run | Split/protocol | Status | N | Gallery size | Prediction dim | Cosine R@1 | CSLS R@1 | RSA Spearman | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `reports/manifold_analysis/20260514_V62a_val_subj01_hardened/` | V62a CLS retrieval, subj01 | validation, full 10k gallery | valid | 900 | 10000 | 768 | 0.9178 | 0.9189 | 0.5832 | Hardened 768-D CLIP manifold report with valid `target_ids`, full gallery mapping, bootstrap CIs, RDM error metrics, neighborhood random baselines, kappa, reliability, NMAS, vector field, and claim tests. |
| `reports/manifold_analysis/20260514_V62a_shared1000_subj01_hardened_with_ids/` | V62a CLS retrieval, subj01 | SHARED1000 queries over full 10k gallery | valid, not directly comparable to `shared1000_metrics.json` | 1000 | 10000 | 768 | not locally extracted | not locally extracted | not locally extracted | Cluster run loaded valid `(1000,768)` predictions/targets and `shared1000_nsd_ids.npy`, with `has_gallery_indices=True`. It had no kappa/text/axis artifacts. Claims: 3 supported, 2 partial, 2 not supported, 7 unavailable. Detailed metrics were not present in this local workspace at update time. |
| `reports/manifold_analysis/20260514_V62a_shared1000_subj01_hardened/` | V62a CLS retrieval, subj01 | SHARED1000 full 10k gallery attempt | invalid/diagnostic | 1000 | 10000 | 768 | excluded | excluded | excluded | Ran before `shared1000_nsd_ids.npy` existed and used diagonal fallback: "No target_gallery_indices and N (1000) != G (10000); assuming diagonal". Exclude retrieval, hubness, neighborhood, and claim conclusions. |
| `reports/manifold_analysis/20260514_V62a_shared1000_1k_gallery_hardened/` | V62a CLS retrieval, subj01 | attempted SHARED1000 target-ID parquet gallery | diagnostic/misconfigured unless regenerated | 1000 | expected 1000, but cluster log still showed 10000 loaded before final mode logging existed | 768 | 0.1160 | 0.1340 | not indexed | This run did not reproduce `shared1000_metrics.json` (`cosine R@1 0.399`, `CSLS R@1 0.483`). Treat as diagnostic until regenerated with a confirmed final gallery shape `(1000,768)`. |
| `reports/manifold_analysis/20260514_V62a_shared1000_target_embeddings_hardened/` | V62a CLS retrieval, subj01 | SHARED1000 target-embeddings 1k gallery | valid protocol; report pending cluster rerun | 1000 expected | 1000 expected | 768 | expected/manual check 0.399 | expected about 0.483 | pending | New direct benchmark-reproduction config: `configs/manifold_analysis_V62a_shared1000_target_embeddings.template.yaml`. Uses `z_target` itself as the 1000-candidate gallery and `target_indices=arange(N)`. Manual cluster check reproduced cosine `shared1000_metrics.json` exactly: R@1 0.399, R@5 0.732, R@10 0.837, median rank 2.0. |
| `reports/manifold_analysis/20260514_V61a_shared1000_mctta_token/` | V61a MC-TTA token-space | SHARED1000-like token-space full gallery | valid token-space report | 1000 | 10000 inferred | 197376 | 0.0920 | 0.1650 | 0.1658 | 197376-D token-space report. It is not a 768-D CLIP manifold report and must not be mixed with V62a 768-D CLIP results. |
| `reports/manifold_analysis/20260514_V61a_shared1000_mctta_token_full/` | V61a MC-TTA token-space | SHARED1000-like token-space full gallery | valid token-space report | 1000 | 10000 inferred | 197376 | 0.0920 | 0.1650 | 0.1658 | Same visible retrieval/hubness/RSA metrics as the token report, with additional figures. |
| `20260514_V62a_val_subj01/` | V62a CLS retrieval, subj01 | historical validation report at repo root | historical | 900 | 10000 inferred | 768 | 0.9178 | 0.9189 | 0.5832 | Older local copy before final hardened report path. Legacy summary displayed ranks as 0-indexed; future reports use 1-indexed ranks. |

## V62a SHARED1000 Artifacts

Expected cluster base path:

`experimental_results/V62a_cls_retrieval_768d/subj01/metrics/`

Files:

- `shared1000_predictions.npy`: shape `(1000, 768)`, dtype `float32`
- `shared1000_ground_truth.npy`: shape `(1000, 768)`, dtype `float32`
- `shared1000_nsd_ids.npy`: shape `(1000,)`, copied from the matching V61a run after verifying other shared1000 ID files had identical shape/first values
- `shared1000_metrics.json`: benchmark metadata and 1k-gallery retrieval metrics
- `shared1000_kappas.npy`: unavailable

First verified `shared1000_nsd_ids.npy` values:

`[2950, 2990, 3049, 3077, 3146, 3157, 3164, 3171, 3181, 3386, ...]`

ID range: min `2950`, max `72948`, unique `1000`.

`shared1000_metrics.json` reports `gallery_size=1000`, `r@1=0.399`, `r@5=0.732`, `r@10=0.837`, `median_rank=2.0`, `mrr=0.5452108744517535`, `csls_r@1=0.483`, `csls_r@5=0.802`, `csls_r@10=0.897`, `r@1_avg=0.401`, and `csls_r@1_avg=0.486`.

Critical protocol finding: the saved benchmark is reproduced exactly for cosine retrieval by using `shared1000_ground_truth.npy` as the 1000-image gallery and `target_indices=np.arange(1000)`. It is not reproduced by filtering `clip.parquet` by `nsdId`.

## Protocol Caveats

- V62a validation hardened is a valid 768-D CLIP manifold report over a full 10k gallery.
- V62a SHARED1000 with IDs over the full 10k gallery is valid, but it is a harder/different protocol than the saved `shared1000_metrics.json`.
- V62a SHARED1000 target-ID parquet mode is useful when the desired gallery is a subset of `clip.parquet`, but the observed `20260514_V62a_shared1000_1k_gallery_hardened` metrics (`0.116/0.134` R@1) do not match `shared1000_metrics.json` and should be treated as diagnostic unless regenerated with a confirmed final 1000-row gallery.
- V62a SHARED1000 target-embeddings mode is the most direct comparable benchmark protocol for `shared1000_metrics.json`; it uses `shared1000_ground_truth.npy` as the gallery and should produce gallery size 1000.
- V61a MC-TTA reports are valid token-space analyses, not 768-D CLIP manifold analyses.
- `fusion_v61_v62` is score/rank-level fusion. No true fused 768-D embedding export was found, so fusion RSA/neighborhood/manifold geometry remains unavailable.
