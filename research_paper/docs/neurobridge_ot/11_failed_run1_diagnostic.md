# Failed Run1 Diagnostic Report

## Run Identification

| Field | Value |
|---|---|
| **Run path** | `experimental_results/neurobridge_ot_8subj_full_run1/` |
| **Config** | `configs/experiments/neurobridge_ot/neurobridge_ot_8subj_full.yaml` |
| **Branch** | `feature/neurobridge-ot` |
| **Status** | Completed — retrieval fully random |
| **Started** | 2026-05-28 22:27 UTC |
| **Stopped** | 2026-05-29 00:02 UTC (early stopping at epoch 30) |
| **Subjects** | All 8 NSD subjects |
| **Total trials** | 213,000 (172,648 train / 19,234 val after SHARED1000 exclusion) |
| **Model params** | 55,964,993 |
| **GPU** | NVIDIA H100 80GB HBM3 |

## Results

| Metric | Value | Expected if working |
|---|---|---|
| Best val R@1 | 5.2×10⁻⁵ (0.005%) | >1% within 10 epochs |
| Val median rank | 9,616 | <5,000 and decreasing |
| Train loss (final) | -6.88 | Irrelevant without retrieval |
| Val loss (final) | -6.88 | — |
| Epochs completed | 30/200 | — |

## Interpretation

This is **NOT a valid scientific negative result**. It is a failed diagnostic run.

- Median rank 9,616 ≈ N/2 where N≈19,234 → pure random retrieval.
- R@1 ≈ 1/19,234 ≈ 5.2×10⁻⁵ → exactly random chance.
- Training loss decreased (optimization happened) but the learned representation has zero retrieval signal.
- The model optimized auxiliary losses (vMF NLL, adversarial) without learning the CLIP target mapping.

## Root Causes Identified

### 1. ROI indices never built (CRITICAL)

```
WARNING | Failed to build ROI indices for subj01: build_roi_index() missing 1 required positional argument: 'roi_names'
```

All 8 subjects produced this warning. The model's ROI tokenizer received empty `roi_indices={}` for every batch. Without ROI partitioning, the "Learned ROI Tokenizer" has no valid voxel groupings to attend over — it processes empty/zero tokens.

**Impact:** The entire NeuroBridge-OT architectural premise (brain-topology-aware tokenization → OT alignment → semantic transformer) is disabled. The model effectively sees nothing.

### 2. Embedding column mismatch (SUSPECTED)

Config specifies `embedding_column: "fused"` but the loaded cache (`clip_multilayer.parquet`) has columns: `['nsdId', 'layer_12', 'layer_12_proj', 'layer_18', 'layer_18_proj']`. No `fused` column exists.

The dataset code does NOT raise an error when a non-existent column is explicitly passed — it just produces empty lookups or zero-vector fallbacks. Needs verification of what targets were actually used.

### 3. Validation gallery too large for trial-level R@1

The validation set has 19,234 trials across 8 subjects. Many trials share the same `nsdId` (image shown 3x). The current `validate()` computes N×N retrieval with diagonal ground truth — meaning trial `i` must rank position `i` highest among 19,234 candidates, even though multiple positions share the same CLIP target.

This is an overly strict metric that underestimates real retrieval quality and makes early-stopping unstable.

### 4. No embedding column validation at load time

The dataset's `_build_clip_lookup()` silently produces an empty lookup if the column doesn't exist or contains non-array values. Combined with the zero-vector fallback in `__getitem__`, the model may have trained against all-zeros targets.

### 5. Model saw no useful input signal

With empty ROI indices, the tokenizer produces zero/random tokens. The semantic transformer processes garbage. The decoding heads map garbage to 768-D. The contrastive loss computes similarities between garbage predictions and (possibly invalid) targets.

## Exact Warnings From Log

```
Failed to build ROI indices for subj01: build_roi_index() missing 1 required positional argument: 'roi_names'
Failed to build ROI indices for subj02: build_roi_index() missing 1 required positional argument: 'roi_names'
Failed to build ROI indices for subj03: build_roi_index() missing 1 required positional argument: 'roi_names'
Failed to build ROI indices for subj04: build_roi_index() missing 1 required positional argument: 'roi_names'
Failed to build ROI indices for subj05: build_roi_index() missing 1 required positional argument: 'roi_names'
Failed to build ROI indices for subj06: build_roi_index() missing 1 required positional argument: 'roi_names'
Failed to build ROI indices for subj07: build_roi_index() missing 1 required positional argument: 'roi_names'
Failed to build ROI indices for subj08: build_roi_index() missing 1 required positional argument: 'roi_names'
```

## Why This Cannot Be Used For Claims

1. The architectural core (ROI tokenization) was disabled.
2. Target alignment is unverified.
3. The result is indistinguishable from random — no learning signal reached retrieval.
4. Run1 artifacts (`best.pt`, `last.pt`) have no scientific value.

## Required Fixes Before Relaunch

1. Pass `roi_names` to `build_roi_index()` — derive from config or use canonical 17-ROI list.
2. Validate and hard-fail on missing `embedding_column`.
3. Add batch alignment audit to prove fMRI↔CLIP target pairing.
4. Add prediction collapse diagnostics.
5. Pass tiny overfit test before scaling up.
6. Fix validation gallery (per-image dedup or per-subject eval).
