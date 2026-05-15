# NSD subj01 — Model results and limitations (pod snapshot)

This document was generated from **live metrics files** on the JupyterHub pod  
(`experimental_results/<experiment>/subj01/metrics/`) as of **2026-05-15**.  
All **shared1000** and **validation** numbers below are **read from `summary.json` / `shared1000_metrics.json`** — not invented.

**Scope:** Experiments under `experimental_results/` with a `subj01/metrics/` directory.  
**Primary subject:** **subj01** (NSD).  
**Retrieval gallery (shared1000):** **1000** unique held-out images (SHARED1000 protocol); CLIP target dimension is reported per run as `clip_dim` (768 = CLS / compact space, 197376 = ViT-L/14 **257×768** token targets flattened).

---

## How to read the metrics

| Field | Meaning |
|--------|--------|
| **Val R@1 / Val CSLS R@1** | Retrieval on the **validation split** (image-level split; shared1000 images excluded from train/val). |
| **Shared1000 R@1 / CSLS R@1** | Retrieval on the **SHARED1000** benchmark (same 1000 images for all rows here). |
| **`final_compact_csls_r@1` in summary** | Often mirrors headline **shared1000 CSLS R@1** when the run saved compact retrieval metrics. |

---

## Cross-cutting limitations (all models)

1. **Val vs shared1000 gap**  
   Many pipelines first **pretrain** on multi-subject or overlapping image pools, then **finetune** with the same random **seed=42** image split. A large fraction of **validation** images can appear in the **parent pretrain train split** (~90% overlap was observed for V60b/V61a-style chains). That **does not** break train/val leakage inside the finetune run, but it **inflates validation retrieval** versus the **fully held-out** shared1000 set. **Shared1000 is the stricter, paper-grade number.**

2. **Target space**  
   **768-D** runs are **not** comparable to **197K-D** token runs on raw percentage points alone: different difficulty, hubness, and score geometry (CSLS partly mitigates hubness but does not remove all effects).

3. **External SOTA**  
   Published **MindEye2**-style numbers often use **different CLIP backbones** (e.g. bigG), **different galleries** (e.g. 300 images), and **different splits**. Direct numeric comparison to this **1000-image shared1000** table requires care.

4. **MC-TTA**  
   Some runs ship extra prediction tensors (e.g. `shared1000_predictions_mctta16.npy`). Where present, **MC-TTA can improve** shared1000 CSLS; the table notes availability but **defaults to the canonical `shared1000_metrics.json`** for the single headline row unless stated otherwise.

---

## Summary table (subj01)

Percentages = value × 100. `—` = file not present on pod.

| Experiment | Val R@1 | Val CSLS R@1 | Shared1000 R@1 | Shared1000 CSLS R@1 | clip_dim | Notes |
|------------|---------|--------------|----------------|----------------------|----------|--------|
| N1v28a_dual_head | — | — | 52.9% | **70.1%** | 197376 | Legacy dual-head baseline. |
| V35_legacy_teacher_distill | 44.9% | — | 42.1% | 51.8% | 768 | Checkpoint metric in summary is `fused_r@1` (not CSLS). |
| V55d_oof_fold_0 | 64.6% | 84.7% | — | — | — | OOF fold; no shared1000 export on pod. |
| V55d_oof_fold_1 | 66.5% | 85.7% | — | — | — | OOF fold. |
| V55d_oof_fold_2 | 67.4% | 86.6% | — | — | — | OOF fold; `best_val_loss` NaN in JSON. |
| V55d_oof_fold_3 | 69.7% | 85.4% | — | — | — | OOF fold. |
| V55d_oof_fold_4 | 68.5% | 84.0% | — | — | — | OOF fold. |
| V56a_fusion_distill_fixed | 35.6% | 39.7% | 30.8% | 35.6% | 768 | Fusion/distill line; weak on shared1000. |
| V56c_projection_rdrop | 28.6% | 34.9% | 22.6% | 29.1% | 768 | Projection + R-Drop experiment. |
| V57a_roi_transformer_dual_head | 27.1% | 32.3% | 36.3% | **44.8%** | 768 | ROI Transformer dual-head. |
| V58a_improved_197k | 45.2% | 57.9% | 47.5% | **61.1%** | 197376 | Improved 197K pipeline. |
| V59a_retrieval_only_768d | 41.0% | 47.1% | 35.6% | 40.3% | 768 | Retrieval-only 768-D. |
| V60a_cross_subject_197k | 38.2% | 43.2% | 34.2% | **42.1%** | 768 | 4-subject pretrain (768-D CLS on pod export). |
| V60b_subj01_finetune_197k | 91.6% | **96.3%** | 59.0% | **73.6%** | 197376 | Large val↔shared1000 gap (decoder / cross-phase). |
| V60c_kappa_gated_197k | 36.9% | 41.2% | 34.8% | 41.8% | 768 | Kappa-gated branch (768-D export). |
| V60d_subj01_finetune_kappa | 93.2% | **97.0%** | 56.7% | **71.7%** | 197376 | Kappa-gated finetune 197K. |
| V61a_finetune_difflr | 95.3% | **98.6%** | 66.0% | **79.1%** | 197376 | **Strongest single 197K shared1000** in this table; has `mctta16` preds on pod. |
| V62a_cls_retrieval_768d | 94.7% | 96.1% | 39.9% | **48.3%** | 768 | Pure CLS; good for **cross-space fusion** with 197K. |
| V62b_triple_head_768d_197k | 94.7% | 96.1% | 38.7% | **46.7%** | 768 | Dual-head; headline compact space. |
| V63a_strong_197k | 91.9% | 96.9% | 59.2% | **75.4%** | 197376 | “Stronger” 197K recipe; below V61a on shared1000. |
| V63b_cls_from_v61a | 94.4% | 95.7% | 39.4% | **46.0%** | 768 | CLS from V61a encoder. |
| V64a_continued_v61a | 95.8% | **97.8%** | 70.6% | **78.6%** | 197376 | Continued V61a training. |
| V64b_frozen_encoder_deep_decoder | 91.8% | 96.9% | 57.3% | **73.7%** | 197376 | Frozen encoder + deeper decoder. |
| V65a_mlp_v9_recipe | 89.4% | 96.4% | 58.3% | **72.5%** | 197376 | V9 loss stack; **below V61a** on shared1000 despite higher val. |
| V66a_roi_pretrain | 22.3% | 23.3% | 35.5% | **39.8%** | 768 | Multi-subject ROI pretrain (768-D). |
| V66b_roi_finetune_197k | 74.8% | **86.2%** | 38.1% | **50.2%** | 197376 | ROI→197K finetune; **weak shared1000** vs V61a. |

**Pod folders without `summary.json` / shared1000 in this scan:**  
`V55a_multi_subject_dual_head`, `V55b_subj01_finetune`, `V55d_oof_merged` (metrics dirs missing or empty for the glob used).

---

## Per-experiment notes

### N1v28a_dual_head
- **Results:** Shared1000 CSLS R@1 **70.1%** (197K space).  
- **Limitations:** Older baseline; `summary.json` on pod shows placeholder/zero training epochs — treat training-time fields as **non-authoritative**; rely on **shared1000_metrics.json**.

### V35_legacy_teacher_distill
- **Results:** Shared1000 CSLS R@1 **51.8%** (768-D).  
- **Limitations:** Optimized for fused / distillation metrics; **not** competitive on shared1000 retrieval vs later V60+.

### V55d OOF folds (0–4)
- **Results:** High **val CSLS R@1** (~84–87%) on each fold’s validation design.  
- **Limitations:** **No `shared1000_metrics.json`** on pod for these dirs — **cannot** report held-out 1000-image benchmark from this snapshot. Fold 2 summary has **NaN** `best_val_loss` (artifact).

### V56a / V56c
- **Results:** Shared1000 CSLS **35.6%** / **29.1%**.  
- **Limitations:** Early fusion / projection experiments; **not** suitable as production retrieval models vs V61+.

### V57a_roi_transformer_dual_head
- **Results:** Shared1000 CSLS **44.8%** (768-D).  
- **Limitations:** ROI Transformer **before** V66-era data/recipe improvements; val metrics low but shared1000 **better than val** in relative terms vs some MLP val-inflated runs.

### V58a_improved_197k / V59a_retrieval_only_768d
- **Results:** V58a shared1000 CSLS **61.1%** (197K); V59a **40.3%** (768-D).  
- **Limitations:** Bridge-era configs; superseded by V60–V64 tuning for 197K / CLS.

### V60a / V60c
- **Results:** V60a shared1000 CSLS **42.1%** (768-D export); V60c **41.8%** (768-D).  
- **Limitations:** **Pretrain / auxiliary** quality — not final single-subject 197K products.

### V60b / V60d (197K finetune from V60 lineage)
- **Results:** Shared1000 CSLS **73.6%** / **71.7%**.  
- **Limitations:** **Very high val CSLS** vs shared1000 → classic **decoder under-training + cross-phase val inflation** story.

### V61a_finetune_difflr (**reference 197K system**)
- **Results:** Shared1000 CSLS **79.1%**; R@1 **66.0%**. MC-TTA predictions (`*_mctta16`) exist on pod for fusion.  
- **Limitations:** Same val inflation pattern as other V60a-initialized MLP runs; still **best single-model shared1000** in this table.

### V62a / V62b / V63b (768-D / compact headline)
- **Results:** Shared1000 CSLS **48.3%** / **46.7%** / **46.0%** (768-D).  
- **Limitations:** **Capacity / objective** trade-offs in CLS-only or triple-head compact metrics; **useful for ensemble** with 197K, not as standalone 197K replacement.

### V63a_strong_197k / V64a / V64b
- **Results:** Shared1000 CSLS **75.4%** / **78.6%** / **73.7%**.  
- **Limitations:** V63a/V64a **did not beat V61a** on shared1000; V64b shows **frozen encoder + new decoder** was harmful vs V61a.

### V65a_mlp_v9_recipe
- **Results:** Shared1000 CSLS **72.5%** (below V61a’s 79.1%).  
- **Limitations:** V9 regularizers helped **val** but **hurt or failed to improve** strict generalization vs V61a on this benchmark.

### V66a_roi_pretrain / V66b_roi_finetune_197k
- **Results:** V66a shared1000 CSLS **39.8%** (768-D); V66b **50.2%** (197K). Val CSLS for V66b is **86.2%** — large gap.  
- **Limitations:** **V66b underperforms** on shared1000 vs V61a; ROI track **did not** deliver complementary strong 197K embeddings for fusion at single-model level (fusion gains instead come mainly from **V61a + V62a + V66a** score fusion — see pod log below).

---

## Ensemble / fusion (pod, 2026-05-15)

From `logs/v65_v66/cross_architecture_fusion_20260515.log` on the pod (score-level z-fusion + CSLS **k** sweep):

- **Best single model (CSLS k=3):** **V61a MC-TTA (16 draws)** ≈ **83.1%** CSLS R@1.  
- **Best triple fusion (example):** **V61a_mctta16 + V62a + V66a** ≈ **86.3%** CSLS R@1 @ **k=3**.  
- **Still below 90%** on this benchmark with current checkpoints.

---

## Reproducibility

- Each run’s `summary.json` → `manifest.system.git_commit` records the **code revision** on the pod when available (`unknown` for some local runs).  
- To refresh this document, re-run the inventory script on the pod and merge any new `experimental_results/*/subj01/metrics/` trees.

---

## Disclaimer

Numbers are **as reported by the training pipeline’s saved JSON**. If a metric file is corrupt, incomplete, or from an abandoned partial run, interpret with care. **No metrics were fabricated** for experiments missing `shared1000_metrics.json`.
