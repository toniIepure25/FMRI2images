# Evaluation Scripts

Evaluation and metrics for fMRI-to-CLIP models.

**10 evaluation scripts**

## Entry Point

-  **`evaluate.py`** - Main wrapper forwarding to `evaluate_experiment.py`

## Implementations

- **`evaluate_experiment.py`** ⭐ - Comprehensive experiment evaluation
  - Retrieval metrics: R@K, MRR, nDCG
  - Identification: 2AFC, AUC
  - Representational similarity: RSA, CKA
  - Probabilistic metrics: calibration, conformal prediction
  - Plots and comprehensive reports
  
- **`eval_stage1_embeddings.py`** ⭐ - Embedding-space evaluation
- `eval_stage1_probabilistic.py` - Probabilistic/uncertainty evaluation
- `eval_comprehensive.py` - All metrics combined
- `eval_shared1000_full.py` - Shared1000 benchmark evaluation
- `eval_reconstruction.py` - Reconstruction quality evaluation
- `eval_retrieval.py` - Retrieval-specific metrics
- `summarize_shared1000.py` - Summarize Shared1000 results

## Usage

```bash
# Using wrapper (recommended)
python scripts/evaluation/evaluate.py --checkpoint outputs/exp0/best.pt

# Direct invocation
python scripts/evaluation/evaluate_experiment.py --checkpoint outputs/exp0/best.pt
```
