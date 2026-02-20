# Experimental Results

Structured evaluation outputs for all training experiments. Each experiment
directory contains the full configuration, evaluation metrics, and analysis
needed for reproducible comparison and paper-ready reporting.

## Directory Structure

```
experimental_results/
├── README.md                          # This file
│
├── exp001_baseline_ultimate/          # Phase 1 baseline (completed)
│   ├── config.yaml                    # Training configuration snapshot
│   ├── training_info.json             # Epoch, checkpoint path, training metrics
│   ├── notes.md                       # Observations and analysis
│   └── evaluation/
│       ├── clip_metrics.json          # Cosine similarity, retrieval R@K
│       ├── probabilistic_metrics.json # KL divergence, reconstruction error
│       ├── eval_results.json          # Full evaluation dump
│       └── summary_report.md          # Formatted summary
│
└── {B0,B1,N1,N2,N3,N4}_{name}/       # Phase 2 experiments
    ├── config.yaml                    # Frozen config snapshot
    ├── training_info.json             # Training metadata
    ├── notes.md                       # Per-experiment analysis
    └── evaluation/
        ├── embedding_metrics.json     # Retrieval: R@K, MRR, MedR, CSLS, hubness
        ├── probabilistic_metrics.json # NLL, Energy Score, ECE, coverage
        ├── risk_coverage.json         # AURC, E-AURC, R@80/90/95
        ├── ceiling_normalized.json    # All metrics as % of noise ceiling
        ├── roi_analysis.json          # Per-ROI kappa, attention, lesion study
        ├── kappa_calibration.json     # kappa quantiles for UA-CFG
        └── summary_report.md          # Formatted comparison-ready summary
```

## Ablation Ladder

| ID | Description | Key Hypothesis | Status |
|----|-------------|---------------|--------|
| **Phase 1** | | | |
| exp001 | Baseline Ultimate (all Phase 1 contributions) | Combined system works | Completed (epoch 28) |
| **Phase 2** | | | |
| **B0** | Deterministic MLP (MSE + InfoNCE + queue + PCR) | Strong standard baseline | Pending |
| **B1** | Gaussian MLP (Gaussian-NCE + KL annealing) | Probabilistic baseline | Pending |
| **N1** | **vMF-NCE (MLP encoder)** | **vMF > Gaussian on S^{d-1}** | Pending |
| **N2** | **ROI Transformer + vMF-NCE** | **ROI inductive bias helps** | Pending |
| **N3** | **ROI-DCF consensus** | **Per-ROI distributions are richer** | Pending |
| **N4** | **Full system** (DCF + Mixture + DUA-CFG + Ceiling-Temp + SPCL) | **Full > any ablation** | Pending |

### What Each Comparison Proves

- **B1 vs N1** = Gaussian vs vMF (distributional choice)
- **N1 vs N2** = Flat MLP vs ROI Transformer (architecture)
- **N2 vs N3** = Single-head vs ROI-DCF (fusion strategy)
- **N3 vs N4** = Base system vs full innovations (generation stack)

## Metric Tiers

### Tier 1: Primary (Paper Tables)

| Metric | Type | Direction | Phase 2 Module |
|--------|------|-----------|----------------|
| R@1, R@5 | Retrieval | Higher | `eval/embedding_metrics.py` |
| MRR | Retrieval | Higher | `eval/embedding_metrics.py` |
| AURC | Selective prediction | Lower | `eval/vmf_risk_coverage.py` |
| PixCorr | Reconstruction | Higher | `eval/image_metrics.py` |
| SSIM | Reconstruction | Higher | `eval/image_metrics.py` |

### Tier 2: Calibration and Uncertainty

| Metric | Type | Direction | Phase 2 Module |
|--------|------|-----------|----------------|
| ECE | Calibration | Lower | `eval/probabilistic_metrics.py` |
| Coverage@95 | Calibration | Close to 95% | `eval/probabilistic_metrics.py` |
| Energy Score | Proper scoring | Lower | `inference/vmf_mixture.py` |
| kappa statistics | Uncertainty | Interpretive | `eval/kappa_calibration.py` |

### Tier 3: Neuroscience and Interpretability

| Metric | Type | Direction | Phase 2 Module |
|--------|------|-----------|----------------|
| ROI attention importance | Interpretability | — | `eval/neuroscience_analysis.py` |
| Frequency band weights | Interpretability | — | `models/freq_roi.py` |
| Ceiling-normalized metrics | Fair comparison | Higher | `eval/ceiling_normalized_eval.py` |

## Running Evaluations

### Phase 2 Training

```bash
python3 scripts/training/train_unified.py \
    --config configs/experiments/N1_vmf_nce.yaml \
    --gpu 0
```

### Full Ablation Ladder

```bash
bash scripts/training/run_ablation_ladder.sh \
    --subjects "subj01 subj02 subj05 subj07" \
    --gpu 0 \
    --start N1
```

### Cross-Experiment Comparison

After running multiple experiments, compare with statistical tests:

```bash
python3 scripts/evaluation/compare_experiments.py \
    --exp-dirs experimental_results/N1_vmf_nce \
               experimental_results/N2_roi_transformer \
               experimental_results/N3_roi_dcf \
    --output experimental_results/comparison_N1_vs_N3.md \
    --paired-test
```

## SOTA Benchmarks (Published, for Reference)

| Method | PixCorr | SSIM | Alex(2) | Alex(5) | R@1 |
|--------|---------|------|---------|---------|-----|
| MindEye (2023) | 0.309 | 0.323 | 0.947 | 0.978 | — |
| MindEye2 (2024) | 0.320 | 0.341 | 0.960 | 0.983 | — |
| Brain Diffuser (2023) | 0.254 | 0.356 | 0.942 | 0.962 | — |

## Conventions

- **Naming**: `{ID}_{short_name}/` matches `configs/experiments/{ID}_{short_name}.yaml`
- **Configs are frozen**: Once an experiment starts, its `config.yaml` is copied here
  and never modified. Configuration changes require a new experiment.
- **Notes are mandatory**: Every completed experiment must have a filled `notes.md`
  documenting observations, surprises, and lessons before starting the next experiment.
- **No fabricated numbers**: All reported metrics must trace to a JSON file in this
  directory. If an experiment hasn't been run, mark it "Pending" — never estimate.
