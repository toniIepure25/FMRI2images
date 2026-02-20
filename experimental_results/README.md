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
└── exp{N}_{name}/                     # Phase 2 experiments (EXP0-EXP14)
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

| Exp | Description | Key Hypothesis | Status |
|-----|-------------|---------------|--------|
| **Phase 1** | | | |
| exp001 | Baseline Ultimate (all 7 Phase 1 contributions) | Combined system works | Completed (epoch 28) |
| **Phase 2** | | | |
| EXP0 | Deterministic MLP (MSE + cosine) | Baseline reference | Pending |
| EXP1 | + center_pcr preprocessing | H1: PCR reduces hubness | Pending |
| EXP2 | + InfoNCE + memory queue | H2: Contrastive + queue helps retrieval | Pending |
| EXP3 | + Gaussian NLL | H3: Gaussian captures uncertainty | Pending |
| EXP4 | + Gaussian-NCE | H4: Gaussian-NCE improves calibration | Pending |
| EXP5 | + KL annealing | H5: Annealing stabilizes training | Pending |
| EXP6 | + whitening (ablation) | H6: Whitening vs PCR | Pending |
| **EXP7** | **vMF-NCE (MLP)** | **H7: vMF > Gaussian on S^{d-1}** | Pending |
| **EXP8** | **ROI Transformer + vMF-NCE** | **H8: ROI inductive bias helps** | Pending |
| **EXP9** | **ROI-DCF consensus** | **H9: Per-ROI distributions are richer** | Pending |
| **EXP10** | + vMF mixture sampling | H10: Mixture > consensus for generation | Pending |
| **EXP11** | + decomposed UA-CFG | H11: Dual uncertainty > heuristic CFG | Pending |
| **EXP12** | + noise-ceiling temperature | H12: Ceiling-temp improves calibration | Pending |
| **EXP13** | + kappa-SPCL curriculum | H13: Curriculum helps convergence | Pending |
| **EXP14** | **Full system** | **H14: Full > any ablation** | Pending |

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
    --config configs/experiments/exp7_vmf_nce.yaml \
    --gpu 0
```

### Full Ablation Ladder

```bash
bash scripts/training/run_ablation_ladder.sh \
    --subjects "subj01 subj02 subj05 subj07" \
    --gpu 0 \
    --start-exp 7
```

### Cross-Experiment Comparison

After running multiple experiments, compare with statistical tests:

```bash
python3 scripts/evaluation/compare_experiments.py \
    --exp-dirs experimental_results/exp7_vmf_nce \
               experimental_results/exp8_roi_transformer \
               experimental_results/exp9_roi_dcf \
    --output experimental_results/comparison_exp7_vs_exp9.md \
    --paired-test
```

## SOTA Benchmarks (Published, for Reference)

| Method | PixCorr | SSIM | Alex(2) | Alex(5) | R@1 |
|--------|---------|------|---------|---------|-----|
| MindEye (2023) | 0.309 | 0.323 | 0.947 | 0.978 | — |
| MindEye2 (2024) | 0.320 | 0.341 | 0.960 | 0.983 | — |
| Brain Diffuser (2023) | 0.254 | 0.356 | 0.942 | 0.962 | — |

## Conventions

- **Naming**: `exp{N}_{short_name}/` matches `configs/experiments/exp{N}_{short_name}.yaml`
- **Configs are frozen**: Once an experiment starts, its `config.yaml` is copied here
  and never modified. Configuration changes require a new experiment number.
- **Notes are mandatory**: Every completed experiment must have a filled `notes.md`
  documenting observations, surprises, and lessons before starting the next experiment.
- **No fabricated numbers**: All reported metrics must trace to a JSON file in this
  directory. If an experiment hasn't been run, mark it "Pending" — never estimate.
