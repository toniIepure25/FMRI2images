# Experiment Configurations

**Research and Ablation Study Templates**

---

## 🧪 Experimental Configurations

### `ablation.yaml` - Ablation Study Template

**Purpose**: Systematic component analysis and ablation studies

```yaml
study:
  name: component_ablation
  components:
    - preprocessing
    - architecture
    - loss_functions
    - training_strategy
  
baseline:
  config: configs/training/two_stage_sota.yaml
  
ablations:
  - name: no_pca
    changes:
      preprocessing.pca.enabled: false
    
  - name: smaller_model
    changes:
      model.latent_dim: 256
      model.num_layers: 2
    
  - name: no_infonce
    changes:
      training.infonce_weight: 0.0
    
  - name: no_residual
    changes:
      model.use_residual: false

evaluation:
  metrics:
    - cosine_similarity
    - retrieval_r1
    - retrieval_r5
    - retrieval_r10
  statistical_tests:
    - paired_t_test
    - wilcoxon_signed_rank
  confidence_level: 0.95
```

---

## 🎯 Types of Experiments

### **1. Component Ablation**

Study the impact of individual components.

**Example**: Effect of preprocessing
```bash
python scripts/ablate_preproc_and_ridge.py \
    --config configs/experiments/ablation.yaml \
    --component preprocessing \
    --variations pca,zscore,both,none
```

**Variations**:
- No PCA
- No normalization
- Different PCA dimensions
- Different normalization methods

### **2. Architecture Ablation**

Study architectural choices.

**Example**: Model capacity
```yaml
ablations:
  - name: small_model
    changes: {model.latent_dim: 256}
  - name: medium_model
    changes: {model.latent_dim: 512}
  - name: large_model
    changes: {model.latent_dim: 768}
  - name: xlarge_model
    changes: {model.latent_dim: 1024}
```

```bash
python scripts/run_architecture_ablation.py \
    --config configs/experiments/ablation.yaml \
    --study model_capacity
```

### **3. Training Strategy Ablation**

Study training hyperparameters.

**Example**: Learning rate schedule
```yaml
ablations:
  - name: constant_lr
    changes: {training.scheduler: constant}
  - name: cosine_lr
    changes: {training.scheduler: cosine}
  - name: exponential_lr
    changes: {training.scheduler: exponential}
  - name: polynomial_lr
    changes: {training.scheduler: polynomial}
```

### **4. Loss Function Ablation**

Study different loss combinations.

**Example**: InfoNCE weight
```yaml
ablations:
  - name: no_infonce
    changes: {training.infonce_weight: 0.0}
  - name: weak_infonce
    changes: {training.infonce_weight: 0.1}
  - name: medium_infonce
    changes: {training.infonce_weight: 0.4}
  - name: strong_infonce
    changes: {training.infonce_weight: 0.8}
```

---

## 📊 Running Ablation Studies

### **Basic Ablation**
```bash
python scripts/run_ablation.py \
    --config configs/experiments/ablation.yaml \
    --study component_ablation \
    --output outputs/ablations/
```

### **With Statistical Analysis**
```bash
python scripts/run_ablation.py \
    --config configs/experiments/ablation.yaml \
    --study component_ablation \
    --statistics \
    --confidence 0.95 \
    --output outputs/ablations/
```

### **Cross-Subject Validation**
```bash
for subject in subj01 subj02 subj03 subj04; do
    python scripts/run_ablation.py \
        --config configs/experiments/ablation.yaml \
        --subject $subject \
        --output outputs/ablations/$subject/
done

# Aggregate results
python scripts/aggregate_ablations.py \
    --input outputs/ablations/*/results.json \
    --output outputs/ablations/summary.csv
```

---

## 📝 Creating Custom Experiments

### **1. Define Study**
```yaml
# configs/experiments/my_study.yaml
_base_: ablation.yaml

study:
  name: my_custom_study
  description: "Investigating effect of X on Y"
  
baseline:
  config: configs/training/two_stage_sota.yaml
  
ablations:
  - name: variation_1
    description: "Test hypothesis A"
    changes:
      model.parameter: value1
  
  - name: variation_2
    description: "Test hypothesis B"
    changes:
      model.parameter: value2
```

### **2. Run Experiment**
```bash
python scripts/run_ablation.py \
    --config configs/experiments/my_study.yaml \
    --subjects subj01,subj02,subj03 \
    --output outputs/experiments/my_study/
```

### **3. Analyze Results**
```bash
python scripts/analyze_ablation.py \
    --input outputs/experiments/my_study/ \
    --report outputs/experiments/my_study/report.pdf
```

---

## 🎓 Best Practices

### **1. Control Variables**

Keep everything constant except what you're testing:

```yaml
ablations:
  - name: test_pca
    changes:
      preprocessing.pca.n_components: 500  # Only change this
    # All other parameters stay same as baseline
```

### **2. Multiple Seeds**

Run with different random seeds for robustness:

```yaml
study:
  seeds: [42, 43, 44, 45, 46]  # Run each ablation 5 times
  aggregate: mean_and_std
```

### **3. Statistical Testing**

Always include statistical significance tests:

```yaml
evaluation:
  statistical_tests:
    - paired_t_test       # Parametric
    - wilcoxon_signed_rank # Non-parametric
  confidence_level: 0.95  # p < 0.05
  multiple_comparison_correction: bonferroni
```

### **4. Document Hypotheses**

Write clear hypotheses before running:

```yaml
ablations:
  - name: no_pca
    hypothesis: "PCA reduces noise and improves performance"
    expected: "Performance drop without PCA"
    changes:
      preprocessing.pca.enabled: false
```

---

## 📊 Example Studies

### **Preprocessing Study**
```bash
python scripts/ablate_preproc_and_ridge.py \
    --config configs/experiments/ablation.yaml \
    --variations pca_500,pca_1000,pca_2000,no_pca \
    --output outputs/ablations/preprocessing/
```

**Questions Answered**:
- How many PCA components are optimal?
- Is PCA necessary?
- Which normalization works best?

### **Architecture Study**
```yaml
study:
  name: architecture_search
  
ablations:
  - {name: depth_2, changes: {model.num_layers: 2}}
  - {name: depth_4, changes: {model.num_layers: 4}}
  - {name: depth_6, changes: {model.num_layers: 6}}
  - {name: width_256, changes: {model.latent_dim: 256}}
  - {name: width_512, changes: {model.latent_dim: 512}}
  - {name: width_768, changes: {model.latent_dim: 768}}
```

**Questions Answered**:
- Optimal model depth?
- Optimal model width?
- Depth vs width tradeoff?

### **Training Study**
```yaml
study:
  name: training_strategy
  
ablations:
  - {name: lr_1e-3, changes: {training.learning_rate: 0.001}}
  - {name: lr_1e-4, changes: {training.learning_rate: 0.0001}}
  - {name: lr_1e-5, changes: {training.learning_rate: 0.00001}}
  - {name: bs_16, changes: {training.batch_size: 16}}
  - {name: bs_32, changes: {training.batch_size: 32}}
  - {name: bs_64, changes: {training.batch_size: 64}}
```

**Questions Answered**:
- Optimal learning rate?
- Optimal batch size?
- LR and batch size interaction?

---

## 📈 Analyzing Results

### **Generate Report**
```bash
python scripts/report_ablation.py \
    --input outputs/ablations/my_study/ \
    --output outputs/ablations/my_study/report.html \
    --include-plots
```

**Report Includes**:
- Performance comparison table
- Statistical significance tests
- Visualization plots
- Confidence intervals
- Effect sizes

### **Plot Results**
```bash
python scripts/plot_ablation.py \
    --input outputs/ablations/my_study/results.json \
    --output outputs/ablations/my_study/plots/ \
    --metrics cosine_similarity,retrieval_r1
```

---

## 🐛 Troubleshooting

### **Issue: Inconsistent Results**

**Solution**: Use multiple seeds and aggregate
```yaml
study:
  seeds: [42, 43, 44, 45, 46]
  aggregate: mean_and_std
```

### **Issue: Not Enough Samples**

**Solution**: Use cross-validation or bootstrap
```yaml
evaluation:
  method: k_fold_cross_validation
  k: 5
```

### **Issue: Multiple Comparisons**

**Solution**: Apply correction
```yaml
evaluation:
  multiple_comparison_correction: bonferroni
  # or: holm, fdr_bh
```

---

## 📚 Related Documentation

- **[Main Config README](../README.md)** - Overview
- **[Training Configs](../training/README.md)** - Training settings
- **[Evaluation Guide](../../docs/guides/EVALUATION_SUITE_GUIDE.md)** - Metrics
- **[Usage Examples](../../USAGE_EXAMPLES.md)** - Complete commands

---

## 💡 Research Tips

1. **Start small**: Test on single subject first
2. **Control variables**: Change one thing at a time
3. **Use statistics**: Always test significance
4. **Document**: Write hypotheses before running
5. **Replicate**: Use multiple seeds
6. **Share**: Make configs and results public

---

**Last Updated**: December 7, 2025  
**Status**: Research Tool  
**Use For**: Ablation studies, hyperparameter search, component analysis
