# Analysis Scripts

Results analysis, comparison, and visualization.

## Scripts

- `analyze_run.py` - Analyze single experiment run
  - Load metrics and logs
  - Generate visualizations
  - Summarize performance
  
- `compare_experiments.py` - Compare multiple experiments
  - Side-by-side comparison
  - Statistical significance testing
  - Generate comparison tables
  
- `compare_evals.py` - Compare evaluation results
  - Compare different evaluation metrics
  - Visualize differences
  
- `ablate_preproc_and_ridge.py` - Preprocessing ablation study
  - Test different preprocessing configurations
  - Ridge baseline comparisons
  
- `build_paper_artifacts.py` - Generate paper figures and tables
  - Publication-ready figures
  - LaTeX tables
  - Result summaries

## Usage

```bash
# Analyze single run
python scripts/analysis/analyze_run.py outputs/exp0/

# Compare multiple experiments
python scripts/analysis/compare_experiments.py \
    experimental_results/exp0_baseline/ \
    experimental_results/exp1_contrastive/

# Generate paper artifacts
python scripts/analysis/build_paper_artifacts.py
```
