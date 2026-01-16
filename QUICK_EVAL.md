# Quick Start: Evaluate Your Training Run

**You're on JupyterHub cluster** - Use this simple command to evaluate your completed training:

```bash
cd ~/Bachelor_V2

python scripts/evaluate_experiment.py \
    --checkpoint runs/20260115_172845_ultimate_novel_subj01/checkpoints/best_model.pt \
    --config experiments/ultimate_novel_subj01.yaml \
    --exp-name exp001_baseline_ultimate \
    --num-samples 1000
```

This will:
1. ✅ Evaluate with all CLIP metrics (cosine similarity, retrieval, KL divergence)
2. ✅ Create organized results in `experimental_results/exp001_baseline_ultimate/`
3. ✅ Generate summary report with automatic recommendations
4. ✅ Save metrics for future comparison

Then check: `experimental_results/exp001_baseline_ultimate/evaluation/summary_report.md`

---

See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for full workflow and [experimental_results/README.md](experimental_results/README.md) for detailed structure.
