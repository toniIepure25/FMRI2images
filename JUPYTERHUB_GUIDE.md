# Running Experiments on JupyterHub Cluster

## Cluster Specs
- **CPU**: 20 cores (Intel Xeon Platinum 8380 @ 2.30GHz)
- **RAM**: 125GB
- **GPU**: NVIDIA A100D-20C (20GB VRAM)
- **CUDA**: 12.2

---

## Quick Start

### 1. Build Preprocessors (1-2 hours)
```bash
# In your terminal on GPU20C-N5
cd ~/Bachelor_V2
source venv/bin/activate

# Build preprocessors
bash scripts/build_all_preprocessors.sh
```

### 2. Run All Experiments (3-7 days)
```bash
# Start experiments (runs in foreground with logging)
bash scripts/run_all_experiments.sh 0
```

**Log file**: Automatically created at `experimental_results/logs/run_all_experiments_YYYYMMDD_HHMMSS.log`

### 3. Run in Background (Recommended for Long Runs)
```bash
# Run in background with nohup
nohup bash scripts/run_all_experiments.sh 0 > /dev/null 2>&1 &

# Get the process ID
echo $!

# Or use tmux/screen (better option)
tmux new -s experiments
bash scripts/run_all_experiments.sh 0
# Press Ctrl+B then D to detach
# Reattach: tmux attach -t experiments
```

---

## Monitoring Progress

### Option 1: Monitor Latest Log (Real-time)
```bash
# In a separate terminal/tmux pane
bash scripts/monitor_experiments.sh

# Or manually
tail -f experimental_results/logs/run_all_experiments_*.log
```

### Option 2: Check Status (Periodic)
```bash
# Get current status of all experiments
bash scripts/check_experiment_status.sh
```

### Option 3: Watch GPU Usage
```bash
# Watch GPU in real-time
watch -n 1 nvidia-smi

# Or check once
nvidia-smi
```

---

## What the Log File Contains

The enhanced logging captures:

1. **System Info**:
   - Hostname, CPU count, memory, GPU details
   - Start timestamp

2. **Per-Experiment Progress**:
   - Start/end times with timestamps
   - Training duration
   - Evaluation results
   - Key metrics (R@1, MeanR, Oracle check)
   - GPU status after each experiment

3. **Summary**:
   - Total duration
   - All experiments with R@1 scores
   - Next steps

### Example Log Output:
```
[2026-01-17 21:00:00] =================================================================================
[2026-01-17 21:00:00] ABLATION SUITE - Full Experiment Run
[2026-01-17 21:00:00] =================================================================================
[2026-01-17 21:00:00] Start time: Sat Jan 17 21:00:00 CET 2026
[2026-01-17 21:00:00] GPU: 0
[2026-01-17 21:00:00] Log file: experimental_results/logs/run_all_experiments_20260117_210000.log
[2026-01-17 21:00:00] 
[2026-01-17 21:00:00] System Information:
[2026-01-17 21:00:00]   Hostname: GPU20C-N5
[2026-01-17 21:00:00]   CPUs: 20
[2026-01-17 21:00:00]   Memory: 125Gi
[2026-01-17 21:00:00]   GPU Info:
[2026-01-17 21:00:00]     GRID A100D-20C, 20480 MiB
...
[2026-01-17 21:00:05] =================================================================================
[2026-01-17 21:00:05] [1/7] Starting: exp0_baseline
[2026-01-17 21:00:05] =================================================================================
...
[2026-01-17 23:15:30] ✅ exp0_baseline training complete
[2026-01-17 23:15:30]    Duration: 2h 15m 25s
...
```

---

## Useful Commands

### Check if experiments are running
```bash
ps aux | grep run_all_experiments
```

### Find log files
```bash
ls -lht experimental_results/logs/
```

### View latest log
```bash
# View latest
cat experimental_results/logs/run_all_experiments_*.log | tail -100

# Search for errors
grep -i "error\|fail" experimental_results/logs/run_all_experiments_*.log
```

### Check experiment outputs
```bash
# List completed experiments
ls -d experimental_results/exp*/evaluation/

# Quick R@1 summary
for exp in exp{0..6}_*; do
    if [ -f "experimental_results/${exp}/evaluation/metrics.json" ]; then
        echo -n "${exp}: "
        jq -r '.retrieval."R@1"' experimental_results/${exp}/evaluation/metrics.json
    fi
done
```

### Stop experiments (if needed)
```bash
# Find process ID
ps aux | grep run_all_experiments

# Kill by PID
kill <PID>

# Or if running in tmux
tmux attach -t experiments
# Then Ctrl+C
```

---

## Expected Timeline (on your A100)

Based on your GPU specs:

| Experiment | Training Time | Evaluation | Total |
|------------|--------------|------------|-------|
| EXP0 (baseline) | 6-12 hours | 30 min | ~12 hours |
| EXP1 (+preproc) | 6-12 hours | 30 min | ~12 hours |
| EXP2 (+queue) | 8-14 hours | 30 min | ~14 hours |
| EXP3 (+NLL) | 8-14 hours | 45 min | ~14 hours |
| EXP4 (+G-NCE) | 8-14 hours | 45 min | ~14 hours |
| EXP5 (+KL) | 8-14 hours | 45 min | ~14 hours |
| EXP6 (whiten) | 8-14 hours | 45 min | ~14 hours |

**Total**: ~3-5 days (70-100 hours)

*Note: Times depend on dataset size and early stopping*

---

## Recommendations for Cluster Usage

### 1. Use tmux for Long Runs
```bash
# Start tmux session
tmux new -s experiments

# Inside tmux
cd ~/Bachelor_V2
source venv/bin/activate
bash scripts/run_all_experiments.sh 0

# Detach: Ctrl+B then D
# Your experiments keep running!

# Later, reattach
tmux attach -t experiments

# List sessions
tmux ls
```

### 2. Monitor in Separate tmux Pane
```bash
# Split tmux horizontally: Ctrl+B then "
# In bottom pane
bash scripts/monitor_experiments.sh

# Or
watch -n 60 bash scripts/check_experiment_status.sh
```

### 3. Optimize for A100
Your A100 has 20GB VRAM, which is plenty. You can potentially:
- Increase batch size in configs (currently 4)
- Increase queue size (currently 8192)
- Run multiple subjects in parallel (if data permits)

---

## Troubleshooting

### "Out of memory" errors
```bash
# Check current GPU usage
nvidia-smi

# Reduce batch size in config
# Edit configs/experiments/exp*.yaml:
#   training.batch_size: 2  # (reduce from 4)
```

### "Permission denied" on log file
```bash
# Check permissions
ls -la experimental_results/logs/

# Fix if needed
chmod 755 experimental_results/logs/
```

### Experiments stalling
```bash
# Check process
ps aux | grep python

# Check GPU
nvidia-smi

# Check log for last activity
tail -50 experimental_results/logs/run_all_experiments_*.log
```

---

## After Experiments Complete

### 1. Verify All Completed
```bash
bash scripts/check_experiment_status.sh
```

### 2. Compare Results
```bash
python scripts/compare_experiments.py \
    --experiments exp0_baseline exp1_preproc exp2_queue exp3_gaussian_nll exp4_gaussian_nce exp5_kl_anneal exp6_whiten \
    --output experimental_results/comparisons/ablation_comparison.md
```

### 3. Check Oracle Validation
```bash
for exp in exp{0..6}_*; do
    echo -n "${exp}: "
    jq -r '.oracle.passed' experimental_results/${exp}/evaluation/metrics.json 2>/dev/null || echo "N/A"
done
```

All should show `true` - if not, evaluation has bugs!

---

## Quick Reference

```bash
# Start experiments
bash scripts/run_all_experiments.sh 0

# Monitor (real-time)
bash scripts/monitor_experiments.sh

# Check status (periodic)
bash scripts/check_experiment_status.sh

# View log
tail -f experimental_results/logs/run_all_experiments_*.log

# GPU status
nvidia-smi
```

---

Good luck with your experiments on GPU20C-N5! 🚀
