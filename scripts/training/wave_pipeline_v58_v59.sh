#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Wave Pipeline: V58a -> V59a -> Fusion Evaluation
#
# Orchestrates Wave 1 + Wave 2a training and evaluation.
# Designed for H100 pod with persistent storage at /home/jovyan/work.
#
# Usage:
#   nohup bash scripts/training/wave_pipeline_v58_v59.sh > wave_pipeline.log 2>&1 &
###############################################################################

REPO="/home/jovyan/work/FMRI2images"
SUBJECT="subj01"
GPU=0

cd "$REPO"

# Environment
set -a && source .env 2>/dev/null || true && set +a
export CUDA_VISIBLE_DEVICES=$GPU
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR="${REPO}/cache/tmp"
export TORCH_HOME="${REPO}/cache/torch"
export XDG_CACHE_HOME="${REPO}/cache/xdg"
mkdir -p "$TMPDIR" "$TORCH_HOME" "$XDG_CACHE_HOME"

pip install -e ".[train]" --quiet 2>/dev/null || true

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

###############################################################################
# WAVE 1: V58a -- Improved 197K-D Expert
###############################################################################
V58A_DIR="$REPO/experimental_results/V58a_improved_197k/$SUBJECT"
V58A_CKPT="$V58A_DIR/checkpoints/checkpoint_best.pt"
V58A_CFG="configs/experiments/V58a_improved_197k.yaml"

if [ -f "$V58A_CKPT" ]; then
    log "WAVE 1: V58a checkpoint already exists, skipping training."
else
    log "WAVE 1: Starting V58a training..."
    python scripts/training/train_unified.py \
        --config "$V58A_CFG" \
        2>&1 | tee "$REPO/V58a_training.log"
    log "WAVE 1: V58a training complete."
fi

# Verify V58a produced shared1000 predictions
if [ ! -f "$V58A_DIR/metrics/shared1000_predictions.npy" ] && \
   [ ! -f "$V58A_DIR/metrics/shared1000_predictions_compact.npy" ]; then
    log "ERROR: V58a did not produce shared1000 predictions."
    exit 1
fi
log "WAVE 1: V58a shared1000 predictions verified."

###############################################################################
# WAVE 1 EVALUATION: V58a + N1v28a fusion sweep
###############################################################################
log "WAVE 1 EVAL: Running fusion sweep V58a + N1v28a..."
python -c "
import numpy as np
from pathlib import Path

base = Path('$REPO/experimental_results')

def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

def csls(p, g, k=10):
    s = norm(p) @ norm(g).T
    tq = np.sort(s, axis=1)[:, -k:].mean(axis=1, keepdims=True)
    tg = np.sort(s, axis=0)[-k:, :].mean(axis=0, keepdims=True)
    return 2*s - tq - tg

def r_at_k(scores, k=1):
    n = scores.shape[0]
    ranks = (scores >= scores[np.arange(n), np.arange(n)][:, None]).sum(axis=1)
    return float((ranks <= k).mean())

def minmax(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-8)

# Load V58a predictions (197K-D)
v58_dir = base / 'V58a_improved_197k/$SUBJECT/metrics'
for name in ['shared1000_predictions.npy', 'shared1000_predictions_compact.npy']:
    p = v58_dir / name
    if p.exists():
        v58_pred = np.load(p)
        break
for name in ['shared1000_ground_truth.npy', 'shared1000_ground_truth_compact.npy']:
    p = v58_dir / name
    if p.exists():
        v58_gt = np.load(p)
        break

# Load N1v28a predictions (197K-D)
n1_dir = base / 'N1v28a_dual_head/$SUBJECT/metrics'
n1_pred = np.load(n1_dir / 'shared1000_predictions_197k.npy')
n1_gt = np.load(n1_dir / 'shared1000_ground_truth.npy')

print(f'V58a: {v58_pred.shape}, N1v28a: {n1_pred.shape}')

# V58a standalone
csls_v58 = csls(v58_pred, v58_gt)
print(f'V58a standalone CSLS R@1 = {r_at_k(csls_v58)*100:.1f}%')

# N1v28a standalone
csls_n1 = csls(n1_pred, n1_gt)
print(f'N1v28a standalone CSLS R@1 = {r_at_k(csls_n1)*100:.1f}%')

# Fusion sweep
csls_v58_n = minmax(csls_v58)
csls_n1_n = minmax(csls_n1)

best_r1 = 0
best_alpha = 0
print()
print('Fusion sweep (alpha = V58a weight):')
for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    fused = alpha * csls_v58_n + (1-alpha) * csls_n1_n
    r1 = r_at_k(fused)
    r5 = r_at_k(fused, 5)
    marker = ''
    if r1 > best_r1:
        best_r1 = r1
        best_alpha = alpha
        marker = ' <-- BEST'
    print(f'  alpha={alpha:.1f}: R@1={r1*100:.1f}% R@5={r5*100:.1f}%{marker}')

print(f'Best fusion: alpha={best_alpha:.1f}, R@1={best_r1*100:.1f}%')

# Error analysis
ranks_v58 = (csls_v58 >= csls_v58[np.arange(1000), np.arange(1000)][:, None]).sum(axis=1)
ranks_n1 = (csls_n1 >= csls_n1[np.arange(1000), np.arange(1000)][:, None]).sum(axis=1)
v58_ok = ranks_v58 <= 1
n1_ok = ranks_n1 <= 1
print(f'Error analysis: both={int((v58_ok&n1_ok).sum())} v58_only={int((v58_ok&~n1_ok).sum())} n1_only={int((~v58_ok&n1_ok).sum())} neither={int((~v58_ok&~n1_ok).sum())} oracle={int((v58_ok|n1_ok).sum())}')
" 2>&1 | tee "$REPO/wave1_fusion_eval.log"

log "WAVE 1 EVAL: Complete."

###############################################################################
# WAVE 2a: V59a -- Retrieval-Only 768-D Expert
###############################################################################
V59A_DIR="$REPO/experimental_results/V59a_retrieval_only_768d/$SUBJECT"
V59A_CKPT="$V59A_DIR/checkpoints/checkpoint_best.pt"
V59A_CFG="configs/experiments/V59a_retrieval_only_768d.yaml"

if [ -f "$V59A_CKPT" ]; then
    log "WAVE 2a: V59a checkpoint already exists, skipping training."
else
    log "WAVE 2a: Starting V59a training..."
    python scripts/training/train_unified.py \
        --config "$V59A_CFG" \
        2>&1 | tee "$REPO/V59a_training.log"
    log "WAVE 2a: V59a training complete."
fi

###############################################################################
# WAVE 2 EVALUATION: Multi-expert fusion
###############################################################################
log "WAVE 2 EVAL: Running multi-expert fusion sweep..."
python -c "
import numpy as np
from pathlib import Path

base = Path('$REPO/experimental_results')

def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

def csls(p, g, k=10):
    s = norm(p) @ norm(g).T
    tq = np.sort(s, axis=1)[:, -k:].mean(axis=1, keepdims=True)
    tg = np.sort(s, axis=0)[-k:, :].mean(axis=0, keepdims=True)
    return 2*s - tq - tg

def r_at_k(scores, k=1):
    n = scores.shape[0]
    ranks = (scores >= scores[np.arange(n), np.arange(n)][:, None]).sum(axis=1)
    return float((ranks <= k).mean())

def minmax(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-8)

def load_expert(name):
    d = base / name / '$SUBJECT/metrics'
    for pname in ['shared1000_predictions.npy', 'shared1000_predictions_197k.npy', 'shared1000_predictions_compact.npy']:
        p = d / pname
        if p.exists():
            pred = np.load(p)
            break
    else:
        return None, None
    for gname in ['shared1000_ground_truth.npy', 'shared1000_ground_truth_compact.npy']:
        p = d / gname
        if p.exists():
            gt = np.load(p)
            break
    return pred, gt

experts = {}
for name in ['V58a_improved_197k', 'N1v28a_dual_head', 'V59a_retrieval_only_768d', 'V57a_roi_transformer_dual_head']:
    pred, gt = load_expert(name)
    if pred is not None:
        csls_mat = csls(pred, gt)
        r1 = r_at_k(csls_mat)
        experts[name] = {'csls': csls_mat, 'csls_norm': minmax(csls_mat), 'r1': r1}
        print(f'{name}: dim={pred.shape[1]}, CSLS R@1={r1*100:.1f}%')

if len(experts) < 2:
    print('Not enough experts for fusion')
else:
    # Pairwise and 3-way fusion
    names = list(experts.keys())
    print()

    # Best 3-expert sweep if available
    if len(experts) >= 3:
        print('3-expert sweep (top 3 by individual R@1):')
        sorted_experts = sorted(experts.items(), key=lambda x: x[1]['r1'], reverse=True)
        top3 = sorted_experts[:3]
        n0, n1, n2 = top3[0][0], top3[1][0], top3[2][0]
        s0, s1, s2 = top3[0][1]['csls_norm'], top3[1][1]['csls_norm'], top3[2][1]['csls_norm']

        best_r1 = 0
        best_w = (0,0,0)
        for w0 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for w1 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                w2 = round(1.0 - w0 - w1, 1)
                if w2 < 0: continue
                fused = w0 * s0 + w1 * s1 + w2 * s2
                r1 = r_at_k(fused)
                if r1 > best_r1:
                    best_r1 = r1
                    best_w = (w0, w1, w2)

        print(f'  Best 3-expert: {n0}={best_w[0]:.1f}, {n1}={best_w[1]:.1f}, {n2}={best_w[2]:.1f} -> R@1={best_r1*100:.1f}%')

    # Oracle
    all_correct = None
    for name, data in experts.items():
        ranks = (data['csls'] >= data['csls'][np.arange(1000), np.arange(1000)][:, None]).sum(axis=1)
        correct = ranks <= 1
        if all_correct is None:
            all_correct = correct.copy()
        else:
            all_correct = all_correct | correct
    if all_correct is not None:
        print(f'  Oracle (any expert correct): {all_correct.sum()} / 1000 = {all_correct.mean()*100:.1f}%')
" 2>&1 | tee "$REPO/wave2_fusion_eval.log"

log "WAVE 2 EVAL: Complete."

###############################################################################
# DONE
###############################################################################
log "PIPELINE COMPLETE. Check wave1_fusion_eval.log and wave2_fusion_eval.log."
