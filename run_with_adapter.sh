#!/bin/bash
set -e

SUBJECT="subj01"
MODEL_ID="stabilityai/stable-diffusion-2-1"
ADAPTER_PATH="checkpoints/clip_adapter/${SUBJECT}/adapter.pt"
DIFF_STEPS=100
GUIDANCE=7.5

echo "=========================================================================="
echo "IMPROVED IMAGE GENERATION WITH ADAPTER"
echo "=========================================================================="
echo "Subject: ${SUBJECT}"
echo "Model: ${MODEL_ID}"
echo "Adapter: ${ADAPTER_PATH}"
echo "Diffusion Steps: ${DIFF_STEPS}"
echo "Guidance Scale: ${GUIDANCE}"
echo ""

# Generate images with adapter
echo "Step 1: Generating images with adapter..."
python scripts/decode_diffusion.py \
    --subject ${SUBJECT} \
    --model-id ${MODEL_ID} \
    --encoder-checkpoint checkpoints/mlp/${SUBJECT}/mlp.pt \
    --adapter-checkpoint ${ADAPTER_PATH} \
    --output outputs/recon/${SUBJECT}/improved_with_adapter \
    --num-inference-steps ${DIFF_STEPS} \
    --guidance-scale ${GUIDANCE} \
    --batch-size 8 \
    --device cuda

echo ""
echo "Step 2: Evaluating results..."
python -c "
import sys
sys.path.insert(0, 'src')
from fmri2img.eval.metrics import evaluate_reconstructions
import json

results = evaluate_reconstructions(
    'outputs/recon/${SUBJECT}/improved_with_adapter',
    'data/indices/test_nsd_index.csv',
    cache_path='cache/clip_embeddings'
)

print('\n' + '='*80)
print('IMPROVED RESULTS (WITH ADAPTER)')
print('='*80)
for metric, value in results.items():
    if isinstance(value, float):
        print(f'{metric}: {value:.4f}')
    else:
        print(f'{metric}: {value}')
print('='*80)

# Save report
import os
os.makedirs('outputs/reports/${SUBJECT}', exist_ok=True)
with open('outputs/reports/${SUBJECT}/improved_with_adapter_eval.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nReport saved to outputs/reports/${SUBJECT}/improved_with_adapter_eval.json')
"

echo ""
echo "Step 3: Comparing with baseline..."
python -c "
import json
import os

baseline_path = 'outputs/reports/${SUBJECT}/ridge_eval.json'
improved_path = 'outputs/reports/${SUBJECT}/improved_with_adapter_eval.json'

if os.path.exists(baseline_path) and os.path.exists(improved_path):
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(improved_path) as f:
        improved = json.load(f)
    
    print('\n' + '='*80)
    print('BASELINE vs IMPROVED (WITH ADAPTER)')
    print('='*80)
    print(f'CLIPScore:  {baseline.get(\"mean_clip_score\", 0):.4f} → {improved.get(\"mean_clip_score\", 0):.4f} ({improved.get(\"mean_clip_score\", 0) - baseline.get(\"mean_clip_score\", 0):+.4f})')
    print(f'R@1:        {baseline.get(\"recall_at_1\", 0):.4f} → {improved.get(\"recall_at_1\", 0):.4f} ({improved.get(\"recall_at_1\", 0) - baseline.get(\"recall_at_1\", 0):+.4f})')
    print(f'R@5:        {baseline.get(\"recall_at_5\", 0):.4f} → {improved.get(\"recall_at_5\", 0):.4f} ({improved.get(\"recall_at_5\", 0) - baseline.get(\"recall_at_5\", 0):+.4f})')
    print(f'R@10:       {baseline.get(\"recall_at_10\", 0):.4f} → {improved.get(\"recall_at_10\", 0):.4f} ({improved.get(\"recall_at_10\", 0) - baseline.get(\"recall_at_10\", 0):+.4f})')
    print(f'Mean Rank:  {baseline.get(\"mean_rank\", 0):.1f} → {improved.get(\"mean_rank\", 0):.1f}')
    print('='*80)
else:
    print('Baseline results not found. Run the baseline first.')
"

echo ""
echo "✅ Complete! Check outputs/recon/${SUBJECT}/improved_with_adapter for images"
