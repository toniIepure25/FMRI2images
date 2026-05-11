#!/usr/bin/env bash
set -euo pipefail

# Fetch live backend artifacts from the Kubernetes pod.
# Usage: cd demo_thesis && bash scripts/fetch_live_backend_artifacts_from_pod.sh

KUBECONFIG="${KUBECONFIG:-$HOME/Downloads/antoniu_iepure.yaml}"
NS="runai-romania-dev"
export KUBECONFIG

echo "=== Cortex2Canvas: fetch backend artifacts from K8s pod ==="
echo "KUBECONFIG=$KUBECONFIG"
echo "Namespace=$NS"

# Discover pod
POD=$(kubectl get pods -n "$NS" -o name 2>/dev/null | grep -Ei "orchestraiq|jupyter" | head -1 | sed 's|pod/||')
if [ -z "$POD" ]; then
    echo "ERROR: No jupyter/orchestraiq pod found in namespace $NS"
    exit 1
fi
echo "Pod=$POD"
echo

# Create local directories
mkdir -p local_backend_data/subj01
mkdir -p local_backend_data/clip
mkdir -p local_backend_data/checkpoints

# Copy fMRI features
if [ -f "local_backend_data/subj01/fmri_features.npy" ]; then
    echo "[SKIP] fmri_features.npy already exists ($(du -h local_backend_data/subj01/fmri_features.npy | cut -f1))"
else
    echo "[COPY] fmri_features.npy (~1.8 GB)..."
    kubectl cp "$NS/$POD:/home/jovyan/local-data/fmri2images_moved/cache/preextracted/subject=subj01/fmri_features.npy" \
        "local_backend_data/subj01/fmri_features.npy"
    echo "       Done: $(du -h local_backend_data/subj01/fmri_features.npy | cut -f1)"
fi

# Copy trial index
if [ -f "local_backend_data/subj01/index.parquet" ]; then
    echo "[SKIP] index.parquet already exists"
else
    echo "[COPY] index.parquet..."
    kubectl cp "$NS/$POD:/home/jovyan/work/FMRI2images/data/indices/nsd_index/subject=subj01/index.parquet" \
        "local_backend_data/subj01/index.parquet"
    echo "       Done: $(du -h local_backend_data/subj01/index.parquet | cut -f1)"
fi

# Copy CLIP gallery
if [ -f "local_backend_data/clip/clip.parquet" ]; then
    echo "[SKIP] clip.parquet already exists"
else
    echo "[COPY] clip.parquet..."
    kubectl cp "$NS/$POD:/home/jovyan/work/FMRI2images/outputs/clip_cache/clip.parquet" \
        "local_backend_data/clip/clip.parquet"
    echo "       Done: $(du -h local_backend_data/clip/clip.parquet | cut -f1)"
fi

echo
echo "=== Checkpoint discovery ==="
echo "Available checkpoint_best.pt files on pod:"
kubectl exec -n "$NS" "$POD" -- find /home/jovyan/work/FMRI2images/experimental_results \
    -name "checkpoint_best.pt" -exec ls -lh {} \; 2>/dev/null || true

echo
echo "To extract a model-only checkpoint (stripping optimizer state),"
echo "run on the pod via kubectl exec:"
echo ""
echo "  kubectl exec -n $NS $POD -- python3 -c \""
echo "  import torch, os"
echo "  ckpt = torch.load('<PATH>/checkpoint_best.pt', map_location='cpu', weights_only=False)"
echo "  slim = {k: ckpt[k] for k in ('model_state_dict','model_config','config','subject','epoch') if k in ckpt}"
echo "  torch.save(slim, '/tmp/model_only.pt')"
echo "  print(f'Saved: {os.path.getsize(chr(47)+\"tmp\"+chr(47)+\"model_only.pt\")/(1024**2):.0f} MB')"
echo "  \""
echo ""
echo "Then copy:"
echo "  kubectl cp $NS/$POD:/tmp/model_only.pt local_backend_data/checkpoints/model_only.pt"
echo

if [ -f "local_backend_data/checkpoints/V62a_model_only.pt" ]; then
    echo "[OK] Checkpoint found: $(du -h local_backend_data/checkpoints/V62a_model_only.pt | cut -f1)"
else
    echo "[INFO] No checkpoint in local_backend_data/checkpoints/ yet."
    echo "       Follow the instructions above to extract and copy one."
fi

echo
echo "=== Summary ==="
du -sh local_backend_data/subj01/fmri_features.npy 2>/dev/null || echo "  fmri_features.npy: MISSING"
du -sh local_backend_data/subj01/index.parquet 2>/dev/null || echo "  index.parquet: MISSING"
du -sh local_backend_data/clip/clip.parquet 2>/dev/null || echo "  clip.parquet: MISSING"
du -sh local_backend_data/checkpoints/*.pt 2>/dev/null || echo "  checkpoint: MISSING"
