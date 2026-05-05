#!/usr/bin/env bash
# =============================================================================
# Cortex2Canvas — Deploy Live Inference Backend to K8s Pod
#
# Prerequisites:
#   1. VPN active
#   2. KUBECONFIG set: export KUBECONFIG=~/Downloads/antoniu_iepure.yaml
#   3. kubectl installed
#
# Usage:
#   chmod +x deploy_to_pod.sh
#   ./deploy_to_pod.sh
#
# This script:
#   1. Finds the running JupyterHub pod
#   2. Copies the backend code to the pod
#   3. Installs dependencies
#   4. Starts the inference server with the best checkpoint
#   5. Sets up port-forward so localhost:8000 → pod:8000
# =============================================================================

set -euo pipefail

NAMESPACE="runai-romania-dev"
KUBECONFIG_PATH="${KUBECONFIG:-$HOME/Downloads/antoniu_iepure.yaml}"
REMOTE_BASE="/home/jovyan/work/FMRI2images"

# Auto-detect the pod name
echo "=== Finding JupyterHub pod ==="
export KUBECONFIG="$KUBECONFIG_PATH"
POD_NAME=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$POD_NAME" ]; then
    echo "ERROR: No pod found in namespace $NAMESPACE"
    echo "       Is the VPN active? Is the pod running?"
    exit 1
fi
echo "Pod: $POD_NAME"

# Find best checkpoint
echo ""
echo "=== Finding best checkpoint ==="
BEST_CKPT=$(kubectl exec -n "$NAMESPACE" "$POD_NAME" -- bash -c '
    cd /home/jovyan/work/FMRI2images
    # Look for the most recent checkpoint_best.pt
    find experimental_results -name "checkpoint_best.pt" -type f 2>/dev/null | \
        xargs ls -t 2>/dev/null | head -1
' 2>/dev/null)

if [ -z "$BEST_CKPT" ]; then
    echo "WARNING: No checkpoint_best.pt found."
    echo "Available checkpoints:"
    kubectl exec -n "$NAMESPACE" "$POD_NAME" -- bash -c '
        cd /home/jovyan/work/FMRI2images
        find experimental_results -name "*.pt" -type f 2>/dev/null | head -20
    '
    echo ""
    read -rp "Enter checkpoint path (relative to repo root): " BEST_CKPT
fi
echo "Using checkpoint: $BEST_CKPT"

# Copy backend to pod
echo ""
echo "=== Copying backend code to pod ==="
kubectl cp backend/main.py "$NAMESPACE/$POD_NAME:$REMOTE_BASE/demo_thesis/backend/main.py"
echo "Backend code copied."

# Install dependencies and start server
echo ""
echo "=== Starting inference server on pod ==="
kubectl exec -n "$NAMESPACE" "$POD_NAME" -- bash -c "
    cd $REMOTE_BASE
    set -a && source .env 2>/dev/null || true && set +a
    export CUDA_VISIBLE_DEVICES=0

    # Install backend dependencies
    pip install fastapi uvicorn sse-starlette Pillow --quiet 2>&1 | tail -1

    # Kill any existing backend
    pkill -f 'demo_thesis/backend/main.py' 2>/dev/null || true
    sleep 1

    # Start backend with nohup
    nohup python demo_thesis/backend/main.py \\
        --checkpoint '$BEST_CKPT' \\
        --subject subj01 \\
        --port 8000 \\
        > demo_backend.log 2>&1 &

    echo 'Backend PID:' \$!
    echo 'Waiting for startup...'
    sleep 5
    tail -5 demo_backend.log
" 2>&1

echo ""
echo "=== Setting up port-forward ==="
echo "Starting kubectl port-forward (Ctrl+C to stop)..."
echo ""
echo "Once ready, open http://localhost:3000/pipeline in your browser."
echo "The frontend will auto-detect the backend at http://localhost:8000"
echo ""

kubectl port-forward -n "$NAMESPACE" "$POD_NAME" 8000:8000
