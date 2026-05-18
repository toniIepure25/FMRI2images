#!/usr/bin/env bash
# Run triple_fusion_shared1000_fixed.py **on the pod** (reads ~1.6 GB from PVC)
# and copy back only a small JSON (~1 KB). Use when ``kubectl cp`` of large npy fails.
#
# Usage:
#   export KUBECONFIG=~/Downloads/antoniu_iepure.yaml
#   export POD_NAME=orchestraiq-jupyter-<suffix>
#   export K8S_NS=runai-romania-dev
#   bash scripts/evaluation/fetch_triple_fusion_json_from_pod.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
POD_NAME="${POD_NAME:-}"
K8S_NS="${K8S_NS:-runai-romania-dev}"
REMOTE_REPO="${REMOTE_FMRI2IMAGES:-/home/jovyan/work/FMRI2images}"
LOCAL_OUT="${ROOT}/experimental_results/fusion_triple_86/triple_fusion_shared1000_86.json"
SCRIPT_LOCAL="${ROOT}/scripts/evaluation/triple_fusion_shared1000_fixed.py"
REMOTE_JSON="/tmp/triple_fusion_shared1000_86.json"
REMOTE_SCRIPT="/tmp/triple_fusion_shared1000_fixed.py"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl not found." >&2
  exit 1
fi
if [[ -z "${POD_NAME}" ]]; then
  echo "Set POD_NAME to your Jupyter pod." >&2
  exit 1
fi

mkdir -p "$(dirname "${LOCAL_OUT}")"

echo "Upload script..."
kubectl cp "${SCRIPT_LOCAL}" "${K8S_NS}/${POD_NAME}:${REMOTE_SCRIPT}"

echo "Run fusion on pod (CPU/GPU torch)..."
kubectl exec -n "${K8S_NS}" "${POD_NAME}" -- \
  python3 "${REMOTE_SCRIPT}" \
  --use-repo-metrics \
  --repo-root "${REMOTE_REPO}" \
  --output-json "${REMOTE_JSON}"

echo "Download JSON -> ${LOCAL_OUT}"
kubectl cp "${K8S_NS}/${POD_NAME}:${REMOTE_JSON}" "${LOCAL_OUT}"

echo "Done."
cat "${LOCAL_OUT}"
