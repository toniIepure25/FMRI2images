#!/usr/bin/env bash
# Generate triple-fusion qualitative query panels ON the pod and copy the
# small PNG outputs back to this machine.
#
# Usage:
#   export KUBECONFIG=~/Downloads/antoniu_iepure.yaml
#   export POD_NAME=orchestraiq-jupyter-54644cff87-nxd9x
#   export K8S_NS=runai-romania-dev
#   bash scripts/evaluation/fetch_triple_fusion_panels_from_pod.sh
#
# Output: thesis/v62_thesis/query_{8262,21279,55649,8509}.png  (overwritten)
#         thesis/v62_thesis/panel_metadata.json                 (new)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
POD_NAME="${POD_NAME:-}"
K8S_NS="${K8S_NS:-runai-romania-dev}"
REMOTE_REPO="${REMOTE_FMRI2IMAGES:-/home/jovyan/work/FMRI2images}"

SCRIPT_FUSION="${ROOT}/scripts/evaluation/triple_fusion_shared1000_fixed.py"
SCRIPT_PANELS="${ROOT}/scripts/evaluation/make_triple_fusion_query_panels.py"

REMOTE_FUSION="/tmp/_tfusion_fixed.py"
REMOTE_PANELS="/tmp/_make_panels.py"
REMOTE_TOP1="/tmp/fusion_top1"
REMOTE_OUT="/tmp/fusion_panels"
REMOTE_NSD_IDS="${REMOTE_REPO}/experimental_results/V61a_finetune_difflr/subj01/metrics/shared1000_nsd_ids.npy"

LOCAL_THESIS="${ROOT}/thesis/v62_thesis"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl not found." >&2; exit 1
fi
if [[ -z "${POD_NAME}" ]]; then
  echo "Set POD_NAME to your Jupyter pod, e.g.:" >&2
  echo "  export POD_NAME=orchestraiq-jupyter-54644cff87-nxd9x" >&2
  exit 1
fi

echo "=== Step 1/4: Upload scripts to pod ==="
kubectl cp "${SCRIPT_FUSION}" "${K8S_NS}/${POD_NAME}:${REMOTE_FUSION}"
kubectl cp "${SCRIPT_PANELS}" "${K8S_NS}/${POD_NAME}:${REMOTE_PANELS}"

echo "=== Step 2/4: Run fusion + save top-1 IDs ==="
kubectl exec -n "${K8S_NS}" "${POD_NAME}" -- \
  python3 "${REMOTE_FUSION}" \
    --use-repo-metrics \
    --repo-root "${REMOTE_REPO}" \
    --save-top1-dir "${REMOTE_TOP1}" \
    --output-json "${REMOTE_TOP1}/metrics.json"

echo ""
echo "=== Step 3/4: Render query panels ==="
kubectl exec -n "${K8S_NS}" "${POD_NAME}" -- \
  python3 "${REMOTE_PANELS}" \
    --top1-dir "${REMOTE_TOP1}" \
    --nsd-ids-npy "${REMOTE_NSD_IDS}" \
    --output-dir "${REMOTE_OUT}" \
    --queries 8262 21279 55649 8509

echo ""
echo "=== Step 4/4: Download PNGs + metadata ==="
mkdir -p "${LOCAL_THESIS}"
for nsd_id in 8262 21279 55649 8509; do
  echo "  query_${nsd_id}.png"
  kubectl cp "${K8S_NS}/${POD_NAME}:${REMOTE_OUT}/query_${nsd_id}.png" \
    "${LOCAL_THESIS}/query_${nsd_id}.png"
done
kubectl cp "${K8S_NS}/${POD_NAME}:${REMOTE_OUT}/panel_metadata.json" \
  "${LOCAL_THESIS}/panel_metadata.json"

echo ""
echo "Done. Panels saved to ${LOCAL_THESIS}/"
echo "Metadata:"
cat "${LOCAL_THESIS}/panel_metadata.json"
