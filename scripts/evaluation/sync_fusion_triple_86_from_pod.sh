#!/usr/bin/env bash
# Pull the six NumPy files for triple_fusion_shared1000_fixed.py via a **streamed tar**
# from the pod (more reliable than ``kubectl cp`` on multi-hundred-MB single files).
#
# If this fails mid-stream (API ``connection reset``), use instead:
#   bash scripts/evaluation/fetch_triple_fusion_json_from_pod.sh
# which runs fusion **on the pod** and copies only a small JSON to this machine.
#
# Usage:
#   export KUBECONFIG=~/Downloads/antoniu_iepure.yaml
#   export POD_NAME=orchestraiq-jupyter-<suffix>
#   export K8S_NS=runai-romania-dev
#   bash scripts/evaluation/sync_fusion_triple_86_from_pod.sh
#
# Remote repo root on pod:
#   /home/jovyan/work/FMRI2images

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${ROOT}/experimental_results/fusion_triple_86"
EXTRACT="${DEST}/_pod_tar_extract"
REMOTE_ROOT="${REMOTE_FMRI2IMAGES:-/home/jovyan/work/FMRI2images}/experimental_results"

POD_NAME="${POD_NAME:-}"
K8S_NS="${K8S_NS:-runai-romania-dev}"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl not found." >&2
  exit 1
fi

if [[ -z "${POD_NAME}" ]]; then
  echo "Set POD_NAME to your Jupyter pod, e.g.:" >&2
  echo "  export POD_NAME=orchestraiq-jupyter-54644cff87-nxd9x" >&2
  exit 1
fi

mkdir -p "${DEST}/v61a_mctta16" "${DEST}/v62a" "${DEST}/v66a"
rm -rf "${EXTRACT}"
mkdir -p "${EXTRACT}"

echo "Streaming tar from pod ${K8S_NS}/${POD_NAME} (this is ~1.6 GB; can take several minutes)..."
kubectl exec -n "${K8S_NS}" "${POD_NAME}" -- bash -c "cd '${REMOTE_ROOT}' && tar cf - \
  'V61a_finetune_difflr/subj01/metrics/shared1000_predictions_mctta16.npy' \
  'V61a_finetune_difflr/subj01/metrics/shared1000_ground_truth.npy' \
  'V62a_cls_retrieval_768d/subj01/metrics/shared1000_predictions.npy' \
  'V62a_cls_retrieval_768d/subj01/metrics/shared1000_ground_truth.npy' \
  'V66a_roi_pretrain/subj01/metrics/shared1000_predictions.npy' \
  'V66a_roi_pretrain/subj01/metrics/shared1000_ground_truth.npy'" \
  | tar xf - -C "${EXTRACT}"

R="${EXTRACT}"

mv -f "${R}/V61a_finetune_difflr/subj01/metrics/shared1000_predictions_mctta16.npy" "${DEST}/v61a_mctta16/"
mv -f "${R}/V61a_finetune_difflr/subj01/metrics/shared1000_ground_truth.npy" "${DEST}/v61a_mctta16/"
mv -f "${R}/V62a_cls_retrieval_768d/subj01/metrics/shared1000_predictions.npy" "${DEST}/v62a/"
mv -f "${R}/V62a_cls_retrieval_768d/subj01/metrics/shared1000_ground_truth.npy" "${DEST}/v62a/"
mv -f "${R}/V66a_roi_pretrain/subj01/metrics/shared1000_predictions.npy" "${DEST}/v66a/"
mv -f "${R}/V66a_roi_pretrain/subj01/metrics/shared1000_ground_truth.npy" "${DEST}/v66a/"

rm -rf "${EXTRACT}"

echo ""
echo "Verifying sizes (expect predictions_mctta16 ~790 MiB)..."
ls -lh "${DEST}/v61a_mctta16" "${DEST}/v62a" "${DEST}/v66a"

echo ""
echo "Run fusion:"
echo "  cd \"${ROOT}\" && python3 scripts/evaluation/triple_fusion_shared1000_fixed.py"
