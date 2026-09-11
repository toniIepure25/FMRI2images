#!/bin/sh
# =============================================================================
# O2.3A-RD / O2.4R Phase-2 canonical execution wrapper (committed per FIX 7).
#
# Two SEALED stages with a mandatory access order; a single invocation runs ONE
# stage only. Stage A NEVER opens M>0 -- the RD result must be sealed and M0
# certified (as committed artifacts) before Stage B is launched separately.
#
#   stageA : data-free selftest -> RD fit (--fit, writes RD_SEAL.json)
#            -> certify M0==P_ZERO_RD (--certify-m0, writes M0_CERT.json). STOP.
#   stageB : O2.4R M>0 frontier (--o2-4r); refuses unless RD_SEAL.json AND a
#            valid M0_CERT.json are present.
#
# Frozen methodology is unchanged. RD config c1b2ddb0 / O2.4R config ad959446.
#
# Canonical execution environment (recorded for provenance):
#   image        nipreps/fmriprep:25.2.5
#   image digest sha256:15cbf8dcd17440d26ff5e80e9f7313f1cb3c54f13673f1ec4aed4465e8e12d77
#   node         k8s-worker-cpu-node-r770  (runai cpu-worker pool)
#   PVC          rd-data (local-path, RWO), mounted ONCE at /rd/data
#   --data       /rd/data/data   (raw NSD inputs: core_b2, nsd_imagery_b0, support)
#   --out        /rd/data/results/rd   (rd_results.json, o2_4r_results.json, state/, seals)
#   BLAS         OPENBLAS/MKL/OMP_NUM_THREADS=1 (determinism)
#   source       injected via ConfigMap `rd-src` at /cm (geometry/frontier/cohort/
#                s2_roi/smoke_pipeline .py + anchor_manifest.csv), reconstructed to /rd/repo
#
# Rebuild the source ConfigMap from a repo checkout with:
#   kubectl -n runai-romania-dev create configmap rd-src \
#     --from-file=geometry.py=src/fmri2img/mindcompiler/operator_o2_3a_rd/geometry.py \
#     --from-file=frontier.py=src/fmri2img/mindcompiler/operator_o2_3a_rd/frontier.py \
#     --from-file=cohort.py=src/fmri2img/mindcompiler/operator_o2_3a_rd/cohort.py \
#     --from-file=s2_roi.py=src/fmri2img/mindcompiler/roy_method_reproduction/s2_roi.py \
#     --from-file=smoke_pipeline.py=src/fmri2img/mindcompiler/roy_method_reproduction/smoke_pipeline.py \
#     --from-file=anchor_manifest.csv=artifacts/mindcompiler/operator_o2_3a/anchor_manifest.csv \
#     --from-file=o2_3a_rd_phase2.sh=scripts/mindcompiler/o2_3a_rd_phase2.sh --dry-run=client -o yaml | kubectl apply -f -
# =============================================================================
set -e
STAGE="${1:-stageA}"
CODE_VERSION="${CODE_VERSION:-unknown}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE
R=/rd/repo
DATA=/rd/data/data
OUT=/rd/data/results/rd
PKG=$R/src/fmri2img/mindcompiler
ROIS="${ROIS:-ventral,lateral}"

# reconstruct the importable package tree from the injected source (empty __init__ to avoid heavy deps)
mkdir -p $PKG/operator_o2_3a_rd $PKG/roy_method_reproduction $R/artifacts/mindcompiler/operator_o2_3a $OUT
: > $R/src/fmri2img/__init__.py
: > $PKG/__init__.py
: > $PKG/operator_o2_3a_rd/__init__.py
: > $PKG/roy_method_reproduction/__init__.py
cp /cm/geometry.py         $PKG/operator_o2_3a_rd/geometry.py
cp /cm/frontier.py         $PKG/operator_o2_3a_rd/frontier.py
cp /cm/cohort.py           $PKG/operator_o2_3a_rd/cohort.py
cp /cm/s2_roi.py           $PKG/roy_method_reproduction/s2_roi.py
cp /cm/smoke_pipeline.py   $PKG/roy_method_reproduction/smoke_pipeline.py
cp /cm/anchor_manifest.csv $R/artifacts/mindcompiler/operator_o2_3a/anchor_manifest.csv
COH=$PKG/operator_o2_3a_rd/cohort.py

echo "=== env ==="; python --version; python -c "import numpy,scipy,nibabel,h5py,pandas;print('deps',numpy.__version__)"
echo "=== data-free driver selftest (must pass before any real compute) ==="
python $COH --selftest --repo $R

case "$STAGE" in
  stageA)
    echo "=== STAGE A: O2.3A-RD FIT (no M>0) ==="
    python $COH --fit        --data $DATA --repo $R --out $OUT --rois "$ROIS" --code-version "$CODE_VERSION"
    echo "=== STAGE A: certify O2_4R_M0_EQUALS_O2_3A_RD ==="
    python $COH --certify-m0              --repo $R --out $OUT --rois "$ROIS"
    echo "=== STAGE A complete. RD sealed + M0 certified. STOP (Stage B is launched separately). ==="
    ls -la $OUT
    echo O2_3A_RD_STAGE_A_DONE
    ;;
  stageB)
    echo "=== STAGE B: O2.4R M>0 FRONTIER (guarded by RD_SEAL + M0_CERT) ==="
    python $COH --o2-4r                  --repo $R --out $OUT --rois "$ROIS"
    ls -la $OUT
    echo O2_4R_STAGE_B_DONE
    ;;
  *)
    echo "unknown STAGE '$STAGE' (use stageA | stageB)"; exit 2 ;;
esac
