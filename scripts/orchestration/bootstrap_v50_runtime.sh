#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the current Jupyter pod so V50 can run even after a pod restart.
# Runtime Python deps are installed into local-data to avoid using ephemeral
# container storage and to persist across pod recreations.

REPO_ROOT="${REPO_ROOT:-/home/jovyan/work/FMRI2images}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/home/jovyan/local-data}"
RUNTIME_PYTHON_ROOT="${RUNTIME_PYTHON_ROOT:-${LOCAL_DATA_ROOT}/python_runtime}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${LOCAL_DATA_ROOT}/pip_cache}"
V50_ROOT="${V50_ROOT:-${LOCAL_DATA_ROOT}/experiment_archive/V50_all8_dense_vmf_hybrid}"
LOG_ROOT="${LOG_ROOT:-${V50_ROOT}/run_logs}"
NSD_STIM_DIR="${NSD_STIM_DIR:-/home/jovyan/work/data/nsd/nsddata_stimuli/stimuli/nsd}"

mkdir -p "${RUNTIME_PYTHON_ROOT}" "${PIP_CACHE_DIR}" "${LOG_ROOT}" "${NSD_STIM_DIR}"

cd "${REPO_ROOT}"
git config --global --add safe.directory "${REPO_ROOT}" >/dev/null 2>&1 || true

# Recreate the key compatibility paths expected by the V50 config.
mkdir -p experimental_results
ln -sfn "${LOCAL_DATA_ROOT}/experiment_archive/V35r_rebuild_from_V33b" \
  experimental_results/V35_legacy_teacher_distill
ln -sfn "${V50_ROOT}" experimental_results/V50_all8_dense_vmf_hybrid
mkdir -p /home/jovyan/work/data
ln -sfn "${REPO_ROOT}" /home/jovyan/work/data/FMRI2images

export PIP_CACHE_DIR
export PYTHONPATH="${RUNTIME_PYTHON_ROOT}:src:${PYTHONPATH:-}"

python3 - <<'PY'
import importlib
mods = ["s3fs", "nibabel", "h5py"]
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)
print("MISSING=" + ",".join(missing))
PY

missing_modules="$(
python3 - <<'PY'
import importlib
mods = ["s3fs", "nibabel", "h5py"]
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)
print(" ".join(missing))
PY
)"

if [[ -n "${missing_modules}" ]]; then
  python3 -m pip install --target "${RUNTIME_PYTHON_ROOT}" ${missing_modules}
fi

echo "Bootstrap complete."
echo "Repo: ${REPO_ROOT}"
echo "Runtime Python: ${RUNTIME_PYTHON_ROOT}"
echo "V50 logs: ${LOG_ROOT}"
echo "NSD stimuli dir: ${NSD_STIM_DIR}"
