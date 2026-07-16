# 09 — Execution Runbook

**Verified working as of 2026-07-16.** Every command below was executed this session unless
marked PLANNED.

---

## Environment facts (verified)

| Fact | Value |
|---|---|
| Local workspace | `D:\ComputaCenter\FMRI2images` |
| Local torch | **2.7.1+cpu — CUDA False.** CPU probes only; no local training |
| Pod | `orchestraiq-jupyter-54644cff87-gz6n2` (Running, H100 80GB, idle) |
| kubeconfig | `C:\Users\ComputaCenter\Downloads\antoniu_iepure.yaml` |
| Pod repo | `/home/jovyan/work/FMRI2images` · pod torch 2.11.0+cu128 · Python 3.13.13 |
| NSD root (pod) | `/home/jovyan/work/data/nsd` |
| Stim info | `/home/jovyan/work/data/nsd/nsddata/experiments/nsd/nsd_stim_info_merged.csv` |

The Bash tool (Git Bash) works fine for kubectl; `export KUBECONFIG=...` then plain
`kubectl`. PowerShell is available separately but was not needed.

## Tests

```bash
python -m pytest tests/test_pcd_gradient_flow.py -v      # 9/9 passing — Gate 0 guards
python -m pytest tests/ -v                               # full suite
```

`tests/test_pcd_gradient_flow.py` pins: per-level kappa heads get no gradient; prediction
flows low→high; level 3 bypasses the chain; all 4 ablation arms construct and run.
**If `test_per_level_kappa_heads_are_unsupervised` ever fails, someone added a kappa
objective — update the claim registry, do not "fix" the test.**

## Pod inspection (read-only, safe)

```bash
export KUBECONFIG="C:/Users/ComputaCenter/Downloads/antoniu_iepure.yaml"
POD=orchestraiq-jupyter-54644cff87-gz6n2

kubectl get pods | grep -i jupyter                       # confirm pod name (it changes on restart)
kubectl exec $POD -- bash -c 'cd /home/jovyan/work/FMRI2images && ps aux | grep train_unified | grep -v grep'
kubectl exec $POD -- nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
kubectl exec $POD -- bash -c 'cd /home/jovyan/work/FMRI2images && git rev-parse HEAD && git status --porcelain | head'
kubectl exec $POD -- bash -c 'cd /home/jovyan/work/FMRI2images && tail -25 pcd_v4_resumed2.log'
```

## Provenance reconciliation (the check that caught F-005)

```bash
# Pod-side digests
kubectl exec $POD -- bash -c 'cd /home/jovyan/work/FMRI2images && sha256sum \
  src/fmri2img/models/predictive_cortical_decoder.py \
  src/fmri2img/models/unified_model.py \
  scripts/training/train_unified.py \
  configs/experiments/PCD_v4_8subject.yaml'

# Local digests — must match, and DO as of c65e834
cd "D:/ComputaCenter/FMRI2images" && sha256sum <same four paths>
```

**Never trust `git rev-parse` on the pod** — its tree is 4 commits behind with hand-copied
files and matches no commit. Content hashes are the authority.

## Leakage audit (reproduces T10)

```bash
kubectl exec $POD -- bash -c 'cd /home/jovyan/work/FMRI2images && python3 -c "
import json, pandas as pd
df = pd.read_csv(\"/home/jovyan/work/data/nsd/nsddata/experiments/nsd/nsd_stim_info_merged.csv\")
shared = set(int(x) for x in df[df[\"shared1000\"]==True][\"nsdId\"])
d = json.load(open(\"experimental_results/PCD_v4_8subject/subj01/split.json\"))
tr, va = set(d[\"train_nsd_ids\"]), set(d[\"val_nsd_ids\"])
print(\"train n val        :\", len(tr & va))
print(\"train n SHARED1000 :\", len(tr & shared))
print(\"val   n SHARED1000 :\", len(va & shared))
"'
```
Expected: `0 / 0 / 0`. **PLANNED:** promote this to `tests/test_split_leakage.py`.

## PLANNED — D-004 checkpoint preservation (next session, first action)

```bash
# 1. Hash both checkpoints
kubectl exec $POD -- bash -c 'cd /home/jovyan/work/FMRI2images && \
  sha256sum experimental_results/PCD_v4_8subject/subj01/checkpoint_{best,last}.pt'

# 2. Dry-load with EXPLICIT key diff — strict=False hides mismatches (R-08)
#    Assert missing keys contain only known buffers; assert unexpected == empty.

# 3. Copy small text artifacts off-pod (safe to version; do NOT copy the 2.68GB weights)
kubectl cp $POD:/home/jovyan/work/FMRI2images/experimental_results/PCD_v4_8subject/subj01/split.json ./...
```

## Training (PLANNED — currently PROHIBITED by D-001)

Do **not** run this until Gate 1 clears. Recorded for completeness only:

```bash
kubectl exec $POD -- bash -c 'cd /home/jovyan/work/FMRI2images && set -a && source .env && set +a && \
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TQDM_DISABLE=1 PYTHONUNBUFFERED=1 && \
  nohup python3 -u scripts/training/train_unified.py --config <CONFIG> --gpu 0 \
  > <LOG>.log 2>&1 & echo STARTED'
```

Constraints verified as real: `TQDM_DISABLE=1` (NFS log flooding), `PYTHONUNBUFFERED=1`,
`python3 -u`, `source .env` first. `bottleneck_dim` **must stay unset** (hung v3 at epoch 2).

## Analysis scripts — status

| Script | Status |
|---|---|
| `scripts/analysis/pcd_neuroscience_analysis.py` | **DO NOT RUN FOR EVIDENCE.** Reads untrained `level_kappas` (F-002) and treats residual magnitude as information (rule 8). Rewrite required. |
| `scripts/analysis/run_pcd_ablations.py` | Not yet audited. `ablation_mode=random` hardcodes seed 42 — single shuffle, needs multi-seed before H2. |
| `scripts/utils/smoke_test_pcd.py` | Works; the only consumer of `_last_pcd_extras`. |

## Recovery

- **Pod name changed** → re-resolve with `kubectl get pods | grep -i jupyter`; update this file.
- **Stale handoff** → re-verify against the pod before acting. `HANDOFF_CONTEXT.md` was ~90
  epochs out of date and would have cost ~30 GPU-hours (R-14).
- **Before killing any process** → confirm owner and command; none is running now.
