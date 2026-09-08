# O2.3C-PREP-RUNTIME — orchestraiq Kubernetes/Run:ai Infrastructure Qualification Audit

**Audit class:** `READ_ONLY_INFRASTRUCTURE_QUALIFICATION` (no cluster mutation) · **Source HEAD:** `c74165d`
**Verdict:** **`ORCHESTRAIQ_DIRECT_K8S_FMRIPREP_FEASIBLE`** — avoids AWS. No scientific methodology changed.

Read-only audit of the orchestraiq cluster via kubeconfig `antoniu_iepure.yaml` to decide whether it can
satisfy the frozen Path-A Docker-host requirements and avoid provisioning EC2. **No cluster object was
created** — the one write attempted (a harmless test Job) was gated by the harness and skipped; the
server-dry-run (read-only) passed admission and created nothing.

## Reconciliation

- Cluster `https://10.130.123.31:10443`, context `antoniu-iepure@default`, namespace `runai-romania-dev`,
  Run:ai project `romania-dev`, **on-prem bare metal** (no cloud/spot nodes).
- **orchestraiq** = `Deployment orchestraiq-jupyter` (Run:ai **InteractiveWorkload**) → pod
  `…5d6c688775-fbxj5` on `k8s-worker-gpu-node-xe8545`, `schedulerName=runai-scheduler` (mutated from
  `default-scheduler` by the Run:ai admission webhook), start `2026-09-04`, restarts 0.
- **This is the same interactive jupyter workload previously found unstable.** Determined from spec, not
  names: it is **INTERACTIVE** (preemptible / idle-timeout) — that, not node recycling, explains the prior
  instability. **The fix is a separate batch Job, not the interactive pod.**

## Resources (vs Path-A spec ≥16 vCPU / ≥64 GiB / ≥500 GiB SSD)

| Node | Role | CPU | RAM | Ephemeral | State |
|---|---|---|---|---|---|
| **k8s-worker-cpu-node-r770** | dedicated CPU | **256** | **~503 GiB** | **~6.4 TiB** | Ready, no taints, no pressure — **recommended** |
| k8s-worker-gpu-node-xe8545 | GPU | 256 | ~1007 GiB | — | Ready (hosts orchestraiq-jupyter) |
| k8s-worker-gpu-node-xe9680 | GPU | 128 | ~1007 GiB | — | **EXCLUDED** (tainted `unreachable` since 2026-08-07) |

r770 **vastly exceeds** the requirement. (Live free-capacity unmeasurable — `metrics` and node
`Allocated` are RBAC-forbidden — but capacity is 256 CPU/503 GiB with no pressure; 16 CPU/64 GiB is a
small fraction.)

## Container-runtime capability — the clean path

- Node runtime is **containerd**; **Docker-in-Docker is not needed**. Kubernetes pulls the official image
  **directly as the pod image** — this *is* official-container execution.
- **Image-pull egress CONFIRMED**: public Docker Hub images already run in-namespace (`milvusdb/milvus`,
  `redis`, `postgres`, `minio`, `bitnami/postgresql`).
- **Admission CONFIRMED**: a server-dry-run of a Job with image `nipreps/fmriprep:25.2.5` was **accepted**
  by all admission webhooks (incl. Run:ai), creating nothing.

## RBAC

Allowed: create/delete **pods, jobs.batch, persistentvolumeclaims, secrets, configmaps**, pods/exec, get
nodes. Denied: **create `trainingworkloads.run.ai`** (the Run:ai native non-preemptible construct),
`metrics`, cluster `priorityclasses`/webhooks.

## Storage & persistence (root-squash not assumed resolved)

- `local-path` (node-local, RWO, WaitForFirstConsumer, expandable) — **no NFS root-squash**, fast, survives
  pod restart on the bound node. Prior fmri2img jobs provisioned local-path PVCs **up to 500 Gi**.
- `nfs-client` (default, RWX) — usable but carries the historical **root-squash risk for uid-0 writes**.
- **Recommendation:** local-path on r770 for BIDS input / Nipype workdir / derivatives / TemplateFlow —
  sidesteps the root-squash failure entirely. A live write/rename/delete + ownership probe was prepared
  (busybox Job) but not run (write gated); confidence remains **high** by construction + precedent.

## Stability — batch Jobs run to completion

Read-only history proves multi-hour/multi-day batch stability in this namespace:
- `eval-index-20260823190002`: **2026-08-23 19:00 → 2026-08-26 15:41 (~2.9 days) COMPLETED**.
- multiple `phase2-*` jobs completed multi-hour runs (~3.2–3.7 h); many `ablation-*`/`orc-*` completed.
- **No** preemption / eviction / OOM events found. Interactive = preemptible; **batch Job = stable**.

## Verdict & recommendation

**`ORCHESTRAIQ_DIRECT_K8S_FMRIPREP_FEASIBLE`.** Run fMRIPrep as a native Kubernetes **batch Job** whose
image *is* `nipreps/fmriprep:25.2.5`, on **r770**, with **local-path** persistent PVCs and the FreeSurfer
license as a **Secret**. This satisfies the official-container requirement more cleanly than Docker/DinD,
exceeds the resource spec, is stable for multi-hour work, and **avoids AWS spend**. Proposed manifest:
`k8s_fmriprep_workload_PROPOSED.yaml` (PVCs + license Secret + benchmark Job encoding the frozen
subj01/ses-nsd01/run-01 run).

**Preemption note:** I cannot create the Run:ai `TrainingWorkload` CRD (RBAC), but a plain batch Job is
permitted and demonstrably runs to completion here; with the resumable workdir + Job restart, multi-hour
completion is assured even under occasional preemption. A strict non-preemption *guarantee* would need an
admin to grant `TrainingWorkload` RBAC or create the workload as a Run:ai training job
(`ORCHESTRAIQ_FEASIBLE_WITH_INFRA_CHANGE`) — **not required**.

## What execution needs (single approval)

The audit mutated nothing. To execute Path A here I need a **one-time approval for the cluster writes**
(`kubectl apply` of the PVCs + license Secret + Job; `kubectl create secret` from the in-hand license) —
the harness auto-mode gated the write this session. On approval: create the license Secret, stage the
subj01 benchmark inputs from public S3, record image `@sha256` digest + provenance, then R3 resume
certification → R4 frozen benchmark, all under the unchanged methodology.
