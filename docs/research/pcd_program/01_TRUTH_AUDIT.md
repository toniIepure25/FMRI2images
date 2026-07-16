# 01 — Gate 0 Truth Audit

**Audit date:** 2026-07-16
**Auditor:** automated Gate 0 pass (roles A/B/C/E executed sequentially — no true subagents spawned)
**Local HEAD:** `c65e834` on `feature/predictive-cortical-decoder`
**Evidence labels:** VERIFIED / SUPPORTED / TENTATIVE / CONTRADICTED / UNKNOWN / INVALID

Every statement below is tagged. Nothing in the seed context was accepted as fact.

---

## 1. Repository state

| Item | Status | Evidence |
|---|---|---|
| Branch `feature/predictive-cortical-decoder` exists, is checked out | VERIFIED | `git branch --show-current` |
| Local HEAD `c65e834`, **0 ahead / 0 behind** `origin/…` | VERIFIED | `git rev-list --left-right --count @{u}...HEAD` → `0  0` |
| Working tree has 4 untracked files, no modified tracked files | VERIFIED | `git status --porcelain` |
| `configs/experiments/PCD_v3_8subject.yaml`, `PCD_v4_8subject.yaml` are **untracked** | VERIFIED | `git status` |

> **Finding T1 (provenance gap).** The config that defines the *current* experiment
> (`PCD_v4_8subject.yaml`) is not in version control and therefore not on `origin`.
> The run that produced the only PCD checkpoint is not reconstructible from git alone.

---

## 2. Local vs pod vs checkpoint reconciliation

Pod: `orchestraiq-jupyter-54644cff87-gz6n2`, Running, 5d22h uptime, H100 80GB.

| Item | Status | Evidence |
|---|---|---|
| Pod git HEAD = `b96affa` — **4 commits behind local** | VERIFIED | `kubectl exec … git rev-parse HEAD` |
| Pod has 3 modified tracked files (`train_unified.py`, `predictive_cortical_decoder.py`, `unified_model.py`) | VERIFIED | pod `git status --porcelain` |
| **Pod file content is byte-identical to local HEAD** for all 4 critical files | VERIFIED | `sha256sum` on both sides — all 4 digests match |
| No training process running; GPU idle (4 MiB / 81559 MiB, 0% util) | VERIFIED | `ps aux`, `nvidia-smi` |

> **Finding T2 (manifest records the wrong commit).** `manifest.json` for the v4 run
> records `"git_commit": "b96affa…"`. The code that actually executed is byte-identical
> to `c65e834`, four commits later. The pod's working tree matches no commit; the
> manifest attributes the run to a commit whose code was *not* what ran.
> Consequence: the run is **reproducible in practice** (content hashes match local HEAD)
> but its recorded provenance is **false**. The manifest writer must hash source files,
> not trust `git rev-parse`, when the tree is dirty.

---

## 3. Artifact inventory

| Artifact | Local | Pod | Status |
|---|---|---|---|
| `experimental_results/PCD_v{1,2,3,4}_8subject/` | **absent** | present | VERIFIED |
| `PCD_v4_8subject/subj01/checkpoint_last.pt` (2.68 GB, Jul 16 09:38) | absent | present | VERIFIED |
| `PCD_v4_8subject/subj01/checkpoint_best.pt` (2.68 GB, Jul 16 06:59) | absent | present | VERIFIED |
| `split.json` (476 KB), `manifest.json`, `metrics/training_log.csv` (90 rows) | absent | present | VERIFIED |
| `docs/paper/pcd_paper.md` | present | — | VERIFIED |

> **Finding T3 (single point of failure).** Every PCD result and both 2.68 GB
> checkpoints exist **only** on pod-local NFS. There is no off-pod copy and no hash
> record. Checkpoint dry-load has **not** been performed (deferred: see Decision D-004).

---

## 4. The seed context is materially stale — CONTRADICTED

`docs/HANDOFF_CONTEXT.md` (dated 2026-07-15) states the v4 run stopped at **epoch 23**
and "needs restart". It did not.

| Handoff claim | Reality | Status |
|---|---|---|
| "completed 24 epochs", stopped at epoch 23 | reached **epoch 113/200**, last write 2026-07-16 09:38 | CONTRADICTED |
| Val R@1 = 5.99% | **16.36%** (epoch 112); best **16.36%** @ epoch 101 | CONTRADICTED |
| CSLS R@1 = 9.86% | **19.85%** | CONTRADICTED |
| "train/val gap ≈ 25%, MUCH better than v1's 82%" | train R@1 **98.83%** @ ep110 vs val 16.36% → **gap ≈ 82 pp** | **CONTRADICTED** |
| "still improving when killed" | plateaued: R@1 = .1636/.1632/.1636 over epochs 110–112 | CONTRADICTED |

> **Finding T4 (the central v4 claim is false).** v4's entire justification was that
> regularization fixed v1's overfitting (82 pp → 25 pp). That 25 pp figure was measured
> at **epoch 20, before convergence**. At epoch 110 the gap is **≈82 pp — identical to
> v1**. The regularization did not reduce overfitting; it **delayed** it.
> v4 best val R@1 = **16.36% @ epoch 101**; v1 reportedly reached **17.97% @ epoch 29**
> (v1 figure UNVERIFIED — from handoff, not yet re-read from v1's CSV).
> **The v1→v2→v3→v4 program produced no measurable improvement at ~4× the compute.**

> **Finding T5 (run terminated, cause unknown).** `pcd_v4_resumed2.log` ends mid-epoch 113
> at 09:38 with no traceback, no OOM message, no early-stop message. Process absent, GPU
> idle. Termination cause: **UNKNOWN** (external kill / node pressure / OOM-killer are all
> consistent with the evidence). Do not record this as "completed".

---

## 5. Architecture: the direction claim — CONTRADICTED

The module docstring (`predictive_cortical_decoder.py:1–31`) claims the model
"propagates only **prediction errors** from lower to higher levels" and that a positive
result "**validates predictive coding theory**", citing Rao & Ballard (1999).

Traced forward pass (`predictive_cortical_decoder.py:673–690`) and confirmed by
executable probe (`tests/test_pcd_gradient_flow.py::test_prediction_flows_low_to_high_not_rao_ballard`):

```
h0        = encode(level0 tokens)                  # V1/V2, raw
predicted = prediction_heads[k](h_k)               # source = level k   (LOWER)
error     = tokens[k+1] - predicted                # target = level k+1 (HIGHER)
h_{k+1}   = encode(error)
```

`prediction_heads[0]: level 0 → 5 tokens (level 1)`; `prediction_heads[1]: level 1 → 7 tokens (level 2)`.

> **Finding T6 (mislabelled architecture).** Prediction flows **lower → higher**.
> Canonical predictive coding requires predictions to **descend** (higher → lower), with
> residuals ascending. The implemented operation is **feed-forward residual extraction**:
> "encode the part of level k+1 not linearly predictable from level k's CLS token."
> That is a legitimate — and arguably interesting — ML inductive bias (hierarchical
> decorrelation / novelty gating). **It is not Rao–Ballard predictive coding.**
> Status: the *code* is defensible; the *label and the cited theory* are not.

> **Finding T7 (the residual level bypasses the hierarchy).** Level 3
> (`nsdgeneral_other`) is encoded **directly** and fed straight to the aggregator with no
> prediction applied (`:686–690`). Per `PCD_v4_8subject.yaml:63–80`, `nsdgeneral_other`
> is allocated **10 000 of ~15 500 voxels ≈ 64%** of all input. So ~2/3 of the signal
> takes an **unmediated, non-predictive path** into the fused representation and competes
> directly against the three "predictive" levels. Any claim that the decoder is
> "driven by prediction errors" must quantify this bypass first. Not yet quantified —
> requires attribution/ablation (Gate 4).

---

## 6. Gradient flow into uncertainty heads — VERIFIED BROKEN

`PCDModel.forward` (`unified_model.py:890–902`) returns **only** `(pcd_out.mu, pcd_out.kappa)`.
`level_kappas` is written to `self._last_pcd_extras`, a plain dict.
Repo-wide grep: `_last_pcd_extras` is read by **exactly one** consumer —
`scripts/utils/smoke_test_pcd.py:37`. **No loss reads it.**

Executable proof (`tests/test_pcd_gradient_flow.py::test_per_level_kappa_heads_are_unsupervised`,
9/9 passing):

```
loss = f(mu, kappa) only   ->  backward()

level_kappa_heads   params=  8   grad_is_None=  8    <-- ZERO GRADIENT
prediction_heads    params=  8   grad_is_None=  0
level_encoders      params= 64   grad_is_None=  0
aggregator          params=  7   grad_is_None=  0
vmf_decoder         params=  6   grad_is_None=  0
roi_projections     params= 68   grad_is_None=  0
```

> **Finding T8 (per-level uncertainty is untrained).** All four
> `PerLevelKappaHeads` linear heads receive `grad is None` under every loss the training
> loop can construct, while every other module trains. They remain at **random
> initialisation for the entire run**. `enable_per_level_kappa: true` in the config is
> therefore misleading — it enables *computation*, not *learning*.
>
> Consequence: **`scripts/analysis/pcd_neuroscience_analysis.py` is INVALID** where it
> touches `level_kappas` (`:263–274`, `:303–304`, `level_kappas_by_category`). It reports
> a **random linear projection of trained features**. Such a projection is not noise —
> it will produce plausible-looking, reproducible, category-varying figures — which makes
> it *more* dangerous, not less. Any "per-level uncertainty decomposition" claim is
> **FORBIDDEN** pending a real objective.
>
> Note the reason this survived review: `level_kappas.grad_fn is not None` — the tensor
> *is* in the autograd graph and looks connected on inspection. Nothing backpropagates
> through it because no loss consumes it. Pinned by
> `test_level_kappas_are_in_graph_but_orphaned`.

> **Finding T9 (global kappa is near-degenerate).** From `pcd_v4_resumed2.log` epoch 112:
> `kappa_mean=9.27, kappa_std=0.47, kappa_min=8.04, kappa_max=10.46`
> (coefficient of variation ≈ 5%). The global concentration is **almost constant across
> samples** and therefore carries almost no per-stimulus information. With
> `kappa_reg.lambda_kappa: 0.05` (5× v1) applying steady downward pressure, this is the
> expected outcome and echoes the documented kappa-collapse failure mode in
> `docs/EXPERIMENT_CONTEXT.md` §13. **Any uncertainty-aware / conformal thesis built on
> this checkpoint is currently unsupported.** No proper scoring or calibration evaluation
> has been run — status UNKNOWN, but the prior is poor.

---

## 7. Data split & leakage audit — PASSES

Tested against the v4 run's own `split.json` and NSD's authoritative
`nsd_stim_info_merged.csv` (73 000 rows, 1 000 flagged `shared1000`).

| Test | Result | Status |
|---|---|---|
| `split_by_image` | `true` | VERIFIED |
| `exclude_shared1000` | `true` | VERIFIED |
| train ids unique | 62 610 / 62 610 | VERIFIED |
| val ids unique | 6 956 / 6 956 | VERIFIED |
| **train ∩ val** | **0** | **PASS** |
| **train ∩ SHARED1000** | **0** | **PASS** |
| **val ∩ SHARED1000** | **0** | **PASS** |
| total unique images | 69 566 (of 72 000 non-shared; 2 434 unaccounted) | VERIFIED |

> **Finding T10 (splits are clean — the good news).** The sealed test set is intact and
> the split is genuinely image-level, so repeated presentations of one image cannot cross
> the train/val boundary. This is the strongest positive finding of Gate 0 and it means
> the project's foundation is sound.
> The 2 434 unaccounted images are consistent with NSD subjects who did not complete all
> sessions (subj03/04/06/08) — plausible, but **not yet verified** (TENTATIVE).
>
> **Scope limit — this does NOT yet establish leakage-freedom.** Still unverified:
> scaler/PCA/reliability fit on train only; the 16 384-entry contrastive queue excluding
> val; hyperparameters not selected on val. `preprocessing.enabled: false` in the v4
> config reduces but does not eliminate this surface. SHARED1000 is clean **because PCD
> never evaluated on it at all** — which is correct conduct, not an accident.

---

## 8. Metric-definition hazard — must be resolved before any reporting

`training_log.csv` carries **two** different R@1 columns:

| Column | Epoch 112 value | Gallery |
|---|---|---|
| `val_r@1_trial` | **0.0330** (3.3%) | 19 236 trials |
| `val_r@1` | **0.1636** (16.4%) | 6 956 images |

> **Finding T11 (a 5× reporting fork).** All prior PCD reporting used `val_r@1` = 16.4%.
> The trial-level number is **3.3%**. The config sets `average_repetitions: false` for
> *training*, yet evaluation aggregates 19 236 trials → 6 956 images. Which number is the
> honest headline depends entirely on the aggregation, which must be read from the eval
> code before publication. Reporting 16.4% without disclosing the aggregation would
> violate rule 14/16.
>
> **Finding T12 (the 77.2% comparison is INVALID).** The frozen retrieval system's
> 77.2% R@1 is on **SHARED1000 (gallery 1 000)**. PCD's 16.4% is on a **6 956-image val
> gallery**, different protocol, different split, and PCD has **never been run on
> SHARED1000**. These numbers are not comparable in either direction. Any table placing
> them side by side is forbidden.

---

## 9. Capability audit

| Capability | Available | Verified by | Constraint / use |
|---|---|---|---|
| Repo browsing, edit, git | Yes | direct | Gate 0 commits |
| Shell (Bash + PowerShell) | Yes | direct | Bash used throughout |
| Test execution (pytest) | Yes | 9/9 passed | regression guards |
| Local torch | Yes — **2.7.1+cpu, CUDA False** | `torch.cuda.is_available()` → False | CPU only; gradient/shape probes only, **no local training** |
| `kubectl` + kubeconfig | Yes | `kubectl get pods` | pod at `…-gz6n2` |
| Pod H100 80GB, CUDA 12.8, torch 2.11.0 | Yes, **idle** | `nvidia-smi` | all training must run here |
| Parallel subagents | Available but **not used** | — | Gate 0 was bounded; roles run sequentially. Revisit for Gate 2 literature sweep. |
| Web / scholarly search | **Not yet exercised** | — | **Literature matrix is EMPTY — novelty claims are unsupported.** Gate 2 blocker. |
| Experiment tracking | None found | — | `manifest.json` + CSV are the only provenance |

---

## 10. Gate 0 verdict

**Exit criterion — "a reproducible authoritative state exists": PARTIALLY MET.**

Met: code content reconciled by hash across local/pod; splits verified clean; the two
central architecture risks resolved with executable evidence and pinned by tests.

Not met: checkpoint dry-load not performed; v1 baseline numbers not independently
re-derived; run manifest provenance is false (T2); all artifacts single-homed on the pod (T3).

**Gate 1 (baseline reproducibility) is NOT cleared. Do not resume training.**
Rationale in `04_DECISION_LOG.md` D-001.
