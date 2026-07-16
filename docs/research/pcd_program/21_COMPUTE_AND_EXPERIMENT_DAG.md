# 21 — Compute Budget and Experiment DAG

**Date:** 2026-07-16 · **Revised Phase 2.5.** Resource: **1× H100 80GB, idle.**

---

## 0. Phase 2.5 revision — the DAG changed shape

**The critical path no longer runs through external data.** Four of six shifts (1, 4, 5, 6)
run on NSD, already on the pod (`27` §1). The matched-control pilot — which decides whether a
neural-target effect exists at all — needs **only NSD**.

```
 ┌──────────────────────────┐        ┌───────────────────────────┐
 │ WIRE NCD into            │        │ B-DATA (parallel, 0 GPU)  │
 │ create_model + train loop│        │ NSD-Synthetic (CC-BY 4.0) │
 │  ── blocks the pilot ──  │        │ NSD-Imagery (CC-BY-NC-ND) │
 └───────────┬──────────────┘        └─────────────┬─────────────┘
             ▼                                     │
 ┌──────────────────────────┐                      │
 │ E-P1 MATCHED PILOT       │                      │
 │ A/B/C/F × 2 seeds        │                      │
 │ 1 subject, NSD only      │                      │
 │ ~8 GPU-h  ★ THE GATE     │                      │
 └───────────┬──────────────┘                      │
             │                                     │
      B ≈ F ─┴─► REPORT NEGATIVE (24 §6).          │
             │   "Auxiliary regularization helps;  │
             │    neural target not supported."    │
             │   Do NOT escalate.                  │
             ▼                                     │
 ┌──────────────────────────┐                      │
 │ measured variance ──────────► rewrite 20 §6 power│
 │ measured throughput ────────► rewrite §2 below   │
 └───────────┬──────────────┘                      │
             ▼                                     ▼
 ┌──────────────────────────┐        ┌───────────────────────────┐
 │ 4-subject pilot          │        │ perception OOD kill tests │
 │ + ARM-D/E/G/H            │        │ (imagery 12 vision stim;  │
 │ ★ PROMOTION GATE         │        │  synthetic from ep98 ckpt)│
 └───────────┬──────────────┘        └─────────────┬─────────────┘
             └──────────────┬─────────────────────-┘
                            ▼
                 ┌──────────────────────┐
                 │ 8-subj confirmatory  │  ── NOT CLEARED ──
                 │ shifts 1,2,4,5,6     │
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │ shift 3 imagery      │  sealed, once, last
                 │ E-08 SHARED1000      │  once, last
                 └──────────────────────┘
```

## 1. Legacy DAG (Phase 2 — superseded above, retained for provenance)

```
                    ┌─────────────────────────────────────┐
                    │ B-DATA: obtain + audit NSD-Imagery  │  ◄── HARD BLOCKER
                    │ (not on pod; 181TB free)            │      external dependency
                    └──────────────┬──────────────────────┘
                                   │
 ┌──────────────────┐    ┌─────────▼──────────┐    ┌────────────────────┐
 │ E-00 protocol    │    │ B-LIT: full-text   │    │ E-01 level-3       │
 │ standardisation  │    │ NSD-Imagery, LEA,  │    │ bypass diagnostic  │
 │ + leakage tests  │    │ Hi-DREAM  (novelty │    │ (PCD ep98 ckpt)    │
 │ + FLOP counter   │    │ + SESOI + power)   │    │ 1 fwd pass         │
 └────────┬─────────┘    └─────────┬──────────┘    └────────────────────┘
          │                        │                    (parallel, cheap,
          │                        │                     NOT a contribution)
          ├────────────────────────┤
          ▼                        ▼
 ┌────────────────────┐   ┌──────────────────────┐
 │ E-02 NCD impl      │   │ 20_SAP frozen        │
 │ + gradient tests   │   │ (SESOI, power, TOST) │
 │ + tiny-overfit     │   └──────────┬───────────┘
 └────────┬───────────┘              │
          ▼                          │
 ┌────────────────────┐              │
 │ E-03 1-subj pilot  │              │
 │ λ ∈ {0, 0.5}       │              │
 └────────┬───────────┘              │
          ▼                          │
 ┌────────────────────┐              │
 │ E-04 4-subj pilot  │◄─────────────┘
 │ + matched-reg      │
 │ ★ PROMOTION GATE   │  ── fail ──► report NEGATIVE (Outcome B). Do not escalate.
 └────────┬───────────┘
          ▼
 ┌────────────────────┐   ┌──────────────────┐   ┌──────────────────┐
 │ E-05 confirmatory  │──►│ E-06 imagery     │──►│ E-07 ablations   │
 │ 8 subj × 3 seeds   │   │ transfer ★ENDPOINT│   │ (FDR corrected)  │
 └────────────────────┘   └──────────────────┘   └────────┬─────────┘
                                                          ▼
                                                 ┌──────────────────┐
                                                 │ E-08 SHARED1000  │
                                                 │ ONCE. LAST.      │
                                                 └──────────────────┘
```

**Critical path runs through B-DATA and B-LIT, not through compute.** The GPU is idle and
will stay idle for the first stretch. That is correct: `01_TRUTH_AUDIT` §4 and F-001 are what
happen when compute runs ahead of understanding.

## 2. Budget

NCD is ~20.5M params (`18` §8) vs PCD's 167M — **8× smaller**. PCD ran ~12 min/epoch on 8
subjects; NCD should be faster, but this is an **estimate, not a measurement**.

| ID | Arms × seeds × subj | Est. GPU-h | Basis |
|---|---|---|---|
| E-01 | 1 fwd pass | **<0.1** | measured-ish (eval was ~3 min in PCD logs) |
| E-02 | tiny overfit | **<0.5** | 1 subject, few hundred steps |
| E-03 | 2 arms × 2 seeds × 1 subj | **~8** | extrapolated ↓ from PCD |
| E-04 | 3 arms × 2 seeds × 4 subj | **~40** | extrapolated |
| E-05 | 4 arms × 3 seeds × 8 subj | **~150** | extrapolated |
| E-06 | eval only | **~2** | |
| E-07 | ~8 arms × 3 seeds × 8 subj | **~250** | the largest line — prune after E-05 |
| E-08 | eval only | **~2** | |
| | **Total** | **~450 GPU-h** | ≈ 19 days wall-clock on one H100 |

**All estimates are TENTATIVE.** PCD's registry says `FLOPs: NOT_MEASURED` — that entry must
not appear twice. `scripts/utils/count_flops.py` is a Gate 3 deliverable and blocks any
parameter-matched comparison.

## 3. Value-of-information ordering

Cheapest-and-most-decisive first:

1. **B-LIT full-text reads (~0 GPU).** Can kill the whole thesis (if NSD-Imagery/LEA already
   tested an auxiliary neural-prediction objective, novelty is dead — `17` §6). **Highest VOI
   in the program and it costs nothing.**
2. **B-DATA audit (~0 GPU).** Can force the `17` §5 fallback.
3. **E-00 + E-01 (<0.1 GPU-h).** E-01 tells us whether our own model is structurally broken.
4. **E-02 (<0.5).** Gradient tests can kill a bad implementation before any training.
5. **E-04 (~40).** The promotion gate — where the thesis actually lives or dies.

**Do not spend E-05's ~150 GPU-h to answer what E-04's ~40 can.** D-001 already prevented one
30-GPU-h run supporting no surviving claim; the same discipline applies here.

## 4. Futility / stop rules

| Trigger | Action |
|---|---|
| E-04 fails its gate | **Stop. Report negative** with equivalence bound. Do not escalate to 8 subjects. |
| E-07 exceeds 250 GPU-h | Prune arms by VOI; publish the subset; disclose |
| Any run plateaus >20 epochs below its arm's gate | Kill — F-001's lesson: a plateau is not a "still improving" |
| B-LIT kills novelty | Invoke Outcome B/C; do not spend GPU |

## 5. Infrastructure obligations (from F-004/F-005)

- Every run writes a manifest that **hashes source files** — never trusts `git rev-parse` on
  a dirty tree (F-005 recorded a commit whose code did not run).
- Heartbeat + resumable checkpoints: PCD died at epoch 113 with **no traceback and no
  diagnosis** (F-004). Unexplained death must not recur silently.
- Checkpoints hashed at write time and copied off-pod (D-004; everything is single-homed on
  NFS today).
- Checkpoint selection criterion logged explicitly (T14 — `val_loss` vs reported R@1).
