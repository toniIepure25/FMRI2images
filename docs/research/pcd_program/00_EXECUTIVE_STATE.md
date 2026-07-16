# 00 — Executive State

**Last updated:** 2026-07-16 · **Gate:** 0 complete (partial exit) → Gate 1 next
**Commit:** `c65e834` · `feature/predictive-cortical-decoder` · 0 ahead / 0 behind origin
**Dirty:** Gate 0 artifacts staged this session; `PCD_v3/v4` configs still untracked (T1)

---

## Current objective

Determine whether the Predictive Cortical Decoder supports **any** defensible scientific
claim. Gate 0 found that its three headline contributions are each independently invalid.
The project is in **claim triage**, not model development.

## Active runs

**None.** Pod `orchestraiq-jupyter-54644cff87-gz6n2` is up; H100 **idle** (0% util).
The PCD_v4 run terminated at epoch 113 on 2026-07-16 09:38, cause **UNKNOWN** (no traceback).
**Do not resume it** — see Decision D-001.

## Verified results (the only numbers that exist)

| Metric | Value | Caveat |
|---|---|---|
| PCD_v4 best val R@1 | **16.36%** @ epoch 101 | 6 956-image gallery |
| PCD_v4 val R@1 (trial-level) | **3.30%** | 19 236 trials — **5× fork, see T11** |
| PCD_v4 CSLS R@1 | 19.85% | same gallery |
| PCD_v4 train R@1 | **98.83%** @ ep110 | → **82 pp overfit gap** |
| PCD_v4 params | 167 291 141 | from `manifest.json` |
| Frozen retrieval system | 77.2% R@1 SHARED1000 | **NOT comparable to PCD** (T12) |

PCD has **never been evaluated on SHARED1000**. The sealed set is intact.

## Top blockers

1. **Per-level kappa heads receive zero gradient** (T8) → "per-level uncertainty" is
   FORBIDDEN; `pcd_neuroscience_analysis.py` is INVALID where it reads `level_kappas`.
2. **Architecture is not predictive coding** (T6) — prediction flows low→high. All
   Rao–Ballard framing must be stripped (D-002).
3. **v4's central claim is false** (T4) — regularization delayed, did not reduce, overfitting.
4. **Literature matrix is empty** — every novelty claim is currently unsupported. Gate 2 blocker.
5. **All artifacts single-homed on pod NFS**, no hashes, checkpoint never dry-loaded (T3, D-004).

## What is solid

- **Splits are clean.** train ∩ val = 0; train/val ∩ SHARED1000 = 0. Image-level splitting
  genuinely works (T10). The project's foundation is sound.
- **Pod code ≡ local HEAD** by sha256 on all 4 critical files — the run *is* reproducible
  in practice, despite its manifest recording the wrong commit (T2).
- Nine regression tests now pin the two architecture findings (`tests/test_pcd_gradient_flow.py`).

## Next five actions

1. **D-004:** hash both pod checkpoints; dry-load `checkpoint_best.pt` with an explicit
   state-dict key diff (`strict=False` currently hides mismatches); copy `split.json`,
   `manifest.json`, `training_log.csv` off-pod.
2. **Gate 1:** re-derive v1's reported 17.97% @ epoch 29 from `PCD_v1_8subject/`'s own CSV.
   Currently UNVERIFIED — it is the *only* evidence that v4 was a regression.
3. **Gate 1:** read the eval aggregation from source; resolve the 16.4% / 3.3% metric fork (T11).
4. **Gate 1 (H1):** quantify the level-3 bypass — `nsdgeneral_other` holds ~64% of voxels
   and skips the hierarchy entirely (T7). One forward pass over val, epoch-101 checkpoint.
5. **Gate 2:** run the literature sweep; populate `10_LITERATURE_MATRIX.csv`.

## Standing prohibitions

- Do not resume PCD_v4 (D-001). Do not rewrite the architecture yet (D-003).
- Do not report `val_r@1` without its gallery size and the trial-level figure beside it (D-005).
- Do not compare PCD to 77.2% (T12).
