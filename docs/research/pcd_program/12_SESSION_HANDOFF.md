# 12 — Session Handoff

**Written:** 2026-07-16 · **Session:** Gate 0 truth audit
**Supersedes `docs/HANDOFF_CONTEXT.md`, which is stale and materially wrong — see below.**

---

## Read this first

`docs/HANDOFF_CONTEXT.md` (2026-07-15) instructs you to "resume training from the epoch-23
checkpoint — the model was still improving." **Do not do this.** Every premise is false:
the run reached epoch 113, plateaued at 16.36% val R@1, and overfits by 82 pp — identical to
v1. Acting on that handoff would have cost ~30 GPU-hours in support of claims that are
independently invalid. It is retained as a historical record only. Verify handoffs against
the pod before acting (R-14).

## Completed this session

- Full Gate 0 truth audit → `01_TRUTH_AUDIT.md` (12 findings, T1–T12).
- **Proved** per-level kappa heads receive zero gradient (executable, not inferred).
- **Proved** prediction flows low→high, not Rao–Ballard.
- **Proved** splits are clean: train ∩ val = 0, train/val ∩ SHARED1000 = 0.
- **Proved** pod code ≡ local HEAD by sha256 (4/4 files), despite manifest recording a false commit.
- Discovered v4's central claim is contradicted (82 pp gap, not 25 pp).
- Created the research OS under `docs/research/pcd_program/` (13 artifacts).
- Added `tests/test_pcd_gradient_flow.py` — 9/9 passing.

## Exact current state

| | |
|---|---|
| Branch / HEAD | `feature/predictive-cortical-decoder` @ `c65e834` (0 ahead/behind origin at session start) |
| Uncommitted | Gate 0 artifacts + test, committed this session; `PCD_v3/v4` configs **still untracked** (T1) |
| Active processes | **None.** H100 idle. Nothing was started, killed, or deleted this session. |
| Last valid checkpoint | `checkpoint_best.pt` (epoch 101, 16.36% val R@1) — **pod only**, unhashed, never dry-loaded |
| Pod | `orchestraiq-jupyter-54644cff87-gz6n2` — name may change on restart, re-resolve |

**Nothing destructive was done. No process killed, no file deleted, no training launched.**

## Next deterministic action

Execute **D-004**, in this order:

```bash
export KUBECONFIG="C:/Users/ComputaCenter/Downloads/antoniu_iepure.yaml"
POD=orchestraiq-jupyter-54644cff87-gz6n2
kubectl exec $POD -- bash -c 'cd /home/jovyan/work/FMRI2images && \
  sha256sum experimental_results/PCD_v4_8subject/subj01/checkpoint_{best,last}.pt'
```

Then dry-load `checkpoint_best.pt` with an **explicit state-dict key diff** (`strict=False`
currently hides mismatches — R-08), and copy `split.json` / `manifest.json` /
`training_log.csv` off-pod. Do **not** copy the 2.68 GB weights into git.

Then Gate 1: re-derive v1's 17.97% from `PCD_v1_8subject/`'s own CSV (currently UNVERIFIED
and the *only* evidence v4 was a regression), resolve the metric fork (T11), and run H1
(the level-3 bypass measurement — one forward pass, can falsify the program).

## Decisions that must not be reopened without new evidence

| ID | Decision |
|---|---|
| **D-001** | Do not resume PCD_v4. Plateaued, 82 pp gap, supports no surviving claim. |
| **D-002** | Predictive-coding framing is stripped. Direction becomes a tested factor, not an assumption. Call them "cross-level residuals" until identifiability is established. |
| **D-003** | Do not rewrite the architecture yet. Measure the level-3 bypass first. |
| **D-005** | Never report `val_r@1` without gallery size and the trial-level figure beside it. Never compare PCD to 77.2%. |

## Unresolved / open questions

1. **Why did the run die at epoch 113?** No traceback, no OOM, no early-stop line. UNKNOWN (F-004).
2. **Is the ~96M/167M per-subject projection figure real?** From the handoff, never verified.
   It is the leading explanation for the 82 pp gap (R-05/R-06) and the basis of Candidate B.
3. **Which R@1 is the honest headline** — 16.4% (image) or 3.3% (trial)? Requires reading the
   eval aggregation from source (T11).
4. **Are the 2 434 unaccounted images** explained by incomplete NSD sessions? TENTATIVE.
5. **Novelty: entirely unknown.** Literature matrix is empty. Hard Gate 2 blocker.

## Context-budget note

`01_TRUTH_AUDIT.md` and this file are sufficient to resume without replaying the session.
Do **not** re-read the full `predictive_cortical_decoder.py` to re-establish the direction or
gradient findings — they are pinned by `tests/test_pcd_gradient_flow.py`. Run the tests
instead; they are the durable evidence. Re-read source only when changing it.
