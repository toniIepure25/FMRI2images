# 00 — MINDCOMPILER Executive State

**Last updated:** 2026-07-17 · **Gate: M0 (Truth, novelty, feasibility) — IN PROGRESS**
**Branch:** `research/mindcompiler-neural-state-operators` @ `c1095de`
**Parent:** `feature/predictive-cortical-decoder` @ `c1095de` (PCD/NCD archived, not deleted)
**Active runs: NONE.** Pod idle (GPU 4 MiB, 0%). **No GPU has been spent on MINDCOMPILER.**

---

## Mission

Investigate whether mental states are related by structured, composable transformations over
a shared neural content manifold:
`y_{p,s,m} = O_{p,m}( T_s(c) ) + ε`

**The object of study is the transformation between mental states**, not the decoder. The
existing decoding stack is substrate.

**The decisive prospective demonstration:** given neural activity from one state, predict the
neural activity the *same* participant would produce for the *same* content in *another*
state — validated on **held-out measured neural data**. A reconstructed image or a semantic
match is **not** a neural counterfactual.

## Terminology status — WORKING HYPOTHESES ONLY

`universal`, `causal`, `algebra`, `operator`, `counterfactual`, `brain-to-brain`,
`mental-state compiler` are **project vocabulary, not claims**. None has passed its test.
None may appear as a scientific claim until it does.

## Gate M0 progress

| § | Task | State |
|---|---|---|
| 3.1 | Reconcile local / remote / pod | ✅ **DONE** |
| 3.2 | Archive PCD/NCD | ✅ **DONE** — `docs/research/archive/PCD_NCD_TERMINATION_MEMO.md` (`c1095de`) |
| 3.3 | Create + publish branch | ✅ **DONE** — verified on server via `git ls-remote` |
| 4 | Research OS scaffold | 🔶 **PARTIAL** — this file + `19_SESSION_HANDOFF.md` only |
| 5 | Frontier literature review + novelty verdict | ❌ **NOT STARTED — blocks everything** |
| 6 | Formal operator algebra (H0–H7) | ❌ NOT STARTED |
| 7 | Dataset/modality matrix + adapters | ❌ NOT STARTED |
| 8 | Falsification ladder B0–B5 | ❌ NOT STARTED |
| 18 | Synthetic operator-recovery benchmarks | ❌ NOT STARTED |

**No novelty verdict has been issued.** The permitted set is
`MOONSHOT_NOVELTY_SUPPORTED` / `PARTIALLY_OVERLAPPING_REQUIRES_REDESIGN` /
`NOVELTY_INSUFFICIENT` / `DATA_INSUFFICIENT_FOR_FLAGSHIP` / `PROSPECTIVE_PROGRAM_REQUIRED`.
Issuing one before the review would repeat exactly the error this program is built to avoid.

## Verified state

| | |
|---|---|
| Remote | `origin/research/mindcompiler-neural-state-operators` = `c1095de` — **server-verified** |
| Parent published | `origin/feature/predictive-cortical-decoder` = `c1095de`; **nothing unpushed** |
| Tests | 115 program tests passing (inherited); 10 pre-existing env failures, verified unrelated |
| Pod | `orchestraiq-jupyter-54644cff87-gz6n2` — no processes, GPU idle. Pod git HEAD `b96affa` **matches no commit** (hand-copied files, F-005). **Pod is NOT synced to this branch.** |

## Inherited assets (from the archive memo §4)

Verified clean NSD splits (train ∩ val = 0; ∩ SHARED1000 = 0) · `partial_conjunction.py`
(r-of-n compound test + 21 simulations — directly applicable to multi-property algebra
claims) · `arm_c_partner.py` (content-shuffled controls + 17 invariant tests — mandatory for
any transport claim) · matched-control family + `assert_param_parity` · per-ROI tokenisation
with low-rank subject adapters (reusable as an **observation model** `O_{p,m}`).

**Inherited rule, program-wide:** *every interpreted quantity must have an identifying
objective, enforced by a gradient test.* PCD's kappa heads violated it and produced stable,
plausible figures from random weights. `T_s`, `S_p` and all uncertainties are interpreted
quantities and inherit this rule.

## The two risks that most likely kill this program

1. **Semantic shortcut (§H6).** Apparent neural transport explained entirely by shared
   semantic labels or pretrained embeddings. **B5 must be built before any operator claim.**
2. **Data insufficiency.** Neural counterfactual prediction needs *the same content measured
   in multiple states, paired, at trial level*. NSD-Imagery has **4 subjects, 18 stimuli**,
   and Spera et al. showed a stronger decoder is **at chance zero-shot** on it — so the
   obvious substrate may be too thin for the flagship. **This is the likeliest route to
   `PROSPECTIVE_PROGRAM_REQUIRED`, and finding that out is a legitimate M0 outcome.**

## Next five actions

1. **§5 frontier literature review** → `02_FRONTIER_LITERATURE_REVIEW.md`,
   `03_NOVELTY_AND_OVERLAP_MATRIX.csv`, adversarial novelty verdict. **Zero GPU. Blocks all.**
   Priority families: hyperalignment / shared response models; neural/Koopman operators;
   causal representation learning; perception-vs-imagery transformation work; cross-subject
   neural translation; optimal transport for neural data.
2. **§7 dataset matrix** → `07_DATASET_AND_MODALITY_MATRIX.csv`. Decide honestly whether
   *any* public data supports trial-level paired multi-state content.
3. **§6 formal theory** → `06_FORMAL_OPERATOR_ALGEBRA.md` with H0–H7 and per-property
   estimand / null / baseline / threshold / kill criterion.
4. **§18 synthetic recovery** — must recover *absence* of composition when absent. **A model
   that always finds an algebra fails this gate.** Build before touching real data.
5. **§8 B0–B5**, with **B5 (semantic shortcut) first** among the transport baselines.

## Standing prohibitions (inherited + new)

- All PCD/NCD FORBIDDEN claims carry forward (archive memo §5) and do not expire.
- **No "first" claims.** Novelty must be conceptual and experimental, never nominal.
- **Cycle consistency alone is not evidence** — degenerate solutions satisfy it.
- **Never call a reconstruction or semantic match a neural counterfactual.**
- **Never call simulation prospective validation.**
- Do not treat trials, voxels, seeds, or reconstruction samples as independent participants.
- Do not begin full-scale GPU training before the relevant promotion gate.
