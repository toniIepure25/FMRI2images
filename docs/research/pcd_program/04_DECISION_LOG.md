# 04 — Decision Log (append-only)

Each entry: context → alternatives → evidence → selection → reviewer objection → revisit trigger.
Disagreement is preserved, not resolved by fiat.

---

## D-001 — Do NOT resume the PCD_v4 training run

**Date:** 2026-07-16 · **Gate:** 0 · **Status:** ACTIVE

**Original proposal (seed context + `HANDOFF_CONTEXT.md` §2, §5):**
"Resume training from the epoch-23 checkpoint — the model was still improving. Let it
train for the full 200 epochs (ETA ~30 hours)."

**Identified problem:** every premise of that instruction is false as of 2026-07-16.

**Evidence (all VERIFIED, see `01_TRUTH_AUDIT.md` §4):**
- The run is not at epoch 23. It reached **epoch 113/200** (log ts 09:38 today).
- It is not "still improving". Val R@1 = .1636/.1632/.1636 across epochs 110–112 — **flat**.
  Best = 16.36% @ epoch 101; +0.006 gained over the 18 epochs since epoch 83.
- The regularization did not work. Train R@1 **98.83%** vs val **16.36%** → **≈82 pp gap**,
  indistinguishable from v1's 82 pp. The advertised "25 pp gap" was an artefact of reading
  the gap at epoch 20, pre-convergence.
- Every scientific claim the run was meant to support is independently invalid:
  per-level kappa heads are untrained (T8), the architecture is not predictive coding (T6),
  and global kappa is near-degenerate (T9).

**Alternatives considered:**
1. *Resume to 200 epochs as instructed.* → ~30 H100-hours to move a plateaued number that
   supports no surviving claim. Value of information ≈ 0. Rejected.
2. *Resume with fixed regularization (v5).* → Premature: we do not yet know whether the
   82 pp gap is a capacity problem at all. Per-subject ROI projections are ~96M of 167M
   params (57%, per handoff — UNVERIFIED). Rejected pending measurement.
3. *Stop; re-derive baselines and fix identifiability first.* → **SELECTED.**

**Selected action:** halt the v4 line. Preserve checkpoints (D-004). Proceed to Gate 1
baseline reproduction and Gate 2 scientific audit before any further GPU spend.

**Reviewer objection (Agent F, preserved):** "Sunk-cost reasoning cuts both ways — the
checkpoint at epoch 101 is a *usable artefact* even if its claims are dead. Do not
mistake 'the claims are invalid' for 'the model is worthless'; the checkpoint is the only
thing that can answer whether hierarchical residuals carry *any* conditional information.
Halting the run is correct; discarding the checkpoint would not be." — Accepted; folded
into D-004.

**Revisit trigger:** if Gate 2 selects a thesis for which a converged v4 checkpoint is a
required baseline arm, resume from `checkpoint_best.pt` (epoch 101) — **not** from scratch,
and **not** past the plateau.

---

## D-002 — Strip predictive-coding framing from the architecture; keep the mechanism

**Date:** 2026-07-16 · **Gate:** 0 · **Status:** ACTIVE

**Original proposal:** PCD "models the visual hierarchy as a predictive coding system
(Rao & Ballard, 1999)"; a positive result "validates predictive coding theory in the
decoding setting" (`predictive_cortical_decoder.py:1–31`).

**Identified problem:** two independent defects.
1. **Direction.** Prediction flows lower→higher (VERIFIED, T6). Rao–Ballard requires
   predictions to descend. The cited theory does not describe this code.
2. **Category error.** Even with the direction corrected, decoding accuracy cannot
   validate a biological theory. A decoder's success is evidence about *information
   present in the measurement*, not about *the brain's implementation*. This is
   non-negotiable rule 10/11.

**Alternatives considered:**
1. *Keep the label; argue "predictive processing" broadly.* → Rejected. This is exactly
   the "biological narrative compensating for weak evidence" failure mode (rule 19). An
   adversarial reviewer finds the direction inversion in one pass and the paper is dead.
2. *Rewrite the code to true top-down predictive coding, keep the claim.* → Rejected **as
   a Gate 0 action** (premature — see D-003), but retained as a candidate *arm*.
3. *Relabel honestly: feed-forward hierarchical residual extraction; make direction a
   tested factor rather than an assumption.* → **SELECTED.**

**Selected action:** the mechanism is defensible ML — "encode the component of level k+1
not linearly predictable from level k" is hierarchical decorrelation / novelty gating, a
real inductive bias worth testing. Keep it. Remove the Rao–Ballard framing and the
"validates predictive coding" claim from code, config comments, and `docs/paper/pcd_paper.md`.
Direction (low→high vs high→low) becomes an **experimental factor with both arms built**,
not an assumption asserted in a docstring.

**Expected consequence:** the paper loses its headline neuroscience claim and gains a
falsifiable question. Venue expectation shifts away from "we validated predictive coding."

**Reviewer objection (Agent B, preserved):** "Calling it 'novelty gating' is itself a
theoretical claim that needs evidence. Until residuals are shown to predict held-out
neural activity, they are *arbitrary task-useful features* (Risk 2) and deserve a name
with no semantics at all — 'cross-level residual', nothing more." — **Accepted.** Use
'cross-level residual' until identifiability is established.

**Revisit trigger:** if a top-down arm is built and shown to beat the feed-forward arm on
*held-out neural predictivity* (not just decoding), predictive-coding framing may be
reconsidered — carefully qualified, never as "validation".

---

## D-003 — Do not rewrite the architecture yet

**Date:** 2026-07-16 · **Gate:** 0 · **Status:** ACTIVE

**Context:** the obvious reflex after D-002 is to immediately implement top-down
recurrent precision-weighted PCD (the specification's Candidate A).

**Evidence against acting now:**
- The current model overfits by 82 pp (T4). Adding recurrence and precision heads adds
  capacity to a model that already cannot generalize. It would very likely make things worse
  while looking like progress.
- Residual identifiability is unestablished (Risk 2). Without a neural-prediction loss,
  a top-down arm's residuals would be *just as arbitrary* as the current ones — we would
  have rebuilt the same epistemic hole facing the other direction.
- 64% of input voxels bypass the hierarchy entirely (T7). Until that is quantified, we do
  not know whether the hierarchy carries *anything*. Rebuilding the hierarchy before
  measuring whether it matters is the wrong order.

**Selected action:** defer all architecture work to Gate 3, gated on Gate 2's thesis
selection and on a cheap attribution measurement of T7 (see `03_HYPOTHESES.md` H1).

**Reviewer objection (Agent C, preserved):** "Deferring is right, but the cheap
measurement must come first and it is *not* free — quantifying the level-3 bypass needs a
forward pass over val with the epoch-101 checkpoint. Budget it explicitly rather than
calling it 'analysis'." — Accepted; logged as the first Gate 1 experiment.

**Revisit trigger:** Gate 2 thesis selection.

---

## D-004 — Preserve and hash the pod checkpoints before any further pod work

**Date:** 2026-07-16 · **Gate:** 0 · **Status:** OPEN — action required

**Context:** both 2.68 GB checkpoints, all metrics, and `split.json` exist **only** on
pod NFS (T3). No hashes recorded. Dry-load never performed.

**Selected action (not yet executed):**
1. `sha256sum` both checkpoints on the pod; record digests here and in
   `05_EXPERIMENT_REGISTRY.csv`.
2. Dry-load `checkpoint_best.pt` and compare state-dict keys against a model constructed
   from `PCD_v4_8subject.yaml` — the audit found `strict=False` loading was introduced for
   PCD (`d99cf03`), which **silently tolerates key mismatches**. Untested.
3. Copy `split.json`, `manifest.json`, `training_log.csv` off-pod into the repo (small,
   text, safe to version). Leave the 2.68 GB weights on NFS.

**Explicitly NOT doing:** deleting anything, killing any process (none running), or
`git add`-ing 2.68 GB binaries.

**Reviewer objection (Agent E, preserved):** "`strict=False` is the single most dangerous
line in this codebase. It was added to 'handle buffer mismatches' — but it equally hides a
*silently half-initialised model*. A checkpoint that loads without error under
`strict=False` is not evidence that it loaded correctly. The dry-load must diff key sets
explicitly and assert the missing set contains only known buffers."
— **Accepted, and elevated:** this becomes a required Gate 1 test, not a nice-to-have.

**Revisit trigger:** none — execute at the start of the next session.

---

## D-005 — Record the metric fork before it becomes a reporting violation

**Date:** 2026-07-16 · **Gate:** 0 · **Status:** ACTIVE

**Context:** `val_r@1` = 16.4% (6 956-image gallery) vs `val_r@1_trial` = 3.3%
(19 236 trials) — a 5× fork (T11). All prior PCD reporting silently used the larger number.

**Selected action:** freeze **both** columns in the registry. Before any external
reporting, read the eval aggregation from source and declare the primary metric *once*,
in writing, with its gallery size. Never place PCD's val R@1 beside the frozen system's
77.2% SHARED1000 figure (T12) — different gallery, different protocol, PCD never ran on
SHARED1000.

**Reviewer objection (Agent D, preserved):** "Declaring the primary metric *after* having
seen both numbers is already contaminated. The only clean move is to declare it on
methodological grounds — which gallery answers the scientific question — and to report the
other alongside it permanently, labelled." — Accepted. Both columns are reported in all
tables, always.

**Revisit trigger:** Gate 2 endpoint definition.

---

## D-006 — Abandon the PCD research direction; select the NCD / imagery-transfer thesis

**Date:** 2026-07-16 · **Gate:** 2 · **Status:** ACTIVE

**Original proposal (Phase 2 mission):** pursue "NeuroPC — Identifiable Predictive Cortical
Inference for Cross-Subject Perception and Mental Imagery Decoding", with a tentative thesis
that a cortical inference model predicting both held-out neural activity and visual
representations learns more transferable, interpretable representations.

**Identified problem:** the systematic literature search (`13`) shows **every enumerated PCD
contribution, and both seed candidates A and B, are already published.**

**Evidence (`14` §1, all from live searches this session, abstract-level):**
- **Hi-DREAM** (arXiv 2511.11437, Nov 2025) uses PCD's *exact* ROI hierarchy — early/mid/late,
  V1/V2 → V3/V4 → category-selective — **and already ran PCD's random-and-reversed control
  experiment**, with a positive result. Our H2 is answered, by someone else.
- **BrainMCLIP** ran a 1 000-permutation ROI-label null (the rigorous version of our single
  `random.Random(42)` shuffle).
- **MindHier** owns hierarchy→CLIP-layer alignment; **DREAM** owns reverse visual pathways.
- **MindEye2** owns shared latent + per-subject projection; **MindTuner** owns low-rank
  (LoRA) subject adapters; **ZEBRA** owns shared/subject-specific disentanglement and claims
  *first* zero-shot, with public weights; **MindAligner** owns explicit functional alignment.
- **ESANN 2025** already shows predictive-coding dynamics improve fMRI predictivity over
  feedforward — seed Candidate A's core result.
- Canonical PC sources independently confirm Gate 0's **T6**: predictions descend, errors
  ascend. PCD's direction is wrong on the field's own definition.

**Alternatives considered:**
1. *Proceed with H1/H2 (hierarchy vs random controls) as the paper.* → Rejected: Hi-DREAM and
   BrainMCLIP already did it. H1 survives only as an **internal diagnostic**, not a contribution.
2. *Candidate A (predictive cortical inference).* → Rejected: novelty dead; we would arrive
   second into a laminar-fMRI/encoding-model literature where retrieval accuracy is not currency.
3. *Candidate B (cross-subject world model).* → Rejected: the most saturated axis in the field,
   moving monthly, against better-resourced groups, with our model at 16.4% R@1. We would lose.
4. *Candidate D (identifiability audit of brain-inspired decoders).* → Retained as **secondary**;
   a critique from a lab scoring 16.4% invites an obvious rejoinder, and it depends on
   reproducing external codebases.
5. *Candidate C, reframed: test whether a **neural-prediction constraint** explains the
   perception→imagery generalization gap.* → **SELECTED.**

**Selected action:** abandon PCD as a research direction. Build **NCD** (`18`) — ~20.5M params
vs PCD's 167M, per-ROI nodes, low-rank subject adapters, no `nsdgeneral_other` bypass, one
manipulated variable (`λ_neural`). Target the open problem NSD-Imagery (CVPR 2025) documented
but did not explain: complex architectures overfit to vision and transfer worse to imagery
than linear models. We test **why**.

**Expected consequence:** the project stops competing on SOTA (which it cannot win) and
competes on mechanism (which it can). Our 82 pp overfit gap stops being an embarrassment and
becomes the case study.

**Reviewer objection (panel, preserved — see `11_REVIEWER_REPORTS/phase2_thesis_review.md`):**
six objections are recorded and **none is resolved**. The load-bearing ones: an auxiliary
task is a regularizer, so the matched-regularization arm must be capacity-matched or the
whole result is confounded (O-1); "identifiability" is a misused term and is replaced by
"neural-prediction constraint" (O-2, adopted in `18`); a stimulus-shuffled control is needed
to rule out functional connectivity (O-3, adopted); the effect may be specific to a weak base
model (O-5); "transfer ratio" is a bad estimand and is replaced by a paired difference /
hierarchical interaction (O-6, adopted).

**Revisit trigger:** full-text reads of **NSD-Imagery** and **LEA**. If either already tested
an auxiliary neural-prediction objective, novelty is dead and this decision reopens
immediately (`17` §6).

---

## D-007 — Drop the "NeuroPC" name and all predictive-coding vocabulary

**Date:** 2026-07-16 · **Gate:** 2 · **Status:** ACTIVE

**Context:** the Phase 2 mission proposes "NeuroPC — Identifiable Predictive Cortical
Inference". Both halves are unusable. "Predictive Cortical" reasserts the exact label Gate 0
disproved (T6) and that Hi-DREAM owns architecturally. "Identifiable" is a statistical term
of art meaning unique up to a known transformation — masked-ROI prediction does not deliver
that, and Reviewer 2 flagged it as smuggling a strong term in for a weak property.

**Selected action:** the model is **NCD (Neural-Constrained Decoder)**; config prefix `NCD_`;
source `src/fmri2img/models/neural_constrained_decoder.py`. We claim a **neural-prediction
constraint** and nothing stronger.

**Reviewer objection (Agent B, preserved):** "Even 'neural-constrained' will be read as a
biological claim by a sympathetic reviewer. The safest name is descriptive of the mechanism —
masked-ROI auxiliary prediction — and the safest claim is that it is an auxiliary objective
whose *effect on transfer* we measure." — **Partially accepted:** the name stands for
brevity; the *claims* are held to exactly this standard throughout `18`–`20`.

**Revisit trigger:** none. Do not resurrect "predictive cortical" without new evidence.
