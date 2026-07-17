# 19 — MINDCOMPILER Session Handoff

**Written:** 2026-07-17 · **Branch:** `research/mindcompiler-neural-state-operators` @ `c1095de`

---

## What this session did

**Executed the repository transition (§3.1–3.3) and nothing else.** That is the whole
delivery, and it is complete and verified.

1. **Reconciled** local / remote / pod. Direct `git ls-remote` confirms `origin` held
   `d4531e9` = local HEAD: **all prior work was already published; nothing was unpushed.**
   (The remote-tracking ref reported parity and I verified it against the server rather than
   trusting it — the F-005 lesson.)
2. **Archived** PCD and NCD → `docs/research/archive/PCD_NCD_TERMINATION_MEMO.md` (`c1095de`).
   Nothing deleted, squashed, rewritten, or force-pushed.
3. **Created and published** `research/mindcompiler-neural-state-operators`, upstream set,
   **server-verified** by `ls-remote`.
4. Scaffolded `00_EXECUTIVE_STATE.md` and this file.

**No GPU was used. No process was started or killed. No file was deleted.**

## What this session did NOT do — and must not be mistaken for done

**Gate M0 is not complete.** §5 literature review, §6 formal theory, §7 dataset matrix,
§8 baselines, §18 synthetic recovery: **none started.** The remaining OS files
(`01`–`18`) do not exist yet.

> **No novelty verdict issued.** Issuing one before the review would be exactly the failure
> mode this program inherits a whole audit trail about. The permitted set is in `00` §Gate M0.

## Exact next action

**§5 frontier literature review.** Zero GPU, blocks everything, and is the single highest-VOI
task in the program — it can return `NOVELTY_INSUFFICIENT` or
`PARTIALLY_OVERLAPPING_REQUIRES_REDESIGN` and save the entire compute budget.

Search families, in priority order (the first three are the most likely to contain a
pre-emption):
1. **Hyperalignment / shared response models / cross-subject neural translation** — the
   closest existing machinery to `S_p`.
2. **Neural operators, Koopman operators, latent dynamics, nonlinear system ID** — the
   closest existing machinery to `T_s`.
3. **Causal representation learning + counterfactual prediction** — the closest existing
   framing of the flagship endpoint.
4. Perception vs imagery; memory reinstatement; content-state disentanglement; optimal
   transport for neural data; representational geometry; multimodal/fMRI–MEG fusion;
   real-time fMRI and neurofeedback; intracranial perception–imagery.

For every relevant work record (per §5): whether it **learns transformations** or only
**aligns/classifies** states; whether transformations **generalize to unseen content**;
whether it **predicts measured target-state neural activity**; whether it tests
**composition or inversion**; retrospective vs **prospective**.

**Do not claim novelty because no paper uses the name.** Novelty must be conceptual and
experimental.

## The honest prior a successor should hold

Two things, stated now, before any evidence, so they cannot be rationalised away later:

1. **The semantic shortcut (H6) is the most likely explanation of any positive transport
   result.** Build B5 before believing anything. This program has already watched a
   plausible-looking figure come out of random weights (F-002).
2. **The public data may be too thin for the flagship.** Neural counterfactual prediction
   requires *the same content, measured in multiple states, paired at trial level*.
   NSD-Imagery is 4 subjects and 18 stimuli, and Spera et al. showed a **stronger** decoder
   is **at chance** zero-shot on it. `PROSPECTIVE_PROGRAM_REQUIRED` is a **plausible and
   legitimate M0 outcome**, not a failure — and §5 of the mission explicitly says not to stop
   the project for it, but to separate what can be built and tested now from what needs
   collaboration.

## State

| | |
|---|---|
| Branch / HEAD | `research/mindcompiler-neural-state-operators`; bootstrap `3c4ebcd`, upstream set, server-verified |
| Parent | `feature/predictive-cortical-decoder` @ `c1095de` — archived, published, intact |
| Working tree | clean but for `docs/CLAUDE_MEGA_PROMPT.md` (user-authored, deliberately untracked) |
| Tests | 115 program tests passing (inherited); 10 pre-existing env failures, verified unrelated |
| Pod | idle, no processes, GPU 0%. **Pod HEAD `b96affa` matches no commit and is NOT synced to this branch.** Sync before any pod work. |
| GPU spent on MINDCOMPILER | **zero** |

## Do not reopen without new evidence

Everything in the archive memo §5 (FORBIDDEN claims) carries forward and does not expire.
PCD is not predictive coding; per-level uncertainty is invalid; **E-P1 never ran and no NCD
scientific result exists**; no "first" claims; cycle consistency alone is not evidence; a
reconstruction is not a neural counterfactual; simulation is not prospective validation.
