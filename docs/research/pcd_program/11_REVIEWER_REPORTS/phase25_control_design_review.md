# Phase 2.5 — Adversarial Review of the Control Design and Novelty Repair

**Date:** 2026-07-16 · Five perspectives, run sequentially, scored independently.
**Question put to the panel:** *would a positive ARM-B result now survive the
generic-regularization objection?*

---

## Reviewer 1 — Top ML venue

**The control family is the first thing in this project I would not reject on sight.** ARM-F
tuned under the same budget, ARM-D for "any target", ARM-E for generic reconstruction, exact
parameter parity enforced by test rather than asserted in prose. That answers O-1 *by
construction* rather than by argument.

**My remaining objection is now narrow but sharp: an under-tuned ARM-F is a straw man.** If
ARM-F gets one lazy sweep and ARM-B gets careful attention, the comparison is worthless and I
will assume the worst. **Log every ARM-F trial in the manifest and report the tuning budget
per arm in the paper**, or the parity claim is unfalsifiable.

**Second:** the pilot is 1 subject × 2 seeds. That cannot distinguish B from F for anything
but a large effect. Do not over-read it — it is a go/no-go, not evidence.

**Score: the design is now publishable-if-positive. Weak accept on design.**

## Reviewer 2 — Computational neuroscience

**The novelty repair is honest and I accept the distinction.** Spera et al. fit on imagery;
sealing it is a genuinely different question, not a rebranding. And demoting imagery to one
of five shifts is what makes it a thesis rather than a benchmark entry.

**ARM-H is the arm I care about** and it is the one most likely to deflate the story. If
predicting an ROI from *itself* works as well as predicting it from *other ROIs*, then the
"cortical context" framing is decorative and what you have is per-ROI autoencoding. That
would be a real finding — **make sure you would actually report it**, because the temptation
to bury it will be strong.

**Objection:** you still call it a "neural-prediction constraint", which invites the reader to
think the constraint is *neural* in some meaningful sense. ARM-D (random targets) is what
licenses that word. **If B ≈ D, drop "neural" from the name of the thing entirely.**

**Score: accept the design; hold the terminology to the ARM-D result.**

## Reviewer 3 — Neuroimaging methodology

**The NSD-Synthetic audit is the strongest document in this program.** Recognising that 82% of
the stimuli have no CLIP-semantic content — *before* running anything and reporting
near-chance as an OOD finding — avoided a self-inflicted wound. The graded contrast/phase axis
is a much better test than the binary one, and the neural-predictivity endpoint on all 284 is
the right instinct.

**But 52 confirmatory stimuli is thin**, and the slope has 5 contrast levels and 4 phase
levels. That is 9 points per subject per family. **The slope estimate will be noisy, and
"margin grows with OOD degree" is a strong prediction to hang on 9 points × 8 subjects.**
Model it hierarchically or do not claim it.

**Also unresolved:** R-20. Your CLIP cache was built for colour natural scenes. Nobody has
verified it produces sane embeddings for grayscale Mooney images or line drawings. **That is a
silent-failure risk and it sits upstream of every synthetic number.**

**Score: borderline. Verify R-20 before clearing anything that touches synthetic.**

## Reviewer 4 — BCI

**Sealing imagery is the right call and it is the only version of this that interests me.** An
adaptation method needs target-state data, which in a real BCI is exactly what you do not have
at deployment. A perception-only constraint that transfers zero-shot is the deployable claim.

**Objection:** shifts 4/5/6 (reduced data, noise, subject) are where BCI relevance actually
lives, and they are all runnable on data you already have. **Do not treat them as filler for
the imagery story.** If B beats F on reduced-data and noise robustness with n=8, that is a
stronger and more useful paper than anything the 4-subject imagery test can deliver.

**Score: accept, and I would reweight the paper toward shifts 4–6.**

## Reviewer 5 — Statistics

**The unit-of-inference correction (subject AND stimulus, seeds are not stimulus evidence) is
correct and was overdue.**

**The exact-test ceiling is the honest bit that most authors would hide:** with n = 4 the
sign-flip permutation distribution has 16 points, so p ≥ 0.0625 is the floor. Stating that
rather than reporting an asymptotic p is right. **Now act on it:** it means shift 3 can never
be confirmatory on its own. **Say so in the paper, not just in this repo.**

**My objection stands from Phase 2 and is now worse:** you have six shifts × multiple arms.
**The multiplicity structure is not yet written down.** FDR within the synthetic families is
specified; the cross-shift family is not. Are the 6 shifts one family? Is B-vs-F tested once
per shift? **Until that is fixed, "improves ≥2 shifts" is a garden of forking paths.**

**Score: reject the analysis plan as specified. The cross-shift multiplicity family must be
declared before the pilot reads out.**

---

## Panel outcome

**Consensus: the control design answers O-1 and the novelty repair is accepted.** The design
is strong enough that a positive ARM-B would survive the generic-regularization objection —
**conditional on ARM-F being tuned in earnest.**

**Unresolved objections, carried forward as blocking:**

| # | Objection | Owner | Blocks |
|---|---|---|---|
| **O-7** | Cross-shift multiplicity family undeclared; "improves ≥2 shifts" is forking paths until fixed | R5 | **any confirmatory read-out** |
| **O-8** | ARM-F tuning budget must be equal and logged, or parity is unfalsifiable | R1 | any positive claim |
| **O-9** | R-20 unverified: CLIP cache never checked on grayscale/Mooney/line-drawing stimuli | R3 | anything touching NSD-Synthetic |
| **O-10** | If B ≈ D, the word "neural" must be dropped from the mechanism's name | R2 | terminology, post-pilot |
| **O-11** | Slope claim rests on 9 OOD points × 8 subjects; model hierarchically or drop it | R3 | the C-015 claim |
| **O-12** | ARM-H deflation (per-ROI autoencoding suffices) must be reported, not buried | R2 | integrity |

**O-7 is the one that matters most and it is cheap to fix:** declare the cross-shift
multiplicity family before the pilot reports. It is not fixed in this phase.

**Panel note on the clearance:** the pilot is cleared because it is 1 subject and ~8 GPU-h and
its purpose is go/no-go. **None of these objections block the pilot; O-7 and O-8 block reading
the pilot as evidence for anything beyond go/no-go.**
