# 27 — Phase 2.5 Pilot Matrix and Gate Decision

**Date:** 2026-07-16 · **Decision: `CLEARED_FOR_MULTI_SHIFT_PILOT`** — scoped and conditional (§5).

---

## 1. The structural insight that unblocks this phase

**Four of the six shifts run on data already on the pod.**

| Shift | Data | Status |
|---|---|---|
| 1 In-distribution | NSD | **available** |
| 2 Stimulus OOD ★ primary | NSD-Synthetic | **not obtained** |
| 3 Cognitive state (zero-shot) | NSD-Imagery | **not obtained** |
| 4 Reduced data | NSD | **available** |
| 5 Noise / missing info | NSD | **available** |
| 6 Subject (LOSO) | NSD | **available** |

The matched-control pilot — the thing that decides whether a neural-target effect
exists at all — needs **only NSD**. External data gates the *confirmatory* OOD
evaluation, not the pilot. This is why the phase can proceed while data access is
pending, and it is the single most useful thing this audit established.

## 2. Cheap discriminating tests, in execution order

| # | Test | Cost | Can kill | Status |
|---|---|---|---|---|
| 1 | Verify official access: NSD-Synthetic (CC-BY 4.0), NSD-Imagery (CC-BY-NC-ND 4.0) | 0 GPU | shifts 2–3 | **NOT DONE** |
| 2 | Reproduce published NSD-Imagery baseline protocol | low | endpoint validity | NOT DONE |
| 3 | **Perception OOD kill test** on the 12 evaluable vision stimuli | eval only | shift 3 | **NOT DONE** — blocked on data |
| 4 | NSD-Synthetic baseline eval from PCD ep98 ckpt | eval only | shift 2 | blocked on data |
| 5 | **Implement ARM-C and ARM-F** | 0 GPU | design validity | **DONE — this phase** |
| 6 | One-subject low-data pilots: A, B, C, F | ~8 GPU-h | **the hypothesis** | **cleared to run** |
| 7 | Estimate effect and variance | 0 GPU | powers everything | after 6 |
| 8 | Update compute plan from measured throughput | 0 GPU | — | after 6 |

## 3. Pilot matrix (E-P1)

| Run | Arm | Objective | λ | Subject | Seeds | Shifts evaluated |
|---|---|---|---|---|---|---|
| E-P1-A | A | `none` | 0.0 | subj01 | 2 | 1, 4, 5 |
| E-P1-B | B | `masked_neural` | 0.5 | subj01 | 2 | 1, 4, 5 |
| E-P1-C | C | `shuffled_neural` | 0.5 | subj01 | 2 | 1, 4, 5 |
| E-P1-F | F | `none` + tuned reg | 0.0 | subj01 | 2 | 1, 4, 5 |

Configs: `configs/experiments/NCD_v1_arm{A,B,C,F}_pilot_1subj.yaml`, mechanically
generated from one base and **asserted to differ only in permitted keys**
(`tests/test_ncd_arm_config_parity.py`). Est. **~8 GPU-h total**.

**Pilot read-out — what each comparison buys:**

- **B vs A** — is there any effect at all? If not, stop.
- **B vs F** — **the decisive one.** If B ≈ F, the effect is generic regularization
  and the thesis reduces to `24` §6's most likely outcome.
- **B vs C** — is stimulus-specific neural content doing the work, or marginal
  statistics / connectivity?
- **Variance across 2 seeds × 1 subject** — the *only* legitimate input to the power
  analysis for shifts 4/5/6. `20` §6's power numbers cannot be written before this.

**Promotion to 4-subject:** B must beat **both** A and F on ≥1 shift with a
subject-consistent direction. **B ≈ F ends the treatment line and we report that.**
**Eight-subject confirmatory is NOT cleared** and remains gated on the pilot
identifying a specific neural-target effect.

## 4. What was implemented this phase

| Deliverable | Artifact |
|---|---|
| Matched auxiliary objectives A–H | `src/fmri2img/models/auxiliary_objectives.py` |
| Arm wiring (target selection, never head capacity) | `neural_constrained_decoder.py`: `aux_objective`, `auxiliary_loss`, `_aux_target` |
| Deterministic target shuffling (C) | `DeterministicImagePermutation` — derangement, whole-image, seeded, manifest-recorded |
| Random pseudo-ROI grouping (G) | `make_pseudo_roi_groups` — size-matched, disjoint, seed-sensitive |
| Fixed random targets (D) | `RandomTargetBank` — variance-matched, buffered |
| Parity assertion | `assert_param_parity` (≤2% tolerance) |
| Control-specific gradient tests | `tests/test_ncd_matched_controls.py` (27) |
| Config-diff validation | `tests/test_ncd_arm_config_parity.py` (13) |

**62 tests passing.** All eight arms are **exactly** parameter-matched (not merely
within tolerance).

**The config-diff test earned its place immediately:** it caught ARM-F carrying a
`stochastic_depth` key the other arms lacked — a config-shape drift that would have
been invisible in results and would have confounded the decisive B-vs-F comparison.

## 5. Decision — `CLEARED_FOR_MULTI_SHIFT_PILOT`

**Cleared for:** the one-subject matched-control pilot (A/B/C/F) on NSD, and shift
regimes 1, 4, 5 (+6 at 4-subject scale).

**Explicitly NOT cleared:** the 8-subject confirmatory program; the ~450 GPU-h plan;
any OOD or imagery claim.

**Why this clears the bar the mission set** — *"novelty and matched-control design
strong enough that a positive result would survive the generic-regularization
objection"*:

1. **Novelty repaired.** Spera et al. (2026) closed imagery *adaptation*; our
   zero-shot, perception-only, multi-shift setting is a different problem (`23` §4),
   and imagery is demoted to one sealed test of five.
2. **The generic-regularization objection is now answerable by construction.**
   ARM-F is tuned under the same budget; ARM-D tests whether *any* target
   regularizes; ARM-E tests generic vs anatomical reconstruction. Parity is
   test-enforced, not asserted. A positive ARM-B that also beats F, D and E is not
   dismissible as "you added a regularizer."
3. **The most likely outcome is pre-committed as publishable** (`24` §6, `25` §4):
   B ≈ F → *"auxiliary regularization improves robustness; neural target structure
   is not specifically supported."*

**Conditions on the clearance:**
- **NCD is not yet wired into `create_model`/the training loop.** The pilot cannot
  execute until it is. Engineering task, not a scientific gate.
- Full PDF reads outstanding: **Spera et al.** (does it include a zero-shot arm? if
  yes, `23` §6 reopens and this clearance is void), **LEA**, **Hi-DREAM**.
- Data access unverified for shifts 2–3 → those remain `DATA_ACCESS_UNVERIFIED`.
- `20` §6 power numbers must be rewritten from measured pilot variance before any
  confirmatory run.
