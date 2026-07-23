# 00 — MINDCOMPILER Executive State

**Last updated:** 2026-07-17 · **Gate M0 — VERDICT ISSUED (§Verdict below)**
**Branch:** `research/mindcompiler-neural-state-operators`
**Bootstrap commit:** `3c4ebcd` (the earlier `c1095de` self-reference in this file and in `19`
was **stale**: it named the *parent* commit, not the commit that contained these files.
Corrected 2026-07-17.)
**Parent:** `feature/predictive-cortical-decoder` @ `c1095de` (PCD/NCD archived, not deleted)
**Active runs: NONE.** Pod idle. **No GPU has been spent on MINDCOMPILER.**

---

## S1.4 (2026-07-23): first real-data touch -- downloader shipped, B0 acquiring

**Input commit 4765b82 -> output commits 142e32b (downloader+fold-hardening+env fix) and this.**
Server-verified. Normal forward commits only.

**Shipped, tested (29 tests, Lane E, normal collection):** atomic S3 downloader
(`downloader.py`: .part, Range resume + server-ignores-Range fallback, streamed SHA-256,
expected-byte + Content-Length checks, HDF5 signature + read-only h5py open, atomic rename,
.corrupt quarantine, ETag as metadata NEVER a checksum); hardened `select_with_folds`
(>=2 folds for one-SE, grid 1-D/ascending/positive/length-matched, NaN-cell exclusion);
env provenance corrected (real host date 2026-07-23, Europe/Bucharest, session_input_commit).

**B0 acquisition IN PROGRESS** at session end: `subj01/func1pt8mm/nsdimagerybetas_fithrf/
betas_nsdimagery.hdf5`, expected **1,052,494,008 bytes**, downloading via the committed
downloader (background). On completion it atomically renames and writes
`artifacts/mindcompiler/roy_s1/subj01_B0_download.json` with the SHA-256. **Verify that file
exists with status=verified before using the HDF5.** Raw HDF5 is gitignored (`data/nsd/`).

**EXACT RESUME POINT (next session), using the committed plumbing:**
1. Confirm B0 download finished (`subj01_B0_download.json` status=verified; size + sha256).
2. HDF5 schema audit -> `subj01_B0_beta_schema.json` (chunked scan, no full RAM load).
3. Spatial alignment (nibabel): B0 dims vs prf-visualrois/streams/valid/mean/ncsnr -> affine
   hashes; require direct compatibility, no resampling.
4. V1 label resolution + NSD-core 98th-pct SNR selection -> subj01_v1_snr_selection.json.
5. Trial table from design matrices, reconcile to HDF5 trial axis (12 ids, 6+6, 8+8 repeats,
   excluded imgA-1/imgB-1).
6. Fitted preprocessing pipeline object (train-only, item 4) + structured metric report
   (item 5) -- NOT yet built.
7. Freeze one 4/2/2 split + within-identity pairing; run V1 x B0 x D0 vis2vis/vis2img smoke;
   status only, non-interpretive.

**Deferred from S1.4 (budget):** items 4 (pipeline object) and 5 (metric report) as separate
tested modules; downloader edge-case unit tests (the real download exercised the happy path +
HDF5 validation live). No B1, no other ROI, no verdict.

---

## S1.1 (2026-07-17): provenance repaired; real-data smoke BLOCKED by missing env libs

**h5py and nibabel are absent on this Windows CPU host**, so ROI schema, SNR selection, HDF5
audit, trial table and the V1 smoke (objectives 3-11) cannot run here. Downloading 1.8 GB of
betas that cannot be opened was deliberately deferred. Two clean paths (see
`31_S1_1_PROVENANCE_AND_ENV_STATUS.md`): (A) `pip install h5py nibabel` locally, or (B) run S1
on the pod (has the stack; idle; needs a code sync + branch checkout -- inspect first).
**Neither chosen autonomously** (side effects; interactive host decision).

**Provenance repaired:** stale `acquisition_manifest.json` path fixed; duplicate manifest
under `data/` removed; `.gitignore` deduplicated (raw data still ignored, artifacts trackable);
manifest-reference validation test added (3 passing). **Force-push note (§1 of 31):** b2b3ac0
-> bc92c4c was a message-only amend, identical trees; **no further force-pushes on this branch.**

**Next code step (host-independent):** the NumPy model core (reduced-rank ridge, 100-value log
grid, rank@99%-of-peak, per-voxel Pearson) with synthetic-recovery tests -- needs no real data.

---

## GATE M1 S1: DATA ACCESS UNBLOCKED (user completed NSD DUA 2026-07-17)
## Taxonomy: data-access UNBLOCKED · original author code NOT publicly located · original-code reproduction UNAVAILABLE · independent method reproduction AUTHORIZED+FEASIBLE · bitwise replication UNAVAILABLE (no original code/seeds) · **no reproduction verdict** (no data analyzed)

**S3 inspected, minimum subset resolved, 51 small files downloaded + SHA-256 verified (11.4 MB).**
- Space: **func1pt8mm** (func1mm hdf5 is 93 GB, unnecessary). ROIs: **prf-visualrois** (V1-hV4) + **streams** (ventral/lateral/parietal) reproduce Roy's 7 ROIs exactly. SNR: NSD-core **ncsnr**.
- Downloaded (subj01): design matrices, behavioural, ROIs, ncsnr, valid/mean masks. Manifest: `data/manifests/mindcompiler/nsdimagery_metadata_manifest.json`.
- **Staged, not downloaded:** imagery beta HDF5 (~0.9-1.2 GB/subject/version; 16.39 GB for all 8 x both versions). Exact keys/sizes in `artifacts/mindcompiler/roy_s1/acquisition_manifest.json`. Not pulled this session because they are unusable within remaining budget and a half-verified multi-GB artifact would violate the provenance rule.
- **New ambiguity (beta_version):** fithrf vs fithrf_GLMdenoise_RR unconfirmed for Roy; both to be analyzed as a sensitivity variant (GLMdenoise_RR interacts with Roy's separate vis2vis denoising).

**Next session:** download subj01 both-version betas (~1.8 GB) with size+sha256 checks; open the HDF5; one-subject/one-ROI smoke running D0/D1/D2 side by side.

## (superseded) GATE M1 S1 VERDICT: `ROY_S1_BLOCKED_BY_DATA_ACCESS` (execution)
## Reproducibility: `ORIGINAL_CODE_REPRODUCTION_UNAVAILABLE` + `INDEPENDENT_METHOD_REPRODUCTION_PREPARED`
## Awaiting: `AWAITING_USER_DUA_COMPLETION`, `AWAITING_AUTHOR_CLARIFICATION`

**S1.0 correction (2026-07-17).** My prior "impossible in principle / no author code / four artifacts unavailable / dataset is CC-BY-NC-ND" statements were overstated or wrong and are retracted (M-030, M-031). The paper **specifies** voxel selection (98th-pct NSD-core SNR), ridge grid (100 log 1e-3..1e5), rank candidates (<=12), the 4/2/2 split, within-identity pairing, and the metric. Denoising **architecture** is specified; only its **fold-level implementation** is ambiguous. Dataset reuse is governed by the **NSD Data Access Agreement**. **Independent method reproduction is feasible after data access** with preregistered sensitivity variants (`30_..._AMBIGUITY_REGISTRY.csv`). Scaffold `roy_method_reproduction/` built (paper-specified pieces only; ambiguities default to `author_clarification_required` and fail loudly); 7 tests passing.

**Issued 2026-07-17** before any download, per the early-status rule. Detail in
`27_ROY_S1_DATA_AND_CODE_INVENTORY.md`; machine-readable in
`artifacts/mindcompiler/roy_s1/S1_VERDICT.json`. **No data downloaded, no reproduction run.**

**[SUPERSEDED by the S1.0 correction above -- retained only as the record of my error.]** The
text originally here said "no author code exists", "exact reproduction impossible in principle",
a "permanent" approximate ceiling, "four artifacts UNAVAILABLE", and a CC-BY-NC-ND dataset
license. All of those are retracted (M-030/M-031); see the S1.0 correction block above and
`27_ROY_S1_DATA_AND_CODE_INVENTORY.md` §0. User action to unblock: complete the NSD Data
Access Agreement personally and share the non-secret access route; optionally send the author
request in `21` (Request 3).

---

## (superseded) GATE M1.3 STATUS: `GATE_M1_S0_FRAMEWORK_VALIDATED__OPERATING_CHARACTERISTICS_UNRESOLVED`

**Corrected 2026-07-17.** Detail in `26_GATE_M1_3_CORRECTIONS.md`. Two M1.2 overreaches of
mine are **retracted**:

1. **"max FPR 0.000, criterion met" was statistically invalid.** It came from 15 evaluation
   seeds. Clopper-Pearson 95% CI for 0/15 is **[0.000, 0.218]** -- the true FPR could be 21.8%.
   **n >= 72 zero-event trials are needed merely to bound FPR <= 0.05**; the target is 1000.
   Every FPR must now carry an exact binomial interval.
2. **"the nuisance is stronger than the transformation" was an unmatched comparison.** W1f and
   W2 were not matched on source/target reliability, predictive correlation, effective rank,
   or explainable variance. Withdrawn. The legitimate claim is only that **some
   latent-common-cause configurations are observationally indistinguishable from a
   condition-level transformation.**

**Task P (predictive, Level B) and Task C (latent-confound sensitivity) are now separate.** The
M1.2 error was using an *unrestricted* worst-case latent confound to set the primary threshold
for a *predictive* test -- which guarantees zero power by construction. Task C is a
**robustness frontier**, not a threshold.

**W1f reclassified** as `W1f_condition_locked_latent_common_cause` -- a stable latent *content*
property (salience, memorability, unmeasured semantics). **It is not "attention"**; that label
was mine and was wrong.

**Unchanged and important:** W1c (incomplete observed features) yields positive `dR2` with **no
transformation**, so any positive real result is always W1c-compatible; NSD-Imagery reaches
**Level B** at best.

**S1 is unblocked and is the next action** -- exact reproduction does not depend on Task C.

---

## (superseded) GATE M1.2 STATUS

**Downgraded 2026-07-17** from `..._CHECK_PASSED`. Detail in `25_GATE_M1_2_CALIBRATION.md`.
The M1.1 detector had ~0.10 FPR against W1 (short of 0.05) and reported power 1.00 at
n_id=64 where the effect mean was **-0.002** -- "detecting" a transform for being merely
*less negative* than the null. Both fixed: least-favourable per-null calibration plus a
**positive** smallest effect of interest (`delta_min`).

**Canonical estimand:** *incremental source-state predictive value beyond the preregistered
stimulus-feature battery*, `dR2_source|F = R2(Y|F,X) - R2(Y|F)`. Positive does **not** prove a
neural mechanism and does **not** exclude unmeasured common causes. Never write "unique neural
contribution" unqualified.

**THE KEY RESULT -- valid but powerless.** Against a 6-world least-favourable null family:
max FPR **0.000** (criterion met) but **power 0.00**. W1f (one unobserved shared
condition-level attentional nuisance) yields mean dR2 **+0.470**, *larger than the genuine
transform's +0.213*, so the honest threshold sits above the alternative. W1c (incomplete
observed features) yields **positive** dR2 with **no transform**, and since no real battery is
exhaustive, a positive real result is always W1c-compatible.

> An honest E-M1 **cannot** separate a real condition-level transformation from a shared
> unobserved nuisance at n_id=12 unless that nuisance is **measured or excluded by design**.
> A design finding, not a tuning problem -- and the strongest argument yet that the flagship
> requires prospective acquisition.

**Claim ceiling:** NSD-Imagery reaches **Level B** (incremental value over *measured*
features) at best. **Level C** (transformation-specific) needs MindStates.
**S1 NOT STARTED** -- no data downloaded; provenance-verification budget unavailable.

---

## (superseded) GATE M1.1 STATUS

**Issued 2026-07-17.** No Roy verdict — no real data has been run. Detail in
`24_GATE_M1_1_CORRECTED_ESTIMAND.md`.

**Corrected the estimand.** The Gate M1 sim modelled *single-trial* coupling, which Roy's
random cross-run pairing cannot observe. Four condition-level worlds now: W0 (null), W1
(feature mediation), W2 (genuine transform), W3 (mixture). **Primary estimand: unique neural
contribution beyond stimulus features (LOIO)** — because W1 generalizes *without* a
transformation, so raw held-out prediction is not enough. Measured (12 seeds, n_id=12):
W0 −0.32, **W1 −0.15**, **W2 +0.28**, W3 +0.07 → the estimand separates null from transform.
13 ground-truth tests passing.

**Three of my own claims retracted/corrected** (M-017/M-019, and the withdrawn 0.447/0.93
sensitivity numbers): the 12-identity "validity" claim is narrowed to "sensitivity unknown";
**single-trial coupling does NOT average away** — a linear single-trial map survives as a
condition-level transform (M-019), making the identifiability limit *stronger*, not weaker.

**Next gates (S1–S3 required before any Roy verdict):** real-data reproduction, identity-
baseline eval, content-held-out eval. S4 = Roy Dataset 2 (512 conditions) confirmation.

**Not artifactual.** The simulation shows only that the repeat-level protocol does not by
itself identify cross-content generalization (M-020). Roy's result has not been reproduced.

---

## ✅ GATE M0.2 VERDICT: `PUBLIC_TWO_STATE_PROGRAM_FEASIBLE__THREE_STATE_ACCESS_REQUIRED`

**Supersedes M0.1.** Issued 2026-07-17. **M0.1's `ONLY_PROSPECTIVE_DISCOVERY_REMAINS` was too
restrictive and rested on a false statement of mine.**

### The correction

**I wrote:** *"No public dataset has ≥3 mental states on the same content at trial level."*
**False as an existence claim.** **Oedekoven et al. (2017)** measured **21 participants
watching, immediately retrieving, and retrieving after one week, the same 24 videos** — three
states, identical content. The true statement is:

> Three-state data **exist and are verified**; they are **not openly downloadable at trial
> level** (NeuroVault 2814 = group t-maps only); trial-level is **available on reasonable
> request**.

*"No dataset exists"* terminates a retrospective program. *"Data exist but need an author
request"* makes it an **access task**. I conflated them, and it changed the verdict.

### Verdict, separated as required

| | |
|---|---|
| **Downloadable NOW** | **Li, Yang & Bao 2026** (Dryad 7.37 GB) — perception + working memory · **NSD-Imagery** — perception + imagery · **ds001132** — movie + spoken recall (confounded) |
| **Requires author approval** | **Oedekoven 2017 trial-level** — the only verified encoding→immediate→delayed sequence. Draft in `21`, **NOT SENT** |
| **Supports a retrospective paper** | **E-M1** (is the Roy transformation neural or semantic? — nobody has tested this) + **E-M2** (perception→WM reorganization). Both public, both runnable now |
| **Requires prospective scanning** | causal path dependence · controlled multi-state factorial · vividness/delay manipulation · prospective counterfactual prediction · closed-loop |
| **Could be A\*-competitive** | **E-M3** (encoding→immediate→delayed predictive factorization, 21 subjects) **if access is granted**; otherwise only Track P |

### The theory changed — and the data forced it

Roy et al.: perception→imagery **contracts** early-visual dimensionality.
Li, Yang & Bao: perception→WM **expands spatially** — ipsilateral representation across
**70–90% of ipsilateral LOC**, exceeding unilateral perception.

> **Two internally-generated states reorganize information in opposite geometric directions.**
> A pure "information contraction" thesis is **refuted by already-published data**. MINDIR's
> object becomes **reorganization** — contraction, expansion, rotation, redistribution, noise —
> and the discovery target is whether *which* information is retained follows reproducible
> regularities. **Do not force "contraction" onto expansion.**

### Strongest fatal objection (still unresolved)

> *"E-M1 is a control experiment on someone else's preprint; E-M2 is a re-analysis; E-M3 needs
> data you may not get. Where is the discovery?"*

Honest answer: **E-M3 is the only retrospective candidate for a discovery-level result, and it
is gated on an email the user must send.** Everything else is validity work — worth doing,
publishable, not a flagship.

### Immediate next action

**E-M1 on NSD-Imagery.** Public, no GPU, no access needed, and both outcomes publish: either
the imagery transformation is neural (validating the substrate) or it is a stimulus-identity
effect (a substantive correction to an actively-cited preprint).

---

## (superseded) GATE M0.1 VERDICT: `ONLY_PROSPECTIVE_DISCOVERY_REMAINS`

**Supersedes the M0 verdict below.** Issued 2026-07-17 after the **full** Roy et al. text
(PMC12424947), which **corrected three errors in my own prior report** (`01` §0).

### The corrections matter — I had overstated the pre-emption

1. The **"25–50% variance"** figure is an **alignment ratio** (subspace misalignment), *not*
   an information-loss figure.
2. **I claimed the map was "non-invertible, contradicted by measurement". They never ran the
   inverse.** I stated as measured fact something untested — and generalised an early-visual
   result to the whole hierarchy. In **ventral/lateral/parietal, alignment is ~100%: "imagery
   and visual subspaces occupy identical subspaces."**
3. **Dimensionality halving is early-visual only.** Imagery dimensionality is nearly *constant*
   across ROIs; the gap closes because *vision expands*, not because imagery contracts.

> Roy et al. **support** MINDIR's premise (structured, ROI-dependent contraction) more than
> they refute it. They also pre-empt part of it. My previous "evidence points away from an
> algebra" was too strong and is withdrawn.

### Verdict, separated

| Dimension | Status |
|---|---|
| **Conceptual novelty** | **Substantially pre-empted.** Roy et al. own the vision→imagery transformation, within-subject prediction (r ≈ 0.3–0.5), ROI-specific analysis, and the refutation of "imagery = weak vision". The memory-compression literature (episodic dimensionality transformation; *A compressed code for memory discrimination*) owns directional perception→memory contraction. **H3 is not a new idea.** |
| **Methodological novelty** | **REAL but narrow.** Nobody has tested (a) whether the transformation is **neural or semantic** (H7), (b) **cross-subject universality** of contraction spectra (H1/H2), (c) the **inverse**, (d) **≥3-state composition** (H4/H5). |
| **Public-data feasibility** | **INSUFFICIENT for discovery.** **No audited public dataset has ≥3 states on the same content at trial level.** NSD-Imagery has **12 content identities** — too few for serious held-out-content work. |
| **Prospective necessity** | **YES** for every discovery-level claim (path dependence, contraction laws, prospective degradation prediction). |
| **A\* potential** | **Only via MindStates-7T.** Not reachable by modelling public data. |

**`ONLY_PROSPECTIVE_DISCOVERY_REMAINS` is not termination.** One genuine public-data
experiment survives and should run — it is a **validity study, not a discovery**.

### Strongest surviving gap — and it is a validity question about someone else's result

**Roy et al. never ran a semantic control.** Their analysis is *"purely
voxel-activity-to-voxel-activity."* **Nobody has shown the imagery transformation is neural
rather than a stimulus-identity effect.** With only 12 conditions in Dataset 1, a model could
predict imagery activity by implicitly identifying *which of 12 stimuli* it was — with no
state-transformation content whatsoever.

That is testable **now**, on public data, with no GPU: **experiment E-M1 (H7)**.
Both outcomes publish. A negative would be a substantive correction to an actively-cited
preprint.

### Strongest fatal objection to MINDIR (unresolved)

> *"Your surviving contribution is a control experiment on a preprint, and your flagship needs
> a scanner you do not have. The contraction phenomenon is already established from two
> directions. What is the discovery?"*

**There is currently no answer that does not require MindStates-7T.** Recorded, not resolved.

### Verdict caveat — 4 dataset families remain UNAUDITED

Generic Object Decoding (Horikawa/Kamitani), **memory-reinstatement**, **working-memory**, and
MEG/EEG imagery datasets were not audited. **Memory-reinstatement and working-memory are the
highest-value remaining audits**: either could supply a *third state* on shared content and
would upgrade this verdict toward `PARTIALLY_PREEMPTED_BUT_FLAGSHIP_SURVIVES`. **The verdict
is provisional pending those two audits.**

---

## (superseded) GATE M0 VERDICT: `PROSPECTIVE_PROGRAM_REQUIRED`

**The retrospective flagship is pre-empted. The surviving claims are untestable on public data.**

| Dimension | Verdict |
|---|---|
| **Conceptual novelty** | **LARGELY PRE-EMPTED.** Roy, Breedlove, St-Yves, Kay & Naselaris (2025) introduce *"the imagery transformation — a mapping from visual to imagery activity patterns evoked by the same stimulus"*, estimate it per visual area on **two 7T datasets**, and **predict held-out imagery activity**. That is MINDCOMPILER's core move, our E2 and our E8, from the field's leading lab, on our own substrate. |
| **Methodological novelty** | **PARTIAL.** Composition, cross-subject transport *of the operator itself*, and calibrated probabilistic target-state prediction were not found. `S_p` (hyperalignment/SRM) and `T_s` machinery (neural/Koopman operators, incl. compositional variants) are mature and **not ours to claim**. |
| **Public-data feasibility** | **INSUFFICIENT for the flagship.** No public dataset has **≥3 mental states on the same content at trial level** — composition, non-commutativity and interpolation are therefore **unmeasurable**. NSD-Imagery is 4 subjects / 18 stimuli, already used by Roy et al. |
| **Prospective necessity** | **YES.** Every surviving claim requires MindStates-class acquisition. |
| **A\* potential** | **REDUCED, not zero** — and now conditional on new measurements, not on modelling. |

**`PROSPECTIVE_PROGRAM_REQUIRED` is not termination** (mission §12). The platform and
retrospective tests may proceed; the *flagship claim* needs new data.

### The finding that decides it — and it is scientific, not bibliographic

Roy et al. measured what the perception→imagery transformation *is*: in early visual cortex it
**halves the active dimensions and reorients them**; reconstructions explain only **25–50% of
variance**; imagery occupies a **distinct subspace**.

1. **The map is strongly non-invertible.** `T_{b→a}∘T_{a→b} ≈ I` is **contradicted by
   measurement**, not untested. Our inverse test has a published answer: *no*.
2. **Composition compounds the loss.** A path routed through imagery is degraded by
   construction; a composed path beating a direct one is *a priori* implausible here.
3. **H1 (additive state offset) is close to dead**; H2/H3's "shared manifold" is weakened —
   imagery is a *different subspace*, not a rescaled one.

With **Spera et al. 2026** (zero-shot perception→imagery **at chance**, CLIP 48.94% vs 50%),
the picture is coherent: perception→imagery is **lossy, dimension-halving, subspace-shifting**
— learnable within-subject with paired data, carrying **no zero-shot transfer**.

> **Do not retrofit an algebra to this.** The published measurements point away from
> composability and invertibility. That is the answer arriving early and cheaply — which is
> what Gate M0 is for.

### Mandatory next action before *any* redesign

**Read Roy et al. in full.** bioRxiv returned **HTTP 403**; only abstract + PubMed extraction
were obtained. Their composition / inversion / cross-subject / semantic-control status is
**UNCONFIRMED** — "not mentioned" is not "not done". If they tested composition, the last
survivors die too. **Zero cost, maximal information.**

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
