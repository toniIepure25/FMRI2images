# 27 — Roy S1: Data and Code Inventory, and Feasibility Determination

**Date:** 2026-07-17 · **Session input commit:** `ef931db` (output commit differs; this file
does not name the commit that will contain it).
**No data was downloaded. No reproduction was run.**

---

## FEASIBILITY STATUS (issued per §3, before any download)

### `S1_APPROXIMATE_REPRODUCTION_ONLY` — ceiling
### `ROY_S1_BLOCKED_BY_DATA_ACCESS` — current execution state

Two independent findings, either of which alone prevents starting:

**1. No author code exists (searched, not found).** Roy et al. 2025 is a **bioRxiv preprint,
not peer-reviewed**, and no author or laboratory repository for the imagery-transformation
analysis was located. Searches over the author set and the NSD ecosystem returned only
adjacent projects — `cvnlab/nsddatapaper`, `gifale95/NSD-synthetic`,
`MedARC-AI/fMRI-reconstruction-NSD` — none of which implements `vis2img`.

> **Consequence: exact reproduction is impossible in principle.** The denoising procedure,
> voxel-inclusion criteria, rank grid, pairing procedure, centering, and solver would have to
> be reconstructed **from prose**. Per §6 and §3, a reconstruction from prose is
> **approximate**, and calling it exact would be false. **The ceiling for S1 is
> `S1_APPROXIMATE_REPRODUCTION_ONLY`, permanently, unless the authors release code.**

**2. NSD-Imagery requires registration and a data-use agreement.** Distribution is via
`naturalscenesdataset.org` under terms requiring account registration and acceptance of a DUA;
the NSD-Imagery release is CC-BY-**NC-ND** 4.0. **§4 requires stopping with a precise access
instruction rather than bypassing authentication.** I have not registered, cannot accept a DUA
on the user's behalf, and have downloaded nothing.

## Required artifact classification

| # | Artifact | Class | Note |
|---|---|---|---|
| 1 | Participant list (8 subjects) | **reconstructible** | stated in paper |
| 2 | Visual runs | **after registration** | NSD-Imagery release |
| 3 | Imagery runs | **after registration** | incl. which runs excluded |
| 4 | Stimulus identities (12: 6 simple, 6 naturalistic) | **reconstructible** | described in paper |
| 5 | Repeat/run labels | **after registration** | |
| 6 | **Trial-level response estimates** | **after registration** | the core requirement |
| 7 | ROI definitions | **reconstructible from public NSD** | NSD-standard ROIs; repo already uses them |
| 8 | **Voxel inclusion criteria** | **UNAVAILABLE** | "visually responsive" threshold not fully specified |
| 9 | Preprocessing details | **partially reconstructible** | prose only |
| 10 | **Denoising inputs/outputs** | **UNAVAILABLE** | paper feeds **denoised vis2vis outputs** into vis2img; exact procedure not released |
| 11 | vis2vis pairing procedure | **partially reconstructible** | "different trial, same stimulus" |
| 12 | vis2img pairing procedure | **partially reconstructible** | random within-identity; **seed count/averaging unspecified** |
| 13 | Rank-selection grid | **UNAVAILABLE** | "4-fold cross-validated line search"; grid bounds not given |
| 14 | Validation protocol | **partially reconstructible** | 4-fold, repeat-level |
| 15 | Evaluation metric | **reconstructible** | per-voxel Pearson r |

**Four artifacts are UNAVAILABLE (8, 10, 13, and the pairing-seed protocol in 12).** Items 10
and 13 are the most consequential: the paper's own limitations state they used **denoised
vision trials as vis2img inputs** to boost statistical power, so the denoising is not
incidental — it is upstream of the headline number. Reconstructing it from prose introduces an
uncontrolled degree of freedom precisely where the effect size is determined.

## Execution state verified

| | |
|---|---|
| Branch / HEAD | `research/mindcompiler-neural-state-operators` @ `ef931db`, **server-verified** |
| Working tree | clean but for `docs/CLAUDE_MEGA_PROMPT.md` (user-authored, untracked) |
| Disk (D:) | **74 GB free** of 184 GB — adequate for an imagery-specific subset |
| Existing NSD-Imagery data locally | **none** (only code/docs referencing it) |
| Pod | not contacted this session; **not synchronized**, per the instruction to inspect before syncing |
| Active processes | none started |

## What the user must do to unblock S1

1. Register at **naturalscenesdataset.org** and accept the NSD / NSD-Imagery data-use
   agreement under their own name and institution.
2. Confirm which access route is granted (AWS S3 prefix or portal download) and share the
   route — **not** credentials.
3. Optionally, **email the authors for the vis2img code**. Given items 8/10/13 are
   unavailable, author code is the only route from *approximate* to *exact*. A draft can be
   added to `21_DATA_ACCESS_REQUESTS.md` on request.

**I will not** register an account, accept a DUA, or bypass authentication (§4).

## Interpretation limits carried into S1

Even on success, S1 establishes **only** that the published repeat-level predictive result was
reproduced. It does **not** establish cross-content generalization, identity-template
insufficiency, incremental value beyond features, a neural transformation, a causal mechanism,
or any MINDIR validation. **No S2/S3 verdict is authorized.**

## What was deliberately not done

No synthetic null worlds were added (§ "do not expand"). No 1 000-seed campaign. No pod sync.
No download. No `roy_reproduction/` package was scaffolded — writing an implementation against
a specification with **four unavailable components** would encode guesses as though they were
the published protocol, which is the failure mode this program exists to prevent.
