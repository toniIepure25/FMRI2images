# O2.16 independent replication — technical brief for MRI facilities (~2 pages)

*Suitable to send to MRI facility staff after a PI expresses interest. No confirmatory scanning is proposed
before ethics and institutional authorization.*

## Scientific purpose (4–5 sentences)
We study how much of the human perception→imagery transformation is shared across people versus
subject-specific, and how little direct imagery data are needed to calibrate a decoder to a person's imagined
neural state. A discovery/development analysis on a public dataset (N = 8) found that a low-dimensional,
subject-specific component of imagery geometry lies outside perception support and can be calibrated from a
small, acquisition-fragile number of imagery measurements. This is a **prospectively frozen, preregistered
independent replication** in a genuinely new cohort — it changes none of the analysis. We are seeking a site to
acquire that independent cohort.

## Acquisition design (frozen)
- **Perception task:** 512 canonical anchor images × 3 presentations = **1536 trials** (image ~3 s + gap ~1 s).
- **Imagery task:** 12 cued identities (6 simple + 6 naturalistic) × 8 repeats = **96 trials** (~4 s each).
- **Cohort:** planned **N = 12**, minimum confirmatory-valid **N ≥ 8**, independent of the original participants.
- **Analysis folds:** 6 outer folds; a fixed minimal-calibration estimator (all frozen; not site-dependent).

## Desired scanner parameters (targets, negotiable with your physicist)
- Whole-brain functional imaging, **~1.8 mm isotropic** target, **TR ~1.6 s** target.
- Structural **T1w ≤ 1 mm** (+ T2w recommended).
- Field strength: **7T preferred where scientifically and operationally justified, not mandatory** — a
  well-controlled 3T acquisition is acceptable; protocol fidelity and stable acquisition matter more than field
  strength alone.

## Experimental software (already built)
A complete, tested experiment stack exists (perception + imagery runners, deterministic randomization with
exact trial replay, BIDS events, scanner-trigger and response abstractions, QC, and a Stage-A/Stage-B analysis
firewall). It runs today in simulation; a real PsychoPy presenter is implemented at software level. It is
mode-guarded: **confirmatory acquisition fails closed** until ethics, consent, site authorization, and stimulus
permissions are all real.

## What needs site validation
- Exact **trigger** signal/code, **display** model + refresh rate, **button box** mapping.
- Final **sequence** parameters (TR/TE/voxel/multiband/distortion correction/dummy volumes).
- **Run/session structure** (we provide balanced partition options for your review).
- **DICOM→BIDS** path and an official `bids-validator` run on dry-run data.
- A short **hardware dry run** (triggers, frame timing, response mapping) — no participant needed.

## Pilot requirement
One engineering pilot (≤ 2 participants, explicitly **non-confirmatory**, never pooled with the confirmatory
cohort) to validate operations. Neural effect size is **not** a go/no-go criterion.

## Commitments
No confirmatory participant is scanned before: site engineering certification, a completed pilot, ethics active,
consent active, stimulus permissions active, and PI authorization. We claim no access or validation until it has
actually occurred.
