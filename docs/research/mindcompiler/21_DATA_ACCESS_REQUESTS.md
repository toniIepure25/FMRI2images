# 21 — Data Access Requests (DRAFTS — NOT SENT)

**Status: NOTHING HAS BEEN SENT.** Sending correspondence to external researchers is outside
autonomous authority (mission §21) and requires the user's explicit action. These are drafts
for the user to review, edit, and send under their own name and affiliation.

**No access is claimed.** All dependent experiments remain blocked.

---

## Request 1 — Oedekoven et al. (2017) · **highest priority**

**Target:** corresponding author, *Reinstatement of memory representations for lifelike events
over the course of a week*, Sci Rep 7 (2017), doi:10.1038/s41598-017-13938-4.
**Why:** the only verified dataset with **three states on identical content** (encoding →
immediate retrieval → 1-week delayed retrieval, 21 participants, 24 videos). NeuroVault 2814
provides group-level t-maps only, which cannot support trial-level transition modelling.

> **Subject:** Data access request — trial-level estimates, Oedekoven et al. (2017) reinstatement study
>
> Dear Dr [NAME],
>
> I am [NAME, ROLE, INSTITUTION]. I am working on computational models of how neural
> representations of the same content are reorganized across cognitive states (perception,
> imagery, working memory, and retrieval).
>
> Your 2017 *Scientific Reports* study is, to my knowledge, one of very few datasets in which
> the **same content is measured in three phases** — encoding, immediate retrieval, and
> retrieval after one week. The NeuroVault collection (2814) provides group-level t-maps,
> which are not sufficient for the trial-level modelling I would like to attempt. The paper
> notes that data are available from the corresponding author on reasonable request, so I am
> writing to ask whether the following could be shared:
>
> 1. trial- or event-level beta estimates per participant, if these exist;
> 2. preprocessed fMRI (or the preprocessing pipeline and parameters, if raw is easier);
> 3. event timing files and video/content identifiers linking the three phases;
> 4. participant-level behavioural memory scores;
> 5. ROI or anatomical metadata used in the published analyses;
> 6. the terms under which any reuse and publication would be permitted, including how you
>    would wish to be credited or involved.
>
> To be explicit about intent: I would be testing whether the encoding→immediate→delayed
> transformation can be predictively factorized — specifically whether immediate retrieval
> mediates what is retained after a week — and whether any such structure is shared across
> participants. This is a re-analysis, not a replication attempt, and I would expect to
> discuss authorship or acknowledgement with you before any submission.
>
> I am happy to sign a data-use agreement, work under any constraints you prefer, or share
> analysis code and outputs back with you. If the data are not available in a usable form,
> I would be grateful to know that too, so I can plan accordingly.
>
> Thank you for considering this.
>
> [SIGNATURE BLOCK]

## Request 2 — Li et al. (2023) · imagery / perception / illusion

**Target:** corresponding author, *Neural Representations in Visual and Parietal Cortex
Differentiate between Imagined, Perceived, and Illusory Experiences*.
**Send only if** no OSF/repository release is found — **that search has not been completed**,
so this request may be unnecessary.

> **Subject:** Data availability — imagined / perceived / illusory representations study
>
> Dear Dr [NAME],
>
> I am [NAME, ROLE, INSTITUTION], working on models of how the same content is represented
> across internally and externally generated perceptual states.
>
> Your study is unusual in measuring **matched orientation content under perception, imagery,
> and illusion**, which makes it valuable for testing whether different internally generated
> states reorganize information in different ways. I could not locate a public release of the
> neural data; if one exists, I would be grateful for a pointer.
>
> If not, would you be willing to share trial-level or condition-level response estimates,
> event timing, condition labels, behavioural recall errors, and eye-tracking (if available),
> along with your terms for reuse and credit?
>
> I am glad to sign a data-use agreement and to discuss involvement or acknowledgement before
> any submission.
>
> Thank you for your time.
>
> [SIGNATURE BLOCK]

## Standing rules

1. **The user sends these, not me.** They must go under a real name, role and institution.
2. **Do not describe access as pending, likely, or probable.** Until a reply grants it,
   E-M3's status is **BLOCKED**, and the claim registry reflects that.
3. **Authorship/involvement is offered, not assumed** — that decision belongs to the user and
   the data owners, not to me (mission §21).
4. If access is refused or unanswered, **E-M3 moves to Track P** and the flagship's dependence
   on MindStates-7T is unchanged.

---

## Request 3 -- Roy et al. (2025) vis2img/vis2vis code (DRAFT -- NOT SENT)

Target: corresponding author, "A transformation from vision to imagery in the human brain"
(bioRxiv 2025.09.02.672180). Do NOT imply an error has been found. This requests details for
an independent methodological reproduction.

Subject: Independent reproduction of your vision-to-imagery transformation analysis

Dear Dr [NAME],

I am [NAME, ROLE, INSTITUTION], carrying out an independent methodological reproduction of the
imagery-transformation analysis in your 2025 preprint. Your Methods specify most of the
pipeline clearly (98th-percentile SNR voxel selection, the 100-value log-spaced ridge grid, the
4/2/2 repeat split, rank selection near 99% of peak validation), and I would like to reproduce
it faithfully. A few implementation details are not fully determined by the text; any you can
share would help:

1. the analysis code for vis2vis and vis2img, if releasable;
2. the exact fold-level denoising procedure -- whether every denoised vision input to vis2img
   is strictly out-of-fold, and how vis2vis fold selection is ordered relative to vis2img;
3. the original train/validation/test split seeds;
4. the random pairing seeds and number of pairing realizations (and whether averaged);
5. rank- and ridge-selection tie-breaking rules;
6. voxel masks/indices and processed response matrices, if distributable under NSD terms;
7. access information for the second (spatial imagery, 512-condition) dataset;
8. whether you would be comfortable with an independent reproduction being published, and how
   you would wish to be credited or involved.

I am happy to share code and results back, work under any data-use constraints, and discuss
authorship or acknowledgement before any submission.

Thank you for considering this.
[SIGNATURE BLOCK]

Rule: the user sends this under a real identity. Access/clarification is never described as
pending, likely, or received until a reply arrives.
