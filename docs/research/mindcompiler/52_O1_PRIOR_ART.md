# O1 — Targeted prior-art audit

**Overall novelty label: `NOVELTY_CANDIDATE`.** No "first ever" claim. The specific O1 combination —
a *within-subject*, *stimulus-identity-held-out*, perception-state→imagery-state linear **operator**,
characterized by an O0–O4 operator-class ladder, with simple↔naturalistic family transfer and
supported-span operator geometry — has no exact precedent found; but several component techniques have
close prior art (recorded honestly below).

| work | question | input→output states | identities held out | transformation generalizes to unseen identity | operator structure analyzed | cross-subject shared operator | label |
|---|---|---|---|---|---|---|---|
| **Roy et al. 2025** (bioRxiv, PMC12424947) | vision→imagery transformation | vision→imagery (fMRI) | **No** (same-identity repeats) | not tested | reduced-rank dim/alignment | no | CLOSE_PRIOR_ART_EXISTS |
| **General Transformations of Object Representations** (J Neurosci 2018, PubMed 30126975) | affine object-view transformation matrices | pre-change→post-change patterns | **Yes** (held-out objects) | **Yes** (affine view changes, not perception→imagery) | transformation matrix, not a model-class ladder | no | CLOSE_PRIOR_ART_EXISTS |
| **MIRAGE** (Kneeland et al. 2025, PLOS Comput Biol) | seen→imagined image reconstruction | fMRI→**image** (decoding) | Yes (NSD-Imagery) | reconstruction generalizes, not a neural-state operator | no (diffusion decoder) | no | CLOSE_PRIOR_ART_EXISTS |
| **Hyperalignment** (Haxby/Guntupalli) & **Shared Response Model** (Chen et al.) | cross-subject common representational space | responses→common space | n/a | n/a | Procrustes / joint-SVD alignment | **yes** (different question) | RELATED_DIFFERENT_QUESTION |

## Positioning
- **Roy 2025** established the vision→imagery transformation on this dataset but evaluated same-identity
  repeats; Track R reproduced it partially. O1's novelty is the **identity-held-out generalization** of
  the transformation itself.
- **"General Transformations of Object Representations" (2018)** is the closest methodological analogue:
  L2-regularized transformation matrices in neural-pattern space that generalize to held-out objects —
  but for *affine view changes within vision*, not perception→imagery state transitions, and without an
  operator-class ladder.
- **MIRAGE (2025)** demonstrates seen→imagined generalization but the output is reconstructed *images*
  via a diffusion decoder, not a characterized neural-state→neural-state operator.
- **Hyperalignment / SRM** address *cross-subject* common spaces — a different question. O1 is strictly
  within-subject; cross-subject shared operators are deferred to O2 and must not be conflated here.

## Conclusion
`NOVELTY_CANDIDATE` for the overall O1 question; component techniques (held-out neural transformation
operators; vision→imagery mapping) each have close prior art. Novelty is **not** claimed as "first."

**Sources:** Roy et al. (PMC12424947 / PubMed 40950062); *General Transformations of Object
Representations in Human Visual Cortex*, J Neurosci 38(40):8526 (PubMed 30126975); MIRAGE, PLOS Comput
Biol 2025 (10.1371/journal.pcbi.1014263); Hybrid Hyperalignment (bioRxiv 2020.11.25.398883); Shared
Response Model (Chen et al. NeurIPS 2015).
