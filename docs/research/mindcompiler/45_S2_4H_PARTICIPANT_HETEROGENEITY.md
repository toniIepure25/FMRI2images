# 45 — S2.4H: Participant heterogeneity audit (post-hoc, descriptive, no rescue)

**Gate:** S2.4H · **Input commit:** `682a1e0` · **Branch:** `research/mindcompiler-neural-state-operators`.

```
ANALYSIS_CLASS = POST_HOC_DESCRIPTIVE_PARTICIPANT_HETEROGENEITY_AUDIT
S2_3_PRIMARY_RESULT_IMMUTABLE = true
```

The seven S2.3 participant outcomes were already observed. S2.4H characterizes heterogeneity using
**only committed S2.3 derived summaries** — no raw HDF5, no RAW/D1/D1b reruns, no new models, no new
preprocessing/ROI/pairing, no new significance threshold, no participant exclusion, no primary
recomputation. It is **not** a new confirmation attempt and **not** an opportunity to reach p<0.05.

**Frozen and immutable:** `rho = 0.6785714…`, exact one-sided permutation `p = 276/5040 = 0.0547619…`,
Criteria 3/4, **`H_A_CROSS_PARTICIPANT_PARTIAL`**. Forbidden wording: "almost significant",
"effectively significant", "would be significant without subj07", "trend-level". Allowed: *the frozen
cross-participant criterion C was not met.*

## Frozen descriptive questions (before computing new summaries)

H1 subj07 technical integrity · H2 subj07 localized vs participant-wide · H3 global level vs
within-participant spatial coupling · H4 imagery-specific vs general reliability factor · H5 baseline
mapping strength · H6 between- vs within-participant dispersion · H7 ROI consistency · H8 influence
without exclusion · H9 aggregation heterogeneity · H10 measurement-size context. Answered as
`SUPPORTED_/NOT_SUPPORTED_BY_DESCRIPTIVE_PATTERN`/`MIXED`/`INCONCLUSIVE` with exact numbers.

## Results (descriptive)

- **subj07 technical:** `NO_TECHNICAL_ANOMALY_IDENTIFIED` → case `VALID_SCIENTIFIC_HETEROGENEITY`. Its
  B1 imagery-reliability loss is positive in 5/7 ROIs (mechanism-consistent), but it sits in a
  **low-baseline RAW regime** (B0 RAW itself weak/negative), so B1 is not worse (L_perf_RAW<0 in 6/7
  ROIs) — a participant-wide RAW-performance shift, not one ROI, and not a defect. subj07 is **not
  excluded**.
- **Imagery vs vision (H4):** rho(S_rel_img, S_perf)=0.679 (frozen primary, context only),
  rho(S_rel_vis, S_perf)=0.786, rho(S_rel_img, S_rel_vis)=0.393 → `GENERAL_RELIABILITY_FACTOR_PLAUSIBLE`
  (a broader beta-preparation reliability factor spanning vision AND imagery). The primary mechanism
  label is **not** renamed.
- **Baseline (H5):** rho(P_B0, S_perf)=+0.214, rho(P_B1, S_perf)=−0.536 (exploratory moderation only).
- **Influence (H8):** leave-one-participant rho positive 7/7 (0.60–0.71) → direction not dependent on
  any one participant. (Not a significance statement.)
- **Aggregation (H9):** median rho 0.679 vs prespecified mean rho 0.393 — both positive, no sign
  reversal; median is the frozen primary and is not re-chosen.
- **Measurement size (H10):** rho(median_nvox, S_perf)=+0.445 → MIXED; no residualization, primary not
  corrected.

## Interpretation boundary

Allowed: *H-A captured a robust directional component of beta-version sensitivity but does not fully
explain participant heterogeneity; a broader beta-preparation reliability factor (vision + imagery)
remains plausible; subj07 is valid heterogeneity, not a technical failure.* Forbidden: H-A confirmed/
disproven; p=0.0548 significant; remove subj07; vision replaces imagery; GLMdenoise harmful; B0/B1
correct/incorrect; Roy reproduced/failed.

## Next (Path A)

Freeze `BETA_PREPARATION_RELIABILITY_SENSITIVITY_WITH_PARTICIPANT_HETEROGENEITY` → next gate
**`S2.4R` — full cross-participant D1 Roy-pipeline characterization** (both B0 and B1 as sensitivity
branches; no beta selection). Track O (operator research) is not executed here and its
measurement-quality selection is frozen separately.
