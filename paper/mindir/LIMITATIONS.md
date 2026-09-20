# Limitations

Mandatory, prominent, and non-negotiable. None of these is mitigated by the exploratory analyses.

1. **Small sample.** N = 8 participants. The inferential unit is the participant; the minimum attainable
   one-sided sign-flip p is 1/256 ≈ 0.0039. All conclusions are correspondingly limited in power and precision.
2. **Discovery / development cohort.** These eight participants were used to *develop and discover* the
   findings. They are not an independent test set.
3. **Repeated use of the same cohort across exploratory gates.** X1–X4 (and the O2 series) reuse the same
   eight participants. Although each gate was prospectively frozen, cumulative exploratory testing on one
   cohort inflates the risk of cohort-specific structure; this is why replication is preregistered.
4. **fMRI spatial/temporal limits.** BOLD spatial resolution, temporal smoothing, and single-trial noise bound
   what any geometric analysis can resolve.
5. **Estimator-specific calibration minimum.** The eight-observation frontier is specific to the frozen
   direct-SVD + two-dimensional outside estimator. A different estimator could yield a different minimum; we do
   not claim eight is universal.
6. **ROI-specific conclusions.** All primary conclusions are for the ventral and lateral visual streams;
   parietal was reserved and is not a primary claim.
7. **Absence of independent replication.** The independent cohort (O2.16) is preregistered and frozen but not
   yet executed. No independent participant has been acquired.
8. **No causal / computational inference.** We characterise geometry and prediction, not mechanism or
   causation. "Drift", "mode", and "state" are mathematical-model quantities, not biological claims.
9. **Unavailable original code/seeds for a bitwise Roy replication.** The project began as an independent
   method reproduction/extension; `ORIGINAL_CODE_REPRODUCTION_UNAVAILABLE` and
   `BITWISE_REPLICATION_UNAVAILABLE_WITHOUT_ORIGINAL_CODE_AND_SEEDS` are preserved. We do not claim an exact
   reproduction of Roy et al.
10. **Bounded trial-level exploratory analyses.** Some X2/X3 trial-level analyses are necessarily bounded by
    the available repeat-level identity data (finite repeats per identity), which limits the resolution of
    latent-state and angular-stability estimates.
11. **Imagery-task provenance partly unresolved.** Full run/session layout and naturalistic-identity details
    for the imagery task retain `O2_16_DATA_IMAGERY_TASK_PROVENANCE_BLOCKER`; the replication task spec is
    certified only at the field level established from sealed provenance.
12. **Acquisition-schedule fragility.** The eight-observation frontier is an average result; a substantial
    minority of schedules failed the operational criterion (coverage 0.656 ventral / 0.579 lateral), so the
    result is a group-average frontier, not a per-schedule guarantee.
13. **Task specificity.** All effects are within the NSD imagery task and its identity set; we cannot claim
    they generalise to other imagery tasks, stimulus classes, or instructions.
14. **No physical cause assigned to repeat geometry.** We characterise which geometric quantity predicts
    calibration failure (Q_OUT) but assign it no physiological, attentional, or temporal cause; "drift",
    "mode", and "state" are mathematical-model quantities only.
