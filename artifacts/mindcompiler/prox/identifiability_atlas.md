# Identifiability atlas (MINDIR-PROX / P3)

What can and cannot be identified, and under which invariances, for the MINDIR target-state model
`Y_s = P_s + C_s + eps` (perception-supported component + subject-specific correction + variability). Verified on
synthetic worlds with known ground truth; these are method-identifiability statements, not biological claims.

| Quantity | Identifiable? | Under which invariance | Conditions / limits | Falsifier |
|---|---|---|---|---|
| **Support fraction** (energy in perception support) | Yes (given a frozen support basis) | invariant to within-support rotation | representation-dependent; needs the frozen support | F4 support randomization |
| **Outside-support residual** | Yes | within-support rotation invariant | depends on frozen D and support | F4 |
| **Private-correction rank** | Yes, up to noise | orthogonal-basis invariant | recoverable when SNR & obs exceed the phase boundary (see phase diagram) | F6 rank-matched random |
| **Private-correction orientation** | Yes **within subject**; NOT across subjects in raw voxel space | orthogonal-basis invariant within a fixed space | cross-subject only via invariant/donor-mean quantities | F2 participant shuffle |
| **Cross-subject shared scaffold** | Yes, as an invariant overlap | basis-invariant | needs >= shared_rank+1 obs/subject | F6 |
| **Adaptive stopping sufficiency** | Yes, prospectively (subspace convergence) | order-only info | shadow, no future/outcome access | F8 repeat shuffle |
| **Zero-shot subject prediction** | Conditionally | participant-disjoint LOSO | requires descriptor→geometry signal above the participant-shuffle null; N~12 power-limited | F2, F7 |
| **Absolute neural coordinates** | **No** | not basis-identifiable | only invariants are identifiable | F5 rotation |
| **Causal direction / mechanism** | **No** | n/a | out of scope for MINDIR | — |

## Core rule
Only **basis-invariant** quantities (angles, overlaps, distances, ranks, invariant fractions) are treated as
identifiable. Coordinate-dependent quantities are reported as descriptive and never as findings. Cross-subject
comparisons use invariant/donor-mean quantities only.
