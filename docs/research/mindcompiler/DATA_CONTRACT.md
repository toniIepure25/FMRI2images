# MINDIR data contract

Exact expected input schemas (pseudonymous only; no participant identity fields ever). Tiny synthetic examples
are produced by `prox.release.tiny_example()` and the synthetic world generator.

| Object | Fields | Notes |
|---|---|---|
| Subject | `subject_id: str`, `cohort` | cohort in {A_HISTORICAL_N8_CLOSED, B_INDEPENDENT_O2_16, C_CONFIRMATION, SYNTHETIC} |
| Representation | `name, family, dim, frozen` | frozen before outcome access |
| TargetState | `subject_id, state, samples[n_obs x dim]` | state in {imagery, recall, ...} |
| ROI matrix | `float[n_obs x n_voxels]` | pseudonymous; native or pre-frozen representation |
| Trial | `identity_id, repeat_index, run, session, site` | frozen counts per O2.16 design |

Validation (`release.validate_target_state`) raises actionable errors on shape/rank problems
(`ShapeError`, `RankDeficiencyError`). No input requires reverse-engineering old scripts; the contract +
`tiny_example()` are sufficient.
