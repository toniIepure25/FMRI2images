# O2.3A-RD / O2.4R — Prospective Re-Derivation of the Dense-Perception Orientation State + Calibration Frontier

**Source HEAD:** `f56ab1d` · **O2.3A-RD config SHA256:** `c1b2ddb0…` · **O2.4R config SHA256:** `ad959446…`

Exact replay of the historical O2.3A machinery is **closed** (`O2_4_PROVREC_HISTORICAL_DETAIL_MISSING`;
`HISTORICAL_O2_3A_STATE_NOT_RECONSTRUCTABLY_REPRODUCIBLE`). This authorizes a **new prospective**
re-derivation, `O2.3A-RD`, then conditionally `O2.4R`. These are **new** analyses — they do **not** overwrite
or relabel historical O2.3A / O2.4 / PROVREC, and the historical results remain immutable. Never call the
historical O2.3A result "reproduced".

## Core principle

O2.3A-RD answers essentially the same question — *can dense target-subject perception identify the
subject-specific native orientation of the reliable imagery residual, with ZERO target imagery in fitting?* —
but **every numerically-determining detail is now fully specified, committed, tested, and persisted**, closing
each gap the PROVREC forensics found:

| PROVREC-missing detail | O2.3A-RD prospective specification |
|---|---|
| generator source | committed under `src/…/operator_o2_3a_rd*` (mandated), determinism-tested |
| DetSRM init/iter/convergence/gauge | **new** deterministic SRM: SVD init, `max_iter=200`, `tol=1e-7`, float64, canonical gauge (QR/SVD orientation; per-component sign so the largest-abs loading is positive; order by descending singular value, stable tie) |
| K-selection block partition | `block_id = anchor_order_index mod 5` over the committed 512-anchor order (frozen image→block manifest) |
| rank inner-CV split | per outer fold, 5 balanced identity-pair inner folds: fold *j* holds out sorted-simple[*j*] + sorted-naturalistic[*j*] |
| null RNG | `SHA256("O2.3A-RD|subject|ROI|fold|iter")` → first 16 hex → uint64 → `numpy PCG64`; `G∼N(0,1)^{K×r}` → reduced QR → canonical signs; 100 samples |

Plus exact tie rules (≤1e-12 → smaller K/r), `K∈{2,4,8,16,32,64}`, `r∈{1..6}`, the frozen 512 anchor manifest
(identity list reused; **data** re-acquired from public S3), reuse of O1/O2 B0 ROI voxels (no new selection,
no interpolation), the immutable `O2_2_IMAGERY_RESIDUAL` (small visual span), the six reused outer folds, and
gauge-invariance certification (≤1e-10 or TECHNICAL_FAILURE). Inference: participant `N=8`, sign-flip `2^8`,
Holm over ventral/lateral. RD status ∈ {`CORE_ANCHOR_RD_TARGET_ORIENTATION_IDENTIFIABLE`,
`…_NOT_IDENTIFIABLE`, `CORE_ANCHOR_RD_INCONCLUSIVE`}. Historical numbers are **not** acceptance targets.

## Two-stage preregistration + strict chronology

`O2.4R` (the calibration frontier: `M∈{0,2,4,6,8,10}` identities, balanced subsets 25/100/100/25/1,
orthogonal-Procrustes-only, identity-correspondence null 100 perms, `N=8` sign-flip + Holm over 10 tests,
M_STAR rule, status family with **"coherently positive" defined now**) is frozen **before** any RD outcome —
so the calibration experiment cannot be adapted to the new M=0 baseline. `M=0 ≡ P_ZERO_RD` (certified
`O2_4R_M0_EQUALS_O2_3A_RD` at hash/numerical level); the frontier is evaluated **internally** (RD M=0 → M>0),
never against the unreconstructable historical machinery.

**Access order:** (1) freeze O2.3A-RD; (2) freeze O2.4R; (3) commit+push+verify both; (4) acquire/reconstruct
inputs (public S3 → orchestraiq persistent); (5) fit; (6) persist RD state; (7) seal RD M=0; (8) certify
O2.4R M0; (9) only then open M>0; (10) frontier; (11) inference; (12) seal. No refreeze after M0. Leakage:
outer-test identities never enter SRM/K/rank/template/oracle/null/scaling/Procrustes (programmatic asserts).

## Stage A result — RD sealed (`CORE_ANCHOR_RD_INCONCLUSIVE`)

Executed on orchestraiq r770 under the corrected driver (code `cb6f3ca`, see
`63_O2_3A_RD_PHASE2_IMPLEMENTATION_CORRECTIONS.md`). All 8 subjects assembled (anchor reps 3/3/3);
target-specific K recorded (ventral `{16,16,16,16,16,32,32,16}`, lateral `{8×7, subj08:16}`).

| primary ROI | median true | median null | n_pos /8 | sign-flip p | Holm reject | group oracle recovery |
|---|---|---|---|---|---|---|
| ventral | 0.0670 | 0.0621 | 5 | 0.219 | no | 0.095 |
| lateral | 0.1217 | 0.1171 | 6 | **0.0156** | **yes** | 0.182 |

**Seal: `CORE_ANCHOR_RD_INCONCLUSIVE`.** Lateral shows a Holm-significant true>null effect (3/4 criteria),
but **oracle recovery** (corrected: `R_ZERO_RD/max(R_ORACLE,ε)`, median folds→participants) is far below the
pre-registered `≥0.50` bar in both ROIs (0.10 / 0.18). The native oracle upper bound is itself strong
(`R_ORACLE≈0.72` in lateral), so the imagery-residual orientation *is* present natively but is only weakly
recovered from dense perception with zero target imagery — not cleanly null, not identifiable. Both primary
median effects are positive (`all_primary_median_positive: True`). Historical numbers were **not** acceptance
targets; the historical O2.3A verdict is untouched and not relabelled.

`M=0 ≡ P_ZERO_RD` certified exactly: **`O2_4R_M0_EQUALS_O2_3A_RD`**, `max_dev = 0.0` over 96 cells. Reusable
RD state (114 derived npz, ~3.9 MB, no raw betas) persisted on the PVC; `state_manifest.json` records the
sha256 of each. Artifacts: `artifacts/mindcompiler/operator_o2_3a_rd/phase2/{rd_results,RD_SEAL,state_manifest,
cache_provenance_subj01}.json`, `artifacts/mindcompiler/operator_o2_4r/M0_CERT.json`.

## Status of this document

Stage A (RD fit) is **sealed** and M0 is certified. **Stage B** (the O2.4R M>0 calibration frontier) runs next
— guarded so it refuses without a valid RD seal + M0 certification. `O3` remains `O3_NOT_READY` through both
O2.3A-RD and O2.4R — a calibration frontier is not yet a shared-operator generalization demonstration.
Historical-vs-RD comparison is descriptive only, performed **after** both are sealed; agreement is **not**
required, and any qualitative outcome (same zero-target failure / new support / materially different) is
admissible.
