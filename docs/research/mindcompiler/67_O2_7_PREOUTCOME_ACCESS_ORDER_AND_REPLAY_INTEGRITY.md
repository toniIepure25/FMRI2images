# O2.7 — Pre-Outcome Access-Order & Replay Integrity Correction

**Status label:** `O2_7_PREOUTCOME_ACCESS_ORDER_AND_REPLAY_INTEGRITY_RESTART` (not a scientific failure).

## Immutable references (unchanged)

- O2.7 methodology freeze: commit `a461b53` (config `config_sha256 d61cd6ff`).
- Precision amendment (float64 trial pipeline): commit `adb676b`.
- Trial-state extension seal: commit `d2729eb` — `O2_7_TRIAL_STATE_EXTENSION_CERTIFIED`,
  max replay discrepancy `1.0893250450560998e-06`, 96/96 cells.

No scientific methodology, threshold, T grid, M=2/D=1, null, inference, or T_STAR rule changed.

## What happened

The first frontier run had begun compute. **No reduced-repeat (T<8) scientific outcome was
inspected** (no T1/T2/T4 `R_AUG` / `E_TRIAL` / `TOTAL_RECOVERY` / axis-fidelity / sign-flip p /
Holm / `T_STAR` / status). The only partial outputs were verification/cert JSONs (no scientific
metric); they were quarantined/deleted. A static audit found access-order and replay-completeness
issues. The run was stopped before outcome inspection. Preserved intact: frozen config, the
certified trial-level state extension, sealed O2.6/O2.3A-RD state, raw NSD data.

## Precision-language clarification

The original freeze said "exact T8 replay". Since the precision amendment, the correct claim is
**`FLOAT32_SOURCE_BOUNDED_T8_REPLAY`**: the trial pipeline is float64, but the source imagery-beta
extraction is float32, so agreement with the sealed O2.6 float32-derived centroid path is bounded
by source/arithmetic precision (~1e-6), **not** bitwise identity. The frozen status string
`O2_7_T8_REPLAYS_O2_6_M2_D1` is kept for compatibility, but all artifacts record `tolerance = 1e-5`,
the achieved per-metric discrepancy, and the `FLOAT32_SOURCE_BOUNDED` class — never "exact".

## Fixes (implementation-integrity only)

1. **True two-stage execution.** `--stage t8` computes ONLY T=8 and writes the T8 certification;
   `--stage frontier` computes T∈{1,2,4} and programmatically refuses (`O2_7_ACCESS_ORDER_VIOLATION`,
   exit 3) unless a valid Stage-A T8 certification (`O2_7_T8_REPLAYS_O2_6_M2_D1`) is present.
2. **O2.6 reference provenance.** The T8 replay reference is bound to the committed
   `operator_o2_6/augmentation_results.csv` (`--o2-6-aug-sha256`), requiring exactly 16 `M=2,d=1`
   rows (no missing, no duplicate) and recording path/hash/row counts.
3. **Trial-manifest provenance.** Bound to the committed
   `operator_o2_7/trial_residual_state_manifest.json` (`--trial-manifest-sha256`): artifact
   `trial_residual_state`, gate `O2.7`, `ok==true`, 96 replay records all ok, 96 files; then all 96
   `.npz` hashes/bytes verified.
4. **Semantic trial-state verification.** Each `.npz`: `trial_native.shape==(10,8,V)`, `train_ids`
   element-for-element == sealed cell `train_ids`, family order matches, `voxel_hash` == sealed ROI
   hash; provenance parsed to require exactly 8 trials/identity, canonical repeat positions 0..7, no
   duplicate beta/repeat index. Failure → `O2_7_TRIAL_STATE_SEMANTIC_ALIGNMENT_FAILURE`.
5. **Complete T8 certification.** Requires all 16 reference rows present; compares `R_BASE`,
   `R_AUG`, and `TOTAL_RECOVERY` (per-metric max discrepancy) at tolerance `1e-5`; requires exactly
   150 cells (6 folds × 25 pairs × 1) per participant×ROI.
6. **T8 structural replay certification** — records M=2, D=1, 25 balanced pairs, 6 folds, same
   W_target/U_res/train order, full-8 identity residual mean, same Procrustes/outside-SVD, same
   full-repeat held-out targets — persisted separately.
7. **No T8 null; T8 never in the Holm family.** T=8 is a reference only; the 6-test family is
   exactly {ventral,lateral} × {T1,T2,T4}.
8. **Multiregime status logic (resolved prospectively).** Both ROIs finite → TWO/FOUR/EIGHT total
   trials by `max(T_STAR)`; exactly one ROI finite → `TARGET_STATE_TRIAL_CALIBRATION_MULTIREGIME`;
   neither → `REDUCED_TRIAL_TARGET_STATE_CALIBRATION_NOT_ESTABLISHED`; technical → `O2_7_TRIAL_CALIBRATION_INCONCLUSIVE`.
9. **Exact cell counts.** Per participant×ROI×T the enumerated cells must equal
   6 folds × 25 pairs × {8,28,70} (= 1200/4200/10500), else the cell is non-evaluable; no selective
   averaging of surviving cells; the real run additionally requires exactly 25 balanced pairs.
10. **Tests (A–N)** added on top of the existing synthetic controls; 20 O2.7 tests pass.

## Relaunch order

verify sealed state → verify committed trial manifest → load trial state → run T8 ONLY → produce
T8 numerical + structural certification → require PASS → only then T∈{1,2,4} → participant-first
aggregation → 6-test sign-flip/Holm → T_STAR → seal. Partial outputs from the interrupted job are
not reused. `O3` remains `O3_NOT_READY`.
