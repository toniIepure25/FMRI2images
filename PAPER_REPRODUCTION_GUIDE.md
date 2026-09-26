# Paper reproduction guide

How to reproduce each MINDIR manuscript figure/table. The discovery manuscript lives in `paper/mindir/`; its
numbers trace to sealed artifacts (see `paper/mindir/evidence_table.csv`). Historical N=8 analyses are **closed**:
their sealed result artifacts are provided and **verified by hash**, not re-run (re-running is neither needed nor
permitted for new discovery). Platform/method figures are reproducible from synthetic data now.

## Reproduction classes
- **EXACT_REPRODUCTION** — deterministic; byte/near-byte identical (synthetic platform outputs with fixed seeds).
- **METHOD_REPRODUCTION** — same pipeline, independently re-derived numbers (historical seals; re-derivation
  would require the sealed inputs on the compute PVC).
- **HASH_VERIFIED_SEAL** — the sealed artifact is checked against its committed hash (historical N=8 results).

| Figure/Table | Command / source | Inputs | Expected artifact | Checksum | Runtime |
|---|---|---|---|---|---|
| Manuscript numbers (all) | `paper/mindir/evidence_table.csv` | sealed gate artifacts | evidence table rows | per-gate `hashes.json` | seconds (read-only) |
| F2 support ceiling vs oracle | sealed O2.5/O2.6 artifacts | closed cohort | `artifacts/.../operator_o2_*` | committed hashes | HASH_VERIFIED_SEAL |
| F4 calibration grid (M4T2) | sealed O2.9 `roi_resource_frontier.csv` | closed cohort | O2.9 artifacts | committed hash | HASH_VERIFIED_SEAL |
| F5 schedule fragility + Q_OUT | sealed O2.14/O2.15 artifacts | closed cohort | O2.14/O2.15 | committed hash | HASH_VERIFIED_SEAL |
| F6 mechanistic (X1–X4) | sealed X1–X4 `results/` | closed cohort | X-gate artifacts | committed hash | HASH_VERIFIED_SEAL |
| Platform demo figure | `mindir demo --out out/` | synthetic (fixed seed) | `out/demo_report.json` | deterministic | seconds (EXACT) |
| Benchmark table | `mindir benchmark synthetic` | synthetic (fixed seeds) | benchmark JSON | deterministic | seconds (EXACT) |
| Phase diagram | `mindir phase` | synthetic | phase JSON | deterministic | seconds (EXACT) |

## No hidden steps
Every row above is a single documented command or a hash-verified sealed artifact. There are no manual,
undocumented, or shell-history-dependent steps. Independent replication of the *science* (not the software)
requires the new O2.16 cohort — that is the point of the program, not a reproduction gap.
