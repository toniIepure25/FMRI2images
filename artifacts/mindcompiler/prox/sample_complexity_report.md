# Sample-complexity report (MINDIR-PROX / P3) — SYNTHETIC

Simulation-based minimum observations n* to reach reliable recovery (subspace overlap >= 0.80) of a private correction, by rank and SNR. Method-feasibility only; NOT a biological law.

| private_rank | SNR | n* (reliable) |
|---|---|---|
| 1 | 1.0 | None |
| 1 | 2.0 | None |
| 1 | 4.0 | None |
| 2 | 1.0 | None |
| 2 | 2.0 | None |
| 2 | 4.0 | None |
| 3 | 1.0 | None |
| 3 | 2.0 | None |
| 3 | 4.0 | None |

**Expected scaling:** n* increases with private rank and decreases with SNR (see rows).

## Phase transitions (n_obs=8)
| rank | SNR | overlap | regime |
|---|---|---|---|
| 1 | 0.5 | 0.016 | UNRECOVERABLE |
| 1 | 1.0 | 0.113 | UNRECOVERABLE |
| 1 | 2.0 | 0.105 | UNRECOVERABLE |
| 1 | 4.0 | 0.700 | PARTIALLY_RECOVERABLE |
| 2 | 0.5 | 0.142 | UNRECOVERABLE |
| 2 | 1.0 | 0.060 | UNRECOVERABLE |
| 2 | 2.0 | 0.304 | UNRECOVERABLE |
| 2 | 4.0 | 0.630 | PARTIALLY_RECOVERABLE |
| 3 | 0.5 | 0.113 | UNRECOVERABLE |
| 3 | 1.0 | 0.193 | UNRECOVERABLE |
| 3 | 2.0 | 0.433 | UNRECOVERABLE |
| 3 | 4.0 | 0.592 | PARTIALLY_RECOVERABLE |
| 4 | 0.5 | 0.125 | UNRECOVERABLE |
| 4 | 1.0 | 0.269 | UNRECOVERABLE |
| 4 | 2.0 | 0.296 | UNRECOVERABLE |
| 4 | 4.0 | 0.592 | PARTIALLY_RECOVERABLE |

Regimes: UNRECOVERABLE < 0.50 <= PARTIALLY_RECOVERABLE < 0.80 <= RELIABLY_RECOVERABLE.