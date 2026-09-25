# Migration plan (target architecture map — no blind reorg)

The target layered architecture is realized as the **new** `prox` platform package plus a mapping of existing
(sealed) code to layers. **No sealed gate module or artifact is moved or renamed in this task** — doing so would
break provenance hashes and server-verified seals. Physical migration, if ever done, is a separate reviewed,
hash-preserving step.

| Target layer | New platform module (this task) | Existing sealed code that logically belongs here (NOT moved) |
|---|---|---|
| `core/` | `prox/objects.py` | geometry.py, frontier.py (operator_o2_3a_rd) |
| `geometry/` | `prox/metrics.py` (geometry family) | x1_gauge_invariants, x3/x4 geometry, o2_10/o2_11 |
| `calibration/` | (metrics CALIBRATION family) | operator_o2_9, o2_12, o2_14, schedule_robustness |
| `prediction/` | benchmark B6 baselines | moonshot_triad/m2_zeroshot |
| `cross_state/` | benchmark B8 | moonshot_triad/m3_crossstate |
| `experiment/` | (unchanged) | o2_16_experiment (+ site-prep adapters) |
| `analysis/` | (unchanged, frozen) | operator_o2_16 (primary), o2_16_secondary_prereg |
| `metrics/` | `prox/metrics.py` | scattered per-gate metric code (now superseded by the registry) |
| `statistics/` | `prox/statistics.py` | frontier.signflip/holm (o2_3a_rd) |
| `simulation/` | `prox/synthworld.py`, `prox/sample_complexity.py` | per-gate synthetic fixtures |
| `benchmark/` | `prox/benchmark.py` | (new) |
| `governance/` | `prox/governance.py` | freeze/provenance conventions across gates |
| `cli/` | `prox/cli.py` (`mindir`) | per-gate CLIs (o2_16_experiment.cli, moonshot cli) |
| `tests/` | `tests/mindcompiler/prox/` | existing gate tests (kept) |
| `docs/` | `docs/research/mindcompiler/` | existing docs (kept) |

## Rules for any future physical migration
1. Preserve every sealed artifact hash; if a file must move, record old→new path + unchanged sha in a migration
   manifest.
2. Never edit a frozen `*_frozen_config.json` (their config_sha256 is load-bearing).
3. Re-run the full test + immutability suite and server-verify after any move.
4. Historical N=8 code stays read-only and behind the firewall regardless of location.
