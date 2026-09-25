# Repository architecture audit (MINDIR-PROX / P1)

Reviewed as by a computational-neuroscience lab, a reproducibility reviewer, an external collaborator, and an
incoming PhD student. Findings are ranked; the PROX platform package implements the justified improvements
**additively** (sealed gate code and artifacts are NOT moved or rewritten — see `migration_plan.md`).

## CRITICAL
- **C1 — Historical-N8 reopen risk.** Many gate modules can read sealed historical results. *Mitigation:* a
  platform-level `governance.assert_no_historical_access` firewall + tests; the platform package never imports
  historical loaders. (No sealed code changed.)
- **C2 — Outcome-firewall not centralized.** Stage-A/Stage-B separation existed per-gate. *Mitigation:* platform
  governance + moonshot firewall centralize the contract; benchmark/metrics never touch protected outcomes.

## HIGH
- **H1 — Metric definitions scattered / re-implemented per gate** (angles, overlaps, recovery). *Mitigation:*
  one canonical `metrics.REGISTRY` with metadata + versions + property tests.
- **H2 — Anonymous dicts everywhere** (results passed as untyped dicts). *Mitigation:* typed `objects.py` model.
- **H3 — No falsification framework** tying claims to the controls that could kill them. *Mitigation:*
  `falsification.py` battery F1–F10 + `claim_falsification_matrix.csv`.
- **H4 — No synthetic ground-truth benchmark** to validate methods independent of real data. *Mitigation:*
  `synthworld.py` + `benchmark.py` scored against known truth.

## MEDIUM
- **M1 — Scattered scientific constants** across gate configs. *Mitigation:* platform reads frozen upstream
  hashes; new platform values live in one place with semantic hashing (`governance.semantic_hash`).
- **M2 — Fragile absolute paths / heredoc config builders** in scratch scripts (historical). *Mitigation:*
  platform uses package-relative logic; not fixing scratch history.
- **M3 — CLI fragmentation** (per-gate CLIs). *Mitigation:* unified `mindir` namespace (validate/metrics/
  simulate/benchmark/phase/falsify).
- **M4 — Reproducibility metadata inconsistent.** *Mitigation:* `governance.repro_record` contract (data hashes,
  commit, config hash, env, seed, participant set, metric version, timestamp, frozen flag).

## LOW
- **L1 — Root README is project-wide**, not MINDIR-specific. *Mitigation:* added `docs/.../MINDIR_README.md`
  (root README untouched).
- **L2 — Numerical-stability behaviour undocumented.** *Mitigation:* stability tests + `numerical_stability_report.md`.
- **L3 — No performance regression tracking.** *Mitigation:* `performance_report.md` + timing tests.

## Not done deliberately (risk-managed)
- No physical move/rename of sealed gate directories or artifacts (would break provenance/hashes). A migration
  MAP is provided instead; any real move is a future, reviewed, hash-preserving step.
