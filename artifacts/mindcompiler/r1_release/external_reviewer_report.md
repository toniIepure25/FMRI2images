# R1 external red-team report

**Software Engineer:** entry point + editable install work; console-script PATH is env-specific (CI covers). Actionable errors added. Fix applied: exact sign-flip capped at n<=12 (perf). No blocker.

**Computational Neuroscientist:** platform makes no biological claim; synthetic-only clearly labelled; demo/benchmark are method-feasibility. No overclaim found.

**Statistician:** sensitivity analysis is simulation-based (effect=0 gives ~alpha false-positive, verified); participant is the unit; exact vs MC sign-flip documented. Warns: never read sensitivity as guaranteed power.

**Reproducibility reviewer:** provenance DAG + claim traceability + repro classes (EXACT/METHOD/UNAVAILABLE) present; historical seals are HASH_VERIFIED not re-run. CI gates real. Coverage % delegated to CI (pytest-cov absent locally) - acceptable, not fabricated.

**New lab member:** README_MINDIR + quickstart + data contract + tiny_example enable use without project history. Root README preserved (broader project).

**Malicious leakage tester:** attempted historical-N8 read + protected-outcome read + config mutation + seed nondeterminism + future-trial leak -> ALL caught by the firewall/immutability/determinism guards (bug_injection_report all_bugs_caught=True). No bypass found.
