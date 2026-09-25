# Performance report (MINDIR-PROX)

| workload | time (s) |
|---|---|
| benchmark (24 synthetic subjects) | 0.172 |
| phase diagram (16 cells) | 0.024 |

Scaling: geometry/metric cost is O(obs * dim * rank); benchmark scales ~linearly in subjects. Optimization (if added) must not change numerical results (guarded by property tests).
