# MINDIR failure map (when NOT to trust MINDIR)

Synthetic-derived; states conditions under which metrics fail and whether real-data analysis should abort.

| condition | affected | failure | detection | mitigation | abort real analysis? |
|---|---|---|---|---|---|
| very low SNR | all geometry/recovery | recovery collapses | few-shot recovery < 0.5; wide bootstrap CI | more observations / higher field | True |
| private rank >> observations | private-rank + orientation | unidentifiable | phase diagram UNRECOVERABLE; rank estimate unstable | more repeats | True |
| rank mismatch (est != true) | grassmann/chordal distances | unstable distance | distance metrics disagree with overlap ordering | use subspace_overlap; fix rank | False |
| heavy-tailed noise | pattern_correlation | outlier-driven inflation | rank-based metric disagrees with correlation | use cosine/rank metrics | False |
| session/site drift | cross-session/-site transfer | false low transfer | within-session >> cross-session; site as covariate | drift correction; E6/E10 | False |
| near rank-deficiency | all geometry | numerical instability | tiny singular values; condition number high | QR guard; report conditioning | True |
| low support overlap | support fraction | residual dominated by noise | support_fraction near 0 with low reliability | re-check representation (E10) | False |