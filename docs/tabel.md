| Experiment                                   | Result                                                                                         |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| B0                                           | Val loss 1.408; best epoch 2; 17 epochs                                                        |
| B1                                           | Val loss 1.018; best epoch 1; 16 epochs                                                        |
| N1                                           | Val loss 1.366; best epoch 7; 22 epochs                                                        |
| N2                                           | Val loss -171.3; best epoch 5; 25 epochs                                                       |
| N3                                           | Val loss 3.508; best epoch 2; 22 epochs                                                        |
| N4                                           | Val loss 3.495; best epoch 2; 22 epochs                                                        |
| B0v2                                         | 0.13% R@1; best epoch 9                                                                        |
| B1v2                                         | 0.23% R@1; best epoch 15                                                                       |
| N1v2                                         | 0.57% R@1; best epoch 28                                                                       |
| N2v2                                         | 0.30% R@1; best epoch 14                                                                       |
| N3v2                                         | 0.33% R@1; best epoch 49                                                                       |
| B0v3                                         | ~0.07% R@1                                                                                     |
| B1v3                                         | ~0.10% R@1                                                                                     |
| N1v3                                         | ~0.10% R@1                                                                                     |
| N2v3                                         | ~0.10% R@1                                                                                     |
| N3v3                                         | ~0.10% R@1                                                                                     |
| N4v3                                         | ~0.10–0.13% R@1                                                                                |
| B0v4                                         | ~20–25% R@1                                                                                    |
| B1v4                                         | ~20–25% R@1                                                                                    |
| N1v4                                         | ~30–36% R@1                                                                                    |
| N2v4                                         | ~30–36% R@1                                                                                    |
| N3v4                                         | ~30–36% R@1                                                                                    |
| N4v4                                         | ~30–36% R@1                                                                                    |
| N1v5                                         | 39.9% R@1                                                                                      |
| N2v5                                         | 35.9% R@1                                                                                      |
| N3v5                                         | 36.4% R@1                                                                                      |
| N4v5                                         | 37.4% R@1                                                                                      |
| N1v6                                         | ~40% R@1                                                                                       |
| N2–N4v6                                      | 35–38% R@1                                                                                     |
| N1v7                                         | 49% R@1                                                                                        |
| N2v7                                         | No valid final result; wiring issues                                                           |
| N3v7                                         | No valid final result; CLIP cache issues                                                       |
| N4v7                                         | No valid final result; CLIP cache issues                                                       |
| N3v8                                         | 50.3% R@1                                                                                      |
| N4v8                                         | 50.8% R@1                                                                                      |
| N1v9                                         | ~48% R@1                                                                                       |
| N3/N4v9                                      | ~51% R@1                                                                                       |
| N1v10                                        | ~48% R@1                                                                                       |
| N4v10                                        | ~50.8–51% R@1                                                                                  |
| N1v11–N4v11                                  | ~51% R@1; no improvement                                                                       |
| N1v12                                        | 43.3% raw R@1; 52.2% CSLS R@1                                                                  |
| N2v12                                        | 42.3% raw R@1; 50.0% CSLS R@1                                                                  |
| N3v12                                        | 25.2% raw R@1; 37.2% CSLS R@1                                                                  |
| N4v12                                        | 27.3% raw R@1; 37.1% CSLS R@1                                                                  |
| N1v13                                        | ~45–46% raw R@1; ~54% CSLS R@1                                                                 |
| N2v13                                        | ~44% raw R@1; ~51–52% CSLS R@1                                                                 |
| N3v13                                        | ~45% raw R@1; ~54–55% CSLS R@1                                                                 |
| N4v13                                        | ~45–46% raw R@1; ~53–54% CSLS R@1                                                              |
| N1v14                                        | ~43% raw R@1; ~52–54% CSLS R@1                                                                 |
| N2v14                                        | ~38–40% raw R@1; ~52–55% CSLS R@1                                                              |
| N3v14                                        | ~40%+ raw R@1; ~52–55% CSLS R@1                                                                |
| N4v14                                        | ~44–45% raw R@1; ~55–58% CSLS R@1                                                              |
| N1v15                                        | 44.0% raw R@1; 51.4% CSLS R@1                                                                  |
| N2v15                                        | 38.9% raw R@1; 45.9% CSLS R@1                                                                  |
| N3v15                                        | No final metric reported                                                                       |
| N4v15                                        | 42.2% raw R@1; 47.2% CSLS R@1                                                                  |
| N1v16–N4v16                                  | ~39% R@1; regression                                                                           |
| N1v17                                        | 47.0% raw R@1; 56.3% CSLS R@1; shared1000 45.2% raw, 52.1% CSLS                                |
| N2v17                                        | 40.3% raw R@1; 46.9% CSLS R@1; shared1000 6.2% raw, 13.3% CSLS                                 |
| N3v17                                        | 40.1% raw R@1; 47.0% CSLS R@1; shared1000 6.1% raw, 12.6% CSLS                                 |
| N4v17                                        | 40.6% raw R@1; 46.3% CSLS R@1; shared1000 6.8% raw, 12.8% CSLS                                 |
| N1v18                                        | 46.0% val raw; 52.3% val CSLS; 39.3% shared1000 raw; 46.8% shared1000 CSLS                     |
| N2v18                                        | 40.6% val raw; 46.0% val CSLS; 30.9% shared1000 raw; 36.0% shared1000 CSLS                     |
| N3v18                                        | 39.6% val raw; 44.9% val CSLS; 33.0% shared1000 raw; 40.1% shared1000 CSLS                     |
| N4v18                                        | 40.4% val raw; 46.2% val CSLS; 34.9% shared1000 raw; 40.7% shared1000 CSLS                     |
| N1v19                                        | 38.1% raw R@1; 49.1% CSLS R@1; shared1000 40.5% raw, 48.3% CSLS                                |
| N2v19                                        | 36.8% raw R@1; 44.7% CSLS R@1; shared1000 36.2% raw, 41.8% CSLS                                |
| N3v19                                        | 36.6% raw R@1; 45.3% CSLS R@1; shared1000 35.3% raw, 41.7% CSLS                                |
| N4v19                                        | 36.9% raw R@1; 44.6% CSLS R@1; shared1000 37.3% raw, 41.8% CSLS                                |
| N1v20                                        | Not evaluated                                                                                  |
| N2v20                                        | Not evaluated                                                                                  |
| N3v20                                        | Not evaluated                                                                                  |
| N4v20                                        | Not evaluated                                                                                  |
| N1v21                                        | 33.2% val raw; 47.2% val CSLS; 34.0% shared1000 raw; 44.0% shared1000 CSLS                     |
| N2v21                                        | 0.1% val raw; 0.1% val CSLS; 0.1% shared1000 raw; 0.1% shared1000 CSLS                         |
| N3v21                                        | 0.1% val raw; 0.1% val CSLS; 0.1% shared1000 raw; 0.1% shared1000 CSLS                         |
| N4v21                                        | 0.1% val raw; 0.1% val CSLS; 0.1% shared1000 raw; 0.1% shared1000 CSLS                         |
| N1v22                                        | 38.4% raw R@1; 49.9% CSLS R@1                                                                  |
| N1v22b                                       | Regression test; no better result reported                                                     |
| N1v23d                                       | 47.0% raw R@1; 56.3% CSLS R@1; shared1000 45.5% raw, 53.4% CSLS                                |
| N1v23b                                       | 43.4% raw R@1; 57.0% CSLS R@1; shared1000 45.7% raw, 55.8% CSLS                                |
| N1v23c                                       | 38.4% raw R@1; 49.9% CSLS R@1                                                                  |
| N1v23a                                       | 47.6% raw R@1; 59.0% CSLS R@1 at epoch 51                                                      |
| N1v24                                        | 55.4% CSLS R@1                                                                                 |
| N1v24b                                       | 57.6% CSLS R@1                                                                                 |
| N1v24c                                       | 57.2% CSLS R@1                                                                                 |
| N1v24d                                       | 55.4% CSLS R@1                                                                                 |
| N1v25_rerun                                  | 46.3% val raw; 57.2% val CSLS; 43.7% shared1000 raw; 52.5% shared1000 CSLS                     |
| N1v25a                                       | 45.6% val raw; 54.9% val CSLS; 45.4% shared1000 raw; 50.8% shared1000 CSLS                     |
| N1v25b                                       | Crashed; optimizer `add_param_group` bug                                                       |
| N2v25c                                       | 39.6% val raw; 46.2% val CSLS; 39.8% shared1000 raw; 45.1% shared1000 CSLS                     |
| N1v26a                                       | 52.8% raw R@1; 69.6% CSLS R@1                                                                  |
| N1v26b                                       | Abandoned; CUDA OOM at epoch 2                                                                 |
| N1v26c                                       | 53.5% raw R@1; 69.6% CSLS R@1                                                                  |
| N1v26d                                       | 54.2% raw R@1; 66.8% CSLS R@1                                                                  |
| N1v27a                                       | 54.2% raw R@1; 67.2% CSLS R@1                                                                  |
| N1v28a                                       | 56.0% raw R@1; 70.3% CSLS R@1                                                                  |
| N1v28b                                       | Deprioritized; no final result                                                                 |
| N1v29a first run                             | 45% R@1; best epoch 34; stopped at epoch 64                                                    |
| N1v29a revised                               | Pending/no final result reported                                                               |
| N1v29b                                       | Pending/no final result reported                                                               |
| V30a                                         | Triple-head baseline introduced; no standalone final metric reported                           |
| V30b                                         | Two-stage retrieval diagnostics added; no standalone final metric reported                     |
| V30c                                         | Larger compact retrieval dimension tested; did not resolve bottleneck                          |
| V30d_full VAL                                | 42.1% compact raw; 51.7% compact CSLS; 21.7% rerank-only; 56.3% oracle rerank; 23.0% two-stage |
| V30d_full shared1000                         | 43.8% compact raw; 50.4% compact CSLS; 15.6% rerank-only; 49.2% oracle rerank; 15.9% two-stage |
| V30e VAL                                     | 42.9% compact raw; 52.9% compact CSLS; 25.0% rerank-only; 61.3% oracle rerank; 25.6% two-stage |
| V30e shared1000                              | 41.1% compact raw; 49.2% compact CSLS; 24.2% rerank-only; 62.7% oracle rerank; 25.3% two-stage |
| V30e post-hoc fusion                         | 56.2% VAL fused R@1; 51.5% shared1000 fused R@1                                                |
| V31                                          | 55.0% VAL fused R@1; did not beat V30e fusion                                                  |
| V32 VAL                                      | 44.4% compact raw; 51.7% compact CSLS; 40.4% rerank-only; 58.6% fused                          |
| V32 shared1000                               | 41.6% compact raw; 48.7% compact CSLS; 41.1% rerank-only; 57.0% fused; 74.1% oracle rerank     |
| V33                                          | Shortlist teacher distillation; no separate final metric reported                              |
| V33b                                         | 59.5% shared1000 two-expert fused R@1                                                          |
| V34 tri-expert fusion                        | 76.1% VAL R@1; 75.3% shared1000 R@1                                                            |
| V35                                          | 64.3% shared1000 fused R@1                                                                     |
| V35 + N1v28a fixed tri-fusion                | 77.6% VAL R@1; 77.2% shared1000 R@1                                                            |
| V36                                          | Underperformed V35; did not beat 64.3% shared1000 fused R@1                                    |
| V37 learned tri-fusion gate                  | 89.6% VAL R@1; 67.1% shared1000 R@1                                                            |
| V37 fixed tri-fusion rerun                   | 77.6% VAL R@1; 77.2% shared1000 R@1                                                            |
| V38                                          | Proposed legacy-compact distillation; no final result reported                                 |
| V39 pre-repair reranker                      | 68.3% VAL R@1; 69.9% shared1000 R@1                                                            |
| V39 fixed tri-fusion baseline                | 77.6% VAL R@1; 77.2% shared1000 R@1                                                            |
| V39 repaired reranker                        | 63.4% VAL R@1; 65.5% shared1000 R@1                                                            |
| V40                                          | OOF tri-gate protocol proposed; no final result reported                                       |
| V41                                          | OOF union vMF resolver proposed; no final result reported                                      |
| V42                                          | Multi-hypothesis vMF retrieval proposed; no final result reported                              |
| V44a_all8_scfr_smoke                         | Ran correctly; retrieval remained near-random; negative result                                 |
| V44b_all8_scfr_full                          | Planned/full variant; no final result reported                                                 |
| V45a_all8_fusion_distill_smoke               | Proposed; no final result reported                                                             |
| V45b_all8_fusion_distill_full                | Proposed; no final result reported                                                             |
| V55a                                         | 16.6% shared1000 raw R@1; 21.3% shared1000 CSLS R@1                                            |
| V55b                                         | 17.0% shared1000 raw R@1; 22.5% shared1000 CSLS R@1                                            |
| V55c                                         | Planned fusion distillation phase; no separate result reported                                 |
| V55d                                         | 8105 merged OOF predictions; OK                                                                |
| V55e                                         | 70.1% shared1000 R@1                                                                           |
| V56a                                         | 30.8% shared1000 raw R@1; 35.6% shared1000 CSLS R@1                                            |
| V56c                                         | 22.6% shared1000 raw R@1; 29.1% shared1000 CSLS R@1                                            |
| V57a compact                                 | 36.3% shared1000 raw R@1; 44.8% shared1000 CSLS R@1                                            |
| V57a rich                                    | 0.2% shared1000 raw R@1; 3.5% shared1000 CSLS R@1                                              |
| V57a + N1v28a fusion                         | 78.0% shared1000 R@1 at alpha 0.4; oracle 79.7%                                                |
| V58a                                         | 47.5% shared1000 raw R@1; 61.1% shared1000 CSLS R@1                                            |
| V59a                                         | 35.6% shared1000 raw R@1; 40.3% shared1000 CSLS R@1                                            |
| N1v28a + V57a fusion                         | 78.4% best R@1; 79.7% oracle                                                                   |
| N1v28a + V59a fusion                         | 74.3% best R@1; 77.9% oracle                                                                   |
| V58a + V57a fusion                           | 72.4% best R@1; 74.8% oracle                                                                   |
| N1v28a + V58a fusion                         | 70.1% best R@1; 77.0% oracle                                                                   |
| V58a + N1v28a + V59a fusion                  | 75.2% best R@1                                                                                 |
| 4-expert fusion                              | 78.0% best R@1; 86.4% oracle                                                                   |
| V61a MC-TTA token-space                      | 9.2% cosine R@1; 16.5% CSLS R@1; 37.4% cosine R@5; 58.1% CSLS R@5                              |
| V62a validation full 10k gallery             | 91.78% cosine R@1; 91.89% CSLS R@1; 93.44% cosine R@5; 94.44% CSLS R@5                         |
| V62a shared1000 full 10k gallery             | Metrics not locally extracted                                                                  |
| V62a shared1000 attempted 1k gallery         | 11.6% cosine R@1; 13.4% CSLS R@1; diagnostic/misconfigured                                     |
| V62a shared1000 target-embeddings 1k gallery | Manual cosine R@1 39.9%; expected CSLS R@1 ~48.3%; framework report pending                    |
