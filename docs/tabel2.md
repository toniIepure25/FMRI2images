| Experiment                       | Val R@1 | Val CSLS R@1 | Shared1000 R@1 | Shared1000 CSLS R@1 | clip_dim | Notes                                                                     |
| -------------------------------- | ------: | -----------: | -------------: | ------------------: | -------: | ------------------------------------------------------------------------- |
| N1v28a_dual_head                 |       — |            — |          52.9% |               70.1% |   197376 | Legacy dual-head baseline.                                                |
| V35_legacy_teacher_distill       |   44.9% |            — |          42.1% |               51.8% |      768 | Checkpoint metric in summary is fused_r@1, not CSLS.                      |
| V55d_oof_fold_0                  |   64.6% |        84.7% |              — |                   — |        — | OOF fold; no shared1000 export on pod.                                    |
| V55d_oof_fold_1                  |   66.5% |        85.7% |              — |                   — |        — | OOF fold.                                                                 |
| V55d_oof_fold_2                  |   67.4% |        86.6% |              — |                   — |        — | OOF fold; best_val_loss NaN in JSON.                                      |
| V55d_oof_fold_3                  |   69.7% |        85.4% |              — |                   — |        — | OOF fold.                                                                 |
| V55d_oof_fold_4                  |   68.5% |        84.0% |              — |                   — |        — | OOF fold.                                                                 |
| V56a_fusion_distill_fixed        |   35.6% |        39.7% |          30.8% |               35.6% |      768 | Fusion/distill line; weak on shared1000.                                  |
| V56c_projection_rdrop            |   28.6% |        34.9% |          22.6% |               29.1% |      768 | Projection + R-Drop experiment.                                           |
| V57a_roi_transformer_dual_head   |   27.1% |        32.3% |          36.3% |               44.8% |      768 | ROI Transformer dual-head.                                                |
| V58a_improved_197k               |   45.2% |        57.9% |          47.5% |               61.1% |   197376 | Improved 197K pipeline.                                                   |
| V59a_retrieval_only_768d         |   41.0% |        47.1% |          35.6% |               40.3% |      768 | Retrieval-only 768-D.                                                     |
| V60a_cross_subject_197k          |   38.2% |        43.2% |          34.2% |               42.1% |      768 | 4-subject pretrain; 768-D CLS on pod export.                              |
| V60b_subj01_finetune_197k        |   91.6% |        96.3% |          59.0% |               73.6% |   197376 | Large val↔shared1000 gap; decoder / cross-phase issue.                    |
| V60c_kappa_gated_197k            |   36.9% |        41.2% |          34.8% |               41.8% |      768 | Kappa-gated branch; 768-D export.                                         |
| V60d_subj01_finetune_kappa       |   93.2% |        97.0% |          56.7% |               71.7% |   197376 | Kappa-gated finetune 197K.                                                |
| V61a_finetune_difflr             |   95.3% |        98.6% |          66.0% |               79.1% |   197376 | Strongest single 197K shared1000 in this table; has mctta16 preds on pod. |
| V62a_cls_retrieval_768d          |   94.7% |        96.1% |          39.9% |               48.3% |      768 | Pure CLS; good for cross-space fusion with 197K.                          |
| V62b_triple_head_768d_197k       |   94.7% |        96.1% |          38.7% |               46.7% |      768 | Dual-head; headline compact space.                                        |
| V63a_strong_197k                 |   91.9% |        96.9% |          59.2% |               75.4% |   197376 | Stronger 197K recipe; below V61a on shared1000.                           |
| V63b_cls_from_v61a               |   94.4% |        95.7% |          39.4% |               46.0% |      768 | CLS from V61a encoder.                                                    |
| V64a_continued_v61a              |   95.8% |        97.8% |          70.6% |               78.6% |   197376 | Continued V61a training.                                                  |
| V64b_frozen_encoder_deep_decoder |   91.8% |        96.9% |          57.3% |               73.7% |   197376 | Frozen encoder + deeper decoder.                                          |
| V65a_mlp_v9_recipe               |   89.4% |        96.4% |          58.3% |               72.5% |   197376 | V9 loss stack; below V61a on shared1000 despite higher val.               |
| V66a_roi_pretrain                |   22.3% |        23.3% |          35.5% |               39.8% |      768 | Multi-subject ROI pretrain; 768-D.                                        |
| V66b_roi_finetune_197k           |   74.8% |        86.2% |          38.1% |               50.2% |   197376 | ROI→197K finetune; weak shared1000 vs V61a.                               |
