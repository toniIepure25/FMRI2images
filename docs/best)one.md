Fusion appendix from logs/v65_v66/cross_architecture_fusion_20260515.log (best singles and best triple ~86.3% @ k=3).
Headline numbers (from that run):

Best single model (CSLS k=3): V61a_mctta16 → 83.1% CSLS R@1
Best pairwise fusion (approx.): up to ~85.6% CSLS R@1 (see log lines Best: w=...)
Best triple fusion seen in the tail: ~86.3% CSLS R@1 @ k=3
Example line: V61a_mctta16 (0.7) + V62a (0.1) + V66a (0.2) → CSLS R@1 = 0.863
So fusion improves over V61a alone, but does not reach 90% with the current checkpoints (V66b is weak on shared1000, V65a is below V61a there).



Not as a single saved “one model” number. In the pod snapshot doc, the strongest single run on shared1000 (from shared1000_metrics.json) is V61a at 79.1% CSLS R@1 (197K). V64a is close at 78.6%.